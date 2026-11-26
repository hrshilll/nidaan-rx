"""
Nidaan-Rx v2.0 — Milestone 1: Prescription OCR + PMBJP Jan Aushadhi Scanner.
Combines PaddleOCR 3.x text extraction, rapidfuzz brand/salt resolution against
1mg (13,432 SKUs) and PMBJP (160 SKUs), v1.0 Chemical Equivalence Gates (G1-G3),
and v2.0 Patient Safety Checks (G6 Pregnancy/Lactation, G7 Allergy, G11 DDI + Cross-Rx DDI).
"""
from __future__ import annotations

import ast
import io
import json
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from rapidfuzz import fuzz, process

from app.core.knowledge_graph import CONDITION_GRAPH, MedicineCatalog, _parse_entities, _parse_price
from app.core.safety_gates import CURATED_DDI_PAIRS, _norm_strength, _vehicles_compatible
from app.core.schemas import PatientProfile, SaltEntity

PROJECT_ROOT = Path(__file__).resolve().parents[3]

STRENGTH_RE = re.compile(
    r"(?<!\w)(\d+(?:\.\d+)?)\s*(%|mg|mcg|g|gm|ml|iu)(?!\w)(?:\s*(?:w/w|w/v|v/v))?",
    re.IGNORECASE,
)

VEHICLE_PATTERNS: list[tuple[str, list[str]]] = [
    ("SHAMPOO", ["shampoo", "scalp wash"]),
    ("POWDER", ["powder", "dusting"]),
    ("OINTMENT", ["ointment", "oint"]),
    ("CREAM", ["cream", "crm"]),
    ("GEL", ["gel", "face wash", "facewash", "wash"]),
    ("LOTION", ["lotion", "solution", "soln", "suspension", "drops", "spray", "foam", "serum"]),
    ("SOAP", ["soap", "bar"]),
    ("TABLET/CAPSULE", ["tab", "tabs", "tablet", "tablets", "cap", "caps", "capsule", "capsules"]),
]

NON_DRUG_HINTS = (
    "dr.", "doctor", "mbbs", "clinic", "hospital", "medical centre", "medical center",
    "patient", "name:", "age:", "sex:", "date:", "reg no", "reg.", "phone", "mob",
    "mobile", "address", "diagnosis", "chief complaint", "follow up", "follow-up",
    "signature", "opd", "prescription", "rx only", "advice", "ahmedabad", "gujarat",
    "mumbai", "delhi", "india", "pin:", "email", "www.", "http", "review after",
    "consult", "emergency", "allergies:", "vitals", "bp:", "weight:",
)

PREGNANCY_UNSAFE_SALTS = {
    "isotretinoin", "tretinoin", "adapalene", "tazarotene", "acitretin",
    "doxycycline", "minocycline", "tetracycline", "itraconazole", "fluconazole",
    "terbinafine", "hydroquinone", "methotrexate", "finasteride", "spironolactone",
    "ivermectin", "coal tar", "selenium", "mometasone", "clobetasol",
}

ALLERGY_CLASS_SYNONYMS: dict[str, set[str]] = {
    "penicillin": {"penicillin", "amoxicillin", "ampicillin", "cloxacillin", "piperacillin"},
    "sulfa": {"sulfamethoxazole", "sulfadiazine", "silver sulfadiazine", "dapsone"},
    "azole": {"clotrimazole", "ketoconazole", "miconazole", "luliconazole", "itraconazole", "fluconazole"},
    "tetracycline": {"tetracycline", "doxycycline", "minocycline"},
    "steroid": {"hydrocortisone", "mometasone", "clobetasol", "betamethasone", "fluticasone", "desonide"},
}



SALT_SUFFIX_RE = re.compile(
    r"\b(?:hyclate|monohydrate|dihydrate|hydrochloride|hcl|sodium|potassium|calcium|"
    r"maleate|tartrate|succinate|fumarate|mesylate|besylate|dipropionate|propionate|"
    r"valerate|furoate|acetate|nitrate|sulfate|sulphate|phosphate|gluconate|ip|bp|usp)\b",
    re.IGNORECASE,
)


def _normalize_salt_base(salt: str) -> str:
    cleaned = SALT_SUFFIX_RE.sub(" ", str(salt).lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def detect_vehicle(text: str) -> str | None:
    low = text.lower()
    for veh, kws in VEHICLE_PATTERNS:
        for kw in kws:
            if re.search(rf"\b{re.escape(kw)}\.?\b", low):
                return veh
    return None


def extract_strengths(text: str) -> list[str]:
    out: list[str] = []
    for m in STRENGTH_RE.finditer(text):
        num, unit = m.group(1), m.group(2).lower()
        if unit == "gm":
            unit = "g"
        out.append(f"{num}{unit}")
    return out


def clean_drug_query(line: str) -> str:
    s = re.sub(r"^\s*(?:rx\.?|\d+[\.\)]\s*|[-*]\s*)", "", line, flags=re.I)
    s = STRENGTH_RE.sub(" ", s)
    s = re.sub(r"\bx\s*\d+\s*(?:d(?:ays?)?|w(?:eeks?)?|m(?:onths?)?)?\b", " ", s, flags=re.I)
    s = re.sub(
        r"\b(?:tab|tabs|tablet|tablets|cap|caps|capsule|capsules|crm|cream|oint|ointment|"
        r"gel|lotion|shampoo|soap|powder|dusting|syr|syrup|susp|suspension|soln|solution|"
        r"drops|spray|foam|ip|bp|usp|apply|twice|once|daily|weekly|thinly|morning|night|"
        r"bedtime|before|after|food|meal|meals|for|days|day|weeks|week|months|od|bd|tds|"
        r"qid|hs|sos|stat|topical|external|use|only|affected|area|locally)\b\.?",
        " ",
        s,
        flags=re.I,
    )
    s = re.sub(r"\b\d-\d-\d\b", " ", s)
    s = re.sub(r"[^a-zA-Z0-9\s\+\-/]", " ", s)
    s = re.sub(r"\b\d+\b", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def is_likely_metadata_line(line: str) -> bool:
    low = line.strip().lower()
    if len(low) < 3:
        return True
    if any(h in low for h in NON_DRUG_HINTS):
        return True
    if re.fullmatch(r"[\d\s/\-:\.\+]+", low):
        return True
    if re.search(r"\b\d{10}\b", low):
        return True
    return False


def _decode_image_to_rgb_np(raw_bytes: bytes, max_side: int = 1600) -> np.ndarray:
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    with Image.open(io.BytesIO(raw_bytes)) as src:
        img = ImageOps.exif_transpose(src).convert("RGB")
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return np.array(img)


class PrescriptionScannerService:
    """OCR + Fuzzy Brand/Salt Matcher + PMBJP Equivalence + Patient Safety Gate Checker."""

    def __init__(
        self,
        pmbjp_csv: Path | None = None,
        onemg_csv: Path | None = None,
        catalog: MedicineCatalog | None = None,
    ) -> None:
        self.pmbjp_csv = pmbjp_csv or (PROJECT_ROOT / "data" / "processed" / "pmbjp_dermatology_parsed.csv")
        self.onemg_csv = onemg_csv or (PROJECT_ROOT / "data" / "processed" / "1mg_dermatology_parsed.csv")
        self.catalog = catalog or MedicineCatalog(
            pmbjp_csv=self.pmbjp_csv if self.pmbjp_csv.exists() else None,
            onemg_csv=self.onemg_csv if self.onemg_csv.exists() else None,
        )

        self._ocr_engine: Any = None
        self._brand_records: list[dict[str, Any]] = []
        self._brand_keys: list[str] = []
        self._brand_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._known_salts: list[str] = []

        self._build_indices()

    def _build_indices(self) -> None:
        salt_set: set[str] = set(PREGNANCY_UNSAFE_SALTS)
        for pair_a, pair_b in CURATED_DDI_PAIRS:
            salt_set.add(pair_a)
            salt_set.add(pair_b)
        for ck in CONDITION_GRAPH.values():
            for opt in ck.otc_options + ck.rx_discussion_options:
                for s in opt.salts:
                    base = _normalize_salt_base(s.salt)
                    if len(s.salt) >= 3:
                        salt_set.add(s.salt.lower().strip())
                    if len(base) >= 3:
                        salt_set.add(base)
        for row in self.catalog.pmbjp_rows:
            for s in row["salts"]:
                base = _normalize_salt_base(s.salt)
                if len(s.salt) >= 3:
                    salt_set.add(s.salt)
                if len(base) >= 3:
                    salt_set.add(base)

        if self.onemg_csv.exists():
            df = pd.read_csv(self.onemg_csv)
            for _, row in df.iterrows():
                pname = str(row.get("product_name", "")).strip()
                if not pname or pname.lower() == "nan":
                    continue
                salts = _parse_entities(row.get("parsed_entities"))
                if not salts:
                    continue
                for s in salts:
                    if len(s.salt) >= 3:
                        salt_set.add(s.salt)
                veh = str(row.get("vehicle", "UNKNOWN")).strip().upper()
                price = _parse_price(row.get("product_price"))
                key = clean_drug_query(pname)
                if len(key) < 3:
                    continue
                rec = {
                    "product_name": pname,
                    "brand_key": key,
                    "salts": salts,
                    "vehicle": veh,
                    "price_inr": price,
                }
                self._brand_records.append(rec)
                self._brand_by_key[key].append(rec)

        self._brand_keys = list(self._brand_by_key.keys())
        self._known_salts = sorted(salt_set)

    def _get_ocr(self) -> Any:
        if self._ocr_engine is not None:
            return self._ocr_engine
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import PaddleOCR

        try:
            self._ocr_engine = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                lang="en",
            )
        except TypeError:
            self._ocr_engine = PaddleOCR(lang="en")
        return self._ocr_engine

    def extract_ocr_lines(self, raw_bytes: bytes) -> list[dict[str, Any]]:
        rgb_np = _decode_image_to_rgb_np(raw_bytes)
        bgr_np = rgb_np[:, :, ::-1].copy()
        ocr = self._get_ocr()

        lines: list[dict[str, Any]] = []
        if hasattr(ocr, "predict"):
            res = ocr.predict(bgr_np)
            for page in res or []:
                rec_texts = None
                rec_scores = None
                if isinstance(page, dict):
                    rec_texts = page.get("rec_texts")
                    rec_scores = page.get("rec_scores")
                elif hasattr(page, "get"):
                    rec_texts = page.get("rec_texts")
                    rec_scores = page.get("rec_scores")
                if rec_texts:
                    scores = list(rec_scores) if rec_scores is not None else [1.0] * len(rec_texts)
                    for txt, sc in zip(rec_texts, scores):
                        clean_t = str(txt).strip()
                        if clean_t:
                            lines.append({"text": clean_t, "confidence": round(float(sc), 4)})
        else:
            res = ocr.ocr(bgr_np)
            for block in res or []:
                for item in block or []:
                    if len(item) >= 2 and isinstance(item[1], (list, tuple)):
                        clean_t = str(item[1][0]).strip()
                        sc = float(item[1][1]) if len(item[1]) > 1 else 1.0
                        if clean_t:
                            lines.append({"text": clean_t, "confidence": round(sc, 4)})
        return lines

    def _match_pmbjp_for_salts(
        self,
        salts: list[SaltEntity],
        target_vehicle: str | None,
        comm_price: float | None,
    ) -> list[dict[str, Any]]:
        if not salts or not self.catalog.pmbjp_rows:
            return []

        req_map = {s.salt.lower().strip(): _norm_strength(s.strength) for s in salts}
        req_set = set(req_map.keys())
        salt_key = "+".join(sorted(req_set))
        fallback_comm = self.catalog.commercial_avg_price_by_salt.get(salt_key)
        ref_comm = comm_price if (comm_price is not None and comm_price > 0) else fallback_comm

        exact_matches: list[dict[str, Any]] = []
        alt_vehicle_matches: list[dict[str, Any]] = []

        for row in self.catalog.pmbjp_rows:
            row_map = {s.salt.lower().strip(): _norm_strength(s.strength) for s in row["salts"]}
            # Gate 1: Exact salt set equality
            if set(row_map.keys()) != req_set:
                continue

            # Gate 2: Dosage alignment (strict when both strengths known)
            dosage_ok = True
            for sname, req_st in req_map.items():
                row_st = row_map.get(sname, "")
                if req_st and row_st and req_st != row_st:
                    dosage_ok = False
                    break
            if not dosage_ok:
                continue

            raw_mrp = row.get("mrp_inr")
            mrp = float(raw_mrp) if (raw_mrp is not None and float(raw_mrp) > 0) else None
            savings_inr = None
            savings_pct = None
            if mrp is not None and ref_comm and ref_comm > 0 and mrp < ref_comm:
                savings_inr = round(ref_comm - mrp, 2)
                savings_pct = round((1.0 - mrp / ref_comm) * 100.0, 1)

            veh_ok = True if not target_vehicle else _vehicles_compatible(target_vehicle, row["vehicle"])
            entry = {
                "product_name": row["product_name"],
                "generic_code": row.get("generic_code"),
                "mrp_inr": mrp,
                "vehicle": row["vehicle"],
                "salts": [{"salt": s.salt, "strength": s.strength} for s in row["salts"]],
                "commercial_ref_inr": ref_comm,
                "savings_inr": savings_inr,
                "savings_pct": savings_pct,
                "equivalence_status": (
                    "Exact Clinical Match (G1-G3 Verified)"
                    if veh_ok
                    else f"Salt & Strength Match (Vehicle: {row['vehicle']} vs {target_vehicle})"
                ),
                "vehicle_exact": veh_ok,
            }
            if veh_ok:
                exact_matches.append(entry)
            else:
                alt_vehicle_matches.append(entry)

        chosen = exact_matches if exact_matches else alt_vehicle_matches
        chosen.sort(key=lambda m: (m["mrp_inr"] is None, m["mrp_inr"] or 999999.0))
        return chosen[:3]

    def _screen_item_safety(
        self,
        salts: list[SaltEntity],
        profile: PatientProfile | None,
    ) -> list[dict[str, str]]:
        if profile is None or not salts:
            return []

        alerts: list[dict[str, str]] = []
        salt_names = list(dict.fromkeys(_normalize_salt_base(s.salt) for s in salts if s.salt.strip()))

        # G7: Drug Allergy Check
        allergies = [a.strip().lower() for a in profile.drug_allergies if a.strip()]
        for alg in allergies:
            expanded = {alg} | ALLERGY_CLASS_SYNONYMS.get(alg, set())
            for sn in salt_names:
                if any(tok == sn or tok in sn or sn in tok for tok in expanded):
                    alerts.append({
                        "gate_id": "G7_ALLERGY",
                        "severity": "BLOCK",
                        "message": f"Allergy Alert: Contains '{sn.title()}', which conflicts with documented allergy '{alg}'.",
                    })

        # G6: Pregnancy / Lactation Warning
        if profile.is_pregnant or profile.is_lactating:
            state_lbl = "pregnancy" if profile.is_pregnant else "lactation"
            for sn in salt_names:
                if sn in PREGNANCY_UNSAFE_SALTS:
                    alerts.append({
                        "gate_id": "G6_PREG_LACTATION",
                        "severity": "WARN",
                        "message": f"Caution in {state_lbl}: '{sn.title()}' requires direct dermatologist/OB-GYN verification before use.",
                    })

        # G11: DDI vs Ongoing Oral Meds
        ongoing = [m.strip().lower() for m in profile.ongoing_oral_meds if m.strip()]
        seen_ddi: set[tuple[str, str]] = set()
        for sn in salt_names:
            for om in ongoing:
                if sn == om:
                    continue
                sev_hit: tuple[str, str] | None = None
                for pair in ((sn, om), (om, sn)):
                    if pair in CURATED_DDI_PAIRS:
                        sev, reason = CURATED_DDI_PAIRS[pair]
                        sev_hit = (sev, f"DDI ({sev}): {sn.title()} + ongoing '{om.title()}' - {reason}")
                        break
                if sev_hit is None and sn in self.catalog.ddi_index:
                    for other_d, eff in self.catalog.ddi_index[sn].items():
                        if other_d == om or other_d in om or om in other_d:
                            if eff in {"SERIOUS", "MAJOR", "MODERATE"}:
                                sev_hit = (eff, f"DDI ({eff}): {sn.title()} interacts with ongoing medication '{om.title()}'.")
                                break
                if sev_hit and (sn, om) not in seen_ddi:
                    seen_ddi.add((sn, om))
                    alerts.append({
                        "gate_id": "G11_DDI",
                        "severity": "BLOCK" if sev_hit[0] in {"SERIOUS", "MAJOR"} else "WARN",
                        "message": sev_hit[1],
                    })

        return alerts

    def _check_cross_rx_ddi(self, items: list[dict[str, Any]]) -> list[dict[str, str]]:
        alerts: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for i in range(len(items)):
            salts_i = {_normalize_salt_base(s["salt"]) for s in items[i]["salts"] if s.get("salt")}
            for j in range(i + 1, len(items)):
                salts_j = {_normalize_salt_base(s["salt"]) for s in items[j]["salts"] if s.get("salt")}
                for sa in salts_i:
                    for sb in salts_j:
                        if not sa or not sb or sa == sb:
                            continue
                        key = tuple(sorted((sa, sb)))
                        if key in seen:
                            continue
                        hit: tuple[str, str] | None = None
                        for pair in ((sa, sb), (sb, sa)):
                            if pair in CURATED_DDI_PAIRS:
                                sev, reason = CURATED_DDI_PAIRS[pair]
                                hit = (sev, f"Cross-Prescription DDI ({sev}): {sa.title()} + {sb.title()} - {reason}")
                                break
                        if hit is None and sa in self.catalog.ddi_index:
                            for other_d, eff in self.catalog.ddi_index[sa].items():
                                if _normalize_salt_base(other_d) == sb and eff in {"SERIOUS", "MAJOR"}:
                                    hit = (eff, f"Cross-Prescription DDI ({eff}): {sa.title()} interacts with {sb.title()}.")
                                    break
                        if hit:
                            seen.add(key)
                            alerts.append({
                                "gate_id": "G11_CROSS_RX_DDI",
                                "severity": "BLOCK" if hit[0] in {"SERIOUS", "MAJOR"} else "WARN",
                                "message": hit[1],
                            })
        return alerts

    def resolve_line(
        self,
        raw_line: str,
        profile: PatientProfile | None = None,
        allow_relaxed: bool = False,
    ) -> dict[str, Any] | None:
        if not allow_relaxed and is_likely_metadata_line(raw_line):
            return None

        cleaned = clean_drug_query(raw_line)
        if len(cleaned) < 3:
            return None

        det_vehicle = detect_vehicle(raw_line)
        det_strengths = extract_strengths(raw_line)

        # 1. Try direct salt match (e.g. "Clotrimazole 1% Cream", "Ketoconazole 2% Shampoo")
        salt_match = None
        if self._known_salts:
            parts = [p.strip() for p in re.split(r"\s*\+\s*", cleaned) if len(p.strip()) >= 3]
            matched_salts: list[SaltEntity] = []
            salt_scores: list[float] = []
            for idx_p, part in enumerate(parts):
                hit = process.extractOne(
                    part,
                    self._known_salts,
                    scorer=fuzz.ratio,
                    score_cutoff=84 if allow_relaxed else 88,
                )
                if hit:
                    s_name, s_score, _ = hit
                    st = det_strengths[idx_p] if idx_p < len(det_strengths) else "N/A"
                    matched_salts.append(SaltEntity(salt=s_name, strength=st))
                    salt_scores.append(float(s_score))
            if matched_salts and len(matched_salts) == len(parts):
                avg_sc = sum(salt_scores) / len(salt_scores)
                salt_key = "+".join(sorted(s.salt for s in matched_salts))
                comm_ref = self.catalog.commercial_avg_price_by_salt.get(salt_key)
                salt_match = {
                    "score": avg_sc,
                    "source": "generic_salt",
                    "display_name": " + ".join(
                        f"{s.salt.title()} {s.strength}" if s.strength != "N/A" else s.salt.title()
                        for s in matched_salts
                    ),
                    "salts": matched_salts,
                    "vehicle": det_vehicle or "UNKNOWN",
                    "commercial_price_inr": comm_ref,
                }

        # 2. Try 1mg branded medicine lookup
        brand_match = None
        if self._brand_keys:
            cutoff = 74 if allow_relaxed else 84
            # Check exact key first, then fuzzy
            if cleaned in self._brand_by_key:
                best_key, best_score = cleaned, 100.0
            else:
                candidates = process.extract(
                    cleaned,
                    self._brand_keys,
                    scorer=fuzz.token_sort_ratio,
                    limit=5,
                    score_cutoff=cutoff,
                )
                # Reject fuzzy matches with wildly different word counts unless relaxed
                filtered = [
                    (k, sc) for k, sc, _ in candidates
                    if allow_relaxed or abs(len(k) - len(cleaned)) <= max(6, len(cleaned) // 2)
                ]
                best_key, best_score = (filtered[0][0], float(filtered[0][1])) if filtered else (None, 0.0)

            if best_key:
                variants = self._brand_by_key[best_key]
                # Prefer variant matching detected vehicle & strength
                def _variant_rank(v: dict[str, Any]) -> tuple[int, int, float]:
                    veh_hit = 1 if (det_vehicle and _vehicles_compatible(det_vehicle, v["vehicle"])) else 0
                    str_hit = 0
                    if det_strengths:
                        v_strs = {_norm_strength(s.strength) for s in v["salts"]}
                        str_hit = sum(1 for ds in det_strengths if _norm_strength(ds) in v_strs)
                    return (veh_hit, str_hit, 1.0 if v["price_inr"] else 0.0)

                chosen = sorted(variants, key=_variant_rank, reverse=True)[0]
                brand_match = {
                    "score": best_score,
                    "source": "1mg_brand",
                    "display_name": chosen["product_name"],
                    "salts": chosen["salts"],
                    "vehicle": det_vehicle or chosen["vehicle"],
                    "commercial_price_inr": chosen["price_inr"],
                }

        # Pick winner (prefer salt_match when exact/near-exact, otherwise higher score)
        winner = None
        if salt_match and brand_match:
            winner = salt_match if salt_match["score"] >= brand_match["score"] else brand_match
        else:
            winner = salt_match or brand_match

        if winner is None:
            return None

        pmbjp_matches = self._match_pmbjp_for_salts(
            salts=winner["salts"],
            target_vehicle=None if winner["vehicle"] == "UNKNOWN" else winner["vehicle"],
            comm_price=winner["commercial_price_inr"],
        )
        safety_alerts = self._screen_item_safety(winner["salts"], profile)

        return {
            "ocr_line": raw_line.strip(),
            "matched_name": winner["display_name"],
            "match_source": winner["source"],
            "match_confidence": round(winner["score"] / 100.0, 4),
            "vehicle": winner["vehicle"],
            "salts": [{"salt": s.salt, "strength": s.strength} for s in winner["salts"]],
            "commercial_price_inr": winner["commercial_price_inr"],
            "pmbjp_matches": pmbjp_matches,
            "safety_alerts": safety_alerts,
        }

    @staticmethod
    def _salt_base_set(item: dict[str, Any]) -> frozenset[str]:
        return frozenset(
            _normalize_salt_base(s["salt"])
            for s in item.get("salts", [])
            if s.get("salt")
        )

    @staticmethod
    def _strengths_compatible(a: dict[str, Any], b: dict[str, Any]) -> bool:
        ma = {_normalize_salt_base(s["salt"]): _norm_strength(s.get("strength", "")) for s in a.get("salts", [])}
        mb = {_normalize_salt_base(s["salt"]): _norm_strength(s.get("strength", "")) for s in b.get("salts", [])}
        for k in set(ma) & set(mb):
            if ma[k] and mb[k] and ma[k] != mb[k]:
                return False
        return True

    @staticmethod
    def _vehicles_match_or_unknown(v1: str, v2: str) -> bool:
        if not v1 or not v2 or v1 == "UNKNOWN" or v2 == "UNKNOWN":
            return True
        return _vehicles_compatible(v1, v2) or _vehicles_compatible(v2, v1)

    def analyze_lines(
        self,
        ocr_lines: list[dict[str, Any]],
        profile: PatientProfile | None = None,
        allow_relaxed: bool = False,
    ) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        item_line_indices: list[int] = []

        for line_idx, entry in enumerate(ocr_lines):
            raw_txt = str(entry.get("text", "")).strip()
            if not raw_txt:
                continue
            resolved = self.resolve_line(raw_txt, profile=profile, allow_relaxed=allow_relaxed)
            if resolved is None:
                continue
            resolved["ocr_confidence"] = round(float(entry.get("confidence", 1.0)), 4)

            res_salts = self._salt_base_set(resolved)
            merged = False

            for prev_idx in range(len(items) - 1, -1, -1):
                prev = items[prev_idx]
                prev_salts = self._salt_base_set(prev)
                near_prev = (line_idx - item_line_indices[prev_idx]) <= 2
                same_salts = bool(res_salts) and res_salts == prev_salts
                sub_salts = bool(res_salts) and res_salts.issubset(prev_salts)

                # Case 1: Generic composition subtitle printed right beneath a branded drug (e.g. Forcan 150 -> FLUCONAZOLE 150MG)
                if (
                    resolved["match_source"] == "generic_salt"
                    and (same_salts or (near_prev and sub_salts))
                    and self._strengths_compatible(prev, resolved)
                    and self._vehicles_match_or_unknown(prev["vehicle"], resolved["vehicle"])
                ):
                    if resolved["ocr_line"] not in prev["ocr_line"]:
                        prev["ocr_line"] = f"{prev['ocr_line']} · {resolved['ocr_line']}"
                    merged = True
                    break

                # Case 2: Brand line read immediately after its own generic subtitle (OCR box-order jitter)
                if (
                    prev["match_source"] == "generic_salt"
                    and resolved["match_source"] == "1mg_brand"
                    and near_prev
                    and same_salts
                    and self._strengths_compatible(prev, resolved)
                    and self._vehicles_match_or_unknown(prev["vehicle"], resolved["vehicle"])
                ):
                    if prev["ocr_line"] not in resolved["ocr_line"]:
                        resolved["ocr_line"] = f"{resolved['ocr_line']} · {prev['ocr_line']}"
                    items[prev_idx] = resolved
                    item_line_indices[prev_idx] = line_idx
                    merged = True
                    break

                # Case 3: Exact same brand/generic + compatible vehicle duplicate
                if (
                    prev["matched_name"].lower() == resolved["matched_name"].lower()
                    and self._vehicles_match_or_unknown(prev["vehicle"], resolved["vehicle"])
                ):
                    merged = True
                    break

            if not merged:
                items.append(resolved)
                item_line_indices.append(line_idx)

        cross_ddi = self._check_cross_rx_ddi(items)

        comm_total = 0.0
        pmbjp_total = 0.0
        comparable_count = 0
        safety_flags = len(cross_ddi)

        for it in items:
            safety_flags += len(it["safety_alerts"])
            best_pmbjp = it["pmbjp_matches"][0] if it["pmbjp_matches"] else None
            if best_pmbjp and best_pmbjp.get("mrp_inr") is not None:
                ref_comm = best_pmbjp.get("commercial_ref_inr") or it.get("commercial_price_inr")
                if ref_comm and ref_comm > 0:
                    comm_total += float(ref_comm)
                    pmbjp_total += float(best_pmbjp["mrp_inr"])
                    comparable_count += 1

        total_savings = round(max(0.0, comm_total - pmbjp_total), 2)
        total_savings_pct = round((total_savings / comm_total) * 100.0, 1) if comm_total > 0 else 0.0

        return {
            "is_prescription": len(items) > 0,
            "raw_ocr_lines": ocr_lines,
            "items": items,
            "cross_rx_ddi_alerts": cross_ddi,
            "summary": {
                "detected_medicines": len(items),
                "pmbjp_matched_medicines": sum(1 for it in items if it["pmbjp_matches"]),
                "safety_alert_count": safety_flags,
                "commercial_total_inr": round(comm_total, 2) if comparable_count else None,
                "pmbjp_total_inr": round(pmbjp_total, 2) if comparable_count else None,
                "total_savings_inr": total_savings if comparable_count else None,
                "total_savings_pct": total_savings_pct if comparable_count else None,
            },
        }

    def scan_prescription_image(
        self,
        raw_bytes: bytes,
        profile: PatientProfile | None = None,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        ocr_lines = self.extract_ocr_lines(raw_bytes)
        result = self.analyze_lines(ocr_lines, profile=profile, allow_relaxed=False)
        result["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        return result

    def search_medicine_text(
        self,
        query: str,
        profile: PatientProfile | None = None,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        raw_parts = [ln.strip() for ln in re.split(r"[\n,;]+", query) if ln.strip()]
        ocr_lines = [{"text": p, "confidence": 1.0} for p in raw_parts]
        result = self.analyze_lines(ocr_lines, profile=profile, allow_relaxed=True)
        result["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        return result
