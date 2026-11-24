#!/usr/bin/env python3
"""
Nidaan-Rx Task 1 — Part 5: Dataset Integrity Auditor, Stratified PaddleOCR Baseline
Benchmark, and Optional Colab Recognition-Crop Exporter.

Usage:
  # 1. Full 5,000-image integrity audit (~10-20s)
  python rx_synth/validate_dataset.py --audit

  # 2. Fast Tier 3 verification after calibrating augmenter.py (~8-10 min on M2 CPU)
  python rx_synth/validate_dataset.py --benchmark --split test --only-tier tier3_hard --sample-n 35

  # 3. Stratified 100-image baseline decision check
  python rx_synth/validate_dataset.py --benchmark --split test --sample-n 100

  # 4. Full 750-image test split benchmark (recommended on Colab T4 GPU for Chapter 9)
  python rx_synth/validate_dataset.py --benchmark --split test --sample-n 0

  # 5. Benchmark a fine-tuned PaddleOCR recognizer (Before vs. After comparison)
  python rx_synth/validate_dataset.py --benchmark --split test --sample-n 0 --rec-model-dir ./finetuned_rec_infer --tag finetuned

  # 6. Export cropped BRAND/SALT lines for Colab T4 recognizer fine-tuning
  python rx_synth/validate_dataset.py --export-rec-crops
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import cv2
import numpy as np

LIST_PREFIX_RE = re.compile(r"^\s*\d{1,2}\s*[\.\)]\s+")
DOSAGE_TOKEN_RE = re.compile(
    r"(?<![\d\.])\d+(?:\.\d+)?\s*(?:%|(?:mg|mcg|gm|g|ml|iu|units)\b)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# String & Edit-Distance Utilities
# ---------------------------------------------------------------------------
def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def normalize_ocr_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.upper().strip())


def strip_list_prefix(text: str) -> str:
    """Strip inline Rx row numbers like '1. ', '2) ' that templates render beside BRAND spans."""
    return LIST_PREFIX_RE.sub("", text).strip()


def normalize_dosage_token(tok: str) -> str:
    tok = re.sub(r"\s+", "", tok.lower().strip())
    return tok.replace("w/w", "").replace("w/v", "").replace("v/v", "")


def extract_dosage_tokens_from_text(text: str) -> list[str]:
    """Extract numeric strength tokens like 0.1%, 2.5%, 500mg, 200 mg, 5000iu."""
    return [normalize_dosage_token(m.group(0)) for m in DOSAGE_TOKEN_RE.finditer(text)]


def dosage_present_in_ocr(norm_st: str, ocr_dosages: set[str], compact_ocr_lower: str) -> bool:
    """Check exact dosage match without letting '1%' falsely match '0.1%' or '21%'."""
    if norm_st in ocr_dosages:
        return True
    pattern = rf"(?<![\d\.]){re.escape(norm_st)}(?![a-z0-9])"
    return bool(re.search(pattern, compact_ocr_lower))


def best_line_match(
    needle: str, haystack_lines: list[str], allow_prefix_strip: bool = False
) -> tuple[float, str]:
    n_norm = normalize_ocr_text(needle)
    if not n_norm:
        return 1.0, ""
    if not haystack_lines:
        return 0.0, ""

    needle_has_prefix = bool(LIST_PREFIX_RE.match(n_norm))
    best_score = 0.0
    best_line = ""

    for line in haystack_lines:
        l_norm = normalize_ocr_text(line)
        if not l_norm:
            continue
        candidates = [l_norm]
        if allow_prefix_strip and not needle_has_prefix:
            stripped = strip_list_prefix(l_norm)
            if stripped and stripped != l_norm:
                candidates.append(stripped)

        for cand in candidates:
            if n_norm == cand:
                return 1.0, cand
            score = SequenceMatcher(None, n_norm, cand).ratio()
            if score > best_score:
                best_score = score
                best_line = cand
    return best_score, best_line


# ---------------------------------------------------------------------------
# Manifest & Storage Helpers
# ---------------------------------------------------------------------------
def load_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    data: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            data[rec["image_id"]] = rec
    return data


def dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return round(total / (1024 * 1024), 2)


# ---------------------------------------------------------------------------
# MODE 1: Full Dataset Integrity Audit
# ---------------------------------------------------------------------------
def run_audit(out_dir: Path, verify_pixels: bool = True) -> dict[str, Any]:
    t0 = time.time()
    gt_path = out_dir / "manifests" / "ground_truth.jsonl"
    warped_path = out_dir / "manifests" / "warped_boxes.jsonl"
    paddle_dir = out_dir / "paddleocr_labels"

    for req in (gt_path, warped_path):
        if not req.exists():
            raise FileNotFoundError(f"Missing required manifest: {req}")

    gt_by_id = load_jsonl_by_id(gt_path)
    warped_by_id = load_jsonl_by_id(warped_path)

    gt_ids = set(gt_by_id.keys())
    warped_ids = set(warped_by_id.keys())
    missing_warped = sorted(gt_ids - warped_ids)
    extra_warped = sorted(warped_ids - gt_ids)

    paddle_ids_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    paddle_parse_errors = 0
    for sp in ("train", "val", "test"):
        pfile = paddle_dir / f"{sp}_det_rec.txt"
        if not pfile.exists():
            continue
        with pfile.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    paddle_parse_errors += 1
                    continue
                fname, ann_json = parts
                paddle_ids_by_split[sp].add(Path(fname).stem)
                try:
                    json.loads(ann_json)
                except json.JSONDecodeError:
                    paddle_parse_errors += 1

    split_leakage = (
        (paddle_ids_by_split["train"] & paddle_ids_by_split["val"])
        | (paddle_ids_by_split["train"] & paddle_ids_by_split["test"])
        | (paddle_ids_by_split["val"] & paddle_ids_by_split["test"])
    )

    missing_jpegs: list[str] = []
    corrupt_jpegs: list[str] = []
    dim_mismatches: list[str] = []
    bad_polygons = 0
    empty_transcriptions = 0

    tag_counts = Counter()
    split_counts = Counter()
    tier_counts = Counter()
    lang_counts = Counter()
    tpl_counts = Counter()
    hn_count = 0
    jpeg_sizes_kb: list[float] = []

    for img_id in sorted(warped_by_id.keys()):
        rec = warped_by_id[img_id]
        img_path = out_dir / rec["file_name"]
        split_counts[rec["split"]] += 1
        tier_counts[rec["difficulty"]] += 1
        lang_counts[rec["language"]] += 1
        tpl_counts[rec["template_id"]] += 1
        if rec.get("is_hard_negative"):
            hn_count += 1

        if not img_path.exists():
            missing_jpegs.append(img_id)
            continue

        fsize_kb = img_path.stat().st_size / 1024.0
        jpeg_sizes_kb.append(fsize_kb)
        if fsize_kb < 5.0:
            corrupt_jpegs.append(img_id)
            continue

        w, h = rec["width"], rec["height"]
        if verify_pixels:
            im = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if im is None:
                corrupt_jpegs.append(img_id)
                continue
            ih, iw = im.shape[:2]
            if iw != w or ih != h:
                dim_mismatches.append(img_id)

        for b in rec.get("boxes", []):
            tag_counts[b.get("tag", "UNKNOWN")] += 1
            if not str(b.get("transcription", "")).strip():
                empty_transcriptions += 1
            pts = b.get("points", [])
            if len(pts) != 4:
                bad_polygons += 1
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            if min(xs) < 0 or min(ys) < 0 or max(xs) >= w or max(ys) >= h:
                bad_polygons += 1
            elif (max(xs) - min(xs)) < 1 or (max(ys) - min(ys)) < 1:
                bad_polygons += 1

    images_mb = dir_size_mb(out_dir / "images")
    raw_mb = dir_size_mb(out_dir / "raw_renders")

    passed = (
        len(missing_warped) == 0
        and len(extra_warped) == 0
        and len(missing_jpegs) == 0
        and len(corrupt_jpegs) == 0
        and len(dim_mismatches) == 0
        and len(split_leakage) == 0
        and paddle_parse_errors == 0
        and bad_polygons == 0
        and empty_transcriptions == 0
    )

    report = {
        "audit_passed": passed,
        "elapsed_sec": round(time.time() - t0, 2),
        "counts": {
            "ground_truth_records": len(gt_by_id),
            "warped_records": len(warped_by_id),
            "hard_negatives": hn_count,
            "splits": dict(split_counts),
            "difficulty_tiers": dict(tier_counts),
            "languages": dict(lang_counts),
            "templates": dict(tpl_counts),
            "total_boxes": sum(tag_counts.values()),
            "boxes_by_tag": dict(tag_counts),
        },
        "paddleocr_split_counts": {k: len(v) for k, v in paddle_ids_by_split.items()},
        "storage": {
            "images_dir_mb": images_mb,
            "raw_renders_dir_mb": raw_mb,
            "avg_jpeg_kb": round(float(np.mean(jpeg_sizes_kb)), 1) if jpeg_sizes_kb else 0.0,
        },
        "issues": {
            "missing_warped": len(missing_warped),
            "extra_warped": len(extra_warped),
            "missing_jpegs": len(missing_jpegs),
            "corrupt_jpegs": len(corrupt_jpegs),
            "dim_mismatches": len(dim_mismatches),
            "split_leakage": len(split_leakage),
            "paddle_parse_errors": paddle_parse_errors,
            "bad_polygons": bad_polygons,
            "empty_transcriptions": empty_transcriptions,
        },
    }

    report_path = out_dir / "manifests" / "audit_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    status = "PASSED" if passed else "FAILED (check issues below)"
    print(f"\n================ DATASET INTEGRITY AUDIT: {status} ================")
    print(f"  Records verified : {len(warped_by_id):,} / {len(gt_by_id):,} ({report['elapsed_sec']}s)")
    print(f"  Splits           : {report['counts']['splits']}")
    print(f"  Difficulty tiers : {report['counts']['difficulty_tiers']}")
    print(f"  Languages        : {report['counts']['languages']}")
    print(f"  Hard negatives   : {hn_count:,}")
    print(f"  Total OCR boxes  : {report['counts']['total_boxes']:,} -> {report['counts']['boxes_by_tag']}")
    print(f"  Disk usage       : images/ = {images_mb:.1f} MB (avg {report['storage']['avg_jpeg_kb']} KB/img) | raw_renders/ = {raw_mb:.1f} MB")
    print(f"  Issues           : {report['issues']}")
    print(f"  Saved report     : {report_path}")
    if passed and raw_mb > 100:
        print(f"\n  [TIP] Audit passed! Keep raw_renders/ until the final benchmark is locked, then reclaim ~{raw_mb:.0f} MB with: rm -rf {out_dir / 'raw_renders'}")
    return report


# ---------------------------------------------------------------------------
# MODE 2: Stratified PaddleOCR Benchmark (Baseline or Fine-Tuned)
# ---------------------------------------------------------------------------
class PaddleOCRCompat:
    """Wrapper supporting PaddleOCR v3.x (predict) and v2.x (ocr), with optional custom inference dirs."""

    def __init__(
        self,
        lang: str = "en",
        det_model_dir: str | None = None,
        rec_model_dir: str | None = None,
    ) -> None:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import PaddleOCR

        self.mode = "v3"
        v3_kwargs: dict[str, Any] = {
            "lang": lang,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        }
        if det_model_dir:
            v3_kwargs["text_detection_model_dir"] = det_model_dir
        if rec_model_dir:
            v3_kwargs["text_recognition_model_dir"] = rec_model_dir

        try:
            self.engine = PaddleOCR(**v3_kwargs)
            if not hasattr(self.engine, "predict"):
                self.mode = "v2"
        except TypeError:
            self.mode = "v2"
            v2_kwargs: dict[str, Any] = {"lang": lang, "use_angle_cls": False, "show_log": False}
            if det_model_dir:
                v2_kwargs["det_model_dir"] = det_model_dir
            if rec_model_dir:
                v2_kwargs["rec_model_dir"] = rec_model_dir
            self.engine = PaddleOCR(**v2_kwargs)

    def extract_lines(self, img_path: Path) -> list[tuple[str, float]]:
        lines: list[tuple[str, float]] = []
        if self.mode == "v3":
            results = self.engine.predict(str(img_path))
            for res in results:
                texts, scores = [], []
                if isinstance(res, Mapping) and "rec_texts" in res:
                    texts = res.get("rec_texts", [])
                    scores = res.get("rec_scores", [])
                elif hasattr(res, "json") and isinstance(res.json, Mapping):
                    inner = res.json.get("res", res.json)
                    texts = inner.get("rec_texts", [])
                    scores = inner.get("rec_scores", [])
                else:
                    texts = getattr(res, "rec_texts", [])
                    scores = getattr(res, "rec_scores", [])
                for t, s in zip(texts, scores):
                    t_str = str(t).strip()
                    if t_str:
                        lines.append((t_str, float(s)))
        else:
            res = self.engine.ocr(str(img_path), cls=False)
            if res and res[0]:
                for entry in res[0]:
                    if len(entry) >= 2 and isinstance(entry[1], (list, tuple)):
                        t_str = str(entry[1][0]).strip()
                        conf = float(entry[1][1])
                        if t_str:
                            lines.append((t_str, conf))
        return lines


def select_stratified_sample(
    records: list[dict[str, Any]], sample_n: int, seed: int
) -> list[dict[str, Any]]:
    """Sample proportionally across difficulty tiers + hard negatives."""
    if sample_n <= 0 or sample_n >= len(records):
        return records

    rng = random.Random(seed)
    buckets: dict[tuple[str, bool], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        buckets[(r["difficulty"], bool(r.get("is_hard_negative", False)))].append(r)

    chosen: list[dict[str, Any]] = []
    total = len(records)
    for _, items in sorted(buckets.items()):
        k = max(1, int(round(sample_n * (len(items) / total))))
        k = min(k, len(items))
        chosen.extend(rng.sample(items, k))

    chosen_ids = {r["image_id"] for r in chosen}
    if len(chosen) > sample_n:
        rng.shuffle(chosen)
        chosen = chosen[:sample_n]
    elif len(chosen) < sample_n:
        remaining = [r for r in records if r["image_id"] not in chosen_ids]
        chosen.extend(rng.sample(remaining, min(sample_n - len(chosen), len(remaining))))

    chosen.sort(key=lambda r: r["image_id"])
    return chosen


def _new_bucket() -> dict[str, Any]:
    return {
        "rx_images": 0,
        "hn_images": 0,
        "hn_false_positives": 0,
        "edit_dist_sum": 0,
        "char_len_sum": 0,
        "brand_salt_lines": 0,
        "salts_total": 0,
        "salts_exact": 0,
        "salts_fuzzy": 0,
        "dosages_total": 0,
        "dosages_exact": 0,
        "latency_sum": 0.0,
    }


def _summarize_bucket(d: dict[str, Any]) -> dict[str, Any]:
    cer = d["edit_dist_sum"] / max(1, d["char_len_sum"])
    return {
        "rx_images": int(d["rx_images"]),
        "hn_images": int(d["hn_images"]),
        "hn_zero_hallucination_rate": round(1.0 - d["hn_false_positives"] / max(1, d["hn_images"]), 4) if d["hn_images"] else 1.0,
        "brand_salt_cer": round(cer, 4),
        "brand_salt_char_acc": round(1.0 - cer, 4),
        "salt_recall_exact": round(d["salts_exact"] / max(1, d["salts_total"]), 4),
        "salt_recall_fuzzy_85": round(d["salts_fuzzy"] / max(1, d["salts_total"]), 4),
        "dosage_exact_acc": round(d["dosages_exact"] / max(1, d["dosages_total"]), 4),
        "avg_latency_sec": round(d["latency_sum"] / max(1, d["rx_images"] + d["hn_images"]), 2),
    }


def run_benchmark(
    out_dir: Path,
    split: str = "test",
    sample_n: int = 100,
    seed: int = 42,
    fuzzy_thresh: float = 0.85,
    only_tier: str | None = None,
    det_model_dir: str | None = None,
    rec_model_dir: str | None = None,
    run_tag: str = "baseline",
) -> dict[str, Any]:
    gt_by_id = load_jsonl_by_id(out_dir / "manifests" / "ground_truth.jsonl")
    warped_by_id = load_jsonl_by_id(out_dir / "manifests" / "warped_boxes.jsonl")

    split_records = [
        warped_by_id[img_id]
        for img_id in sorted(warped_by_id.keys())
        if warped_by_id[img_id]["split"] == split
        and img_id in gt_by_id
        and (only_tier is None or warped_by_id[img_id]["difficulty"] == only_tier)
    ]
    if not split_records:
        raise ValueError(f"No records found for split='{split}', only_tier='{only_tier}'.")

    selected = select_stratified_sample(split_records, sample_n, seed)
    print(f"[INFO] Loading PaddleOCR ({run_tag})...")
    ocr = PaddleOCRCompat(lang="en", det_model_dir=det_model_dir, rec_model_dir=rec_model_dir)
    tier_label = f" [{only_tier}]" if only_tier else ""
    print(f"[INFO] Running {run_tag} benchmark on {len(selected)} '{split}'{tier_label} images (PaddleOCR {ocr.mode})...")

    known_salts = {
        s["salt"].lower()
        for gt in gt_by_id.values()
        for m in gt.get("medications", [])
        for s in m.get("salts", [])
        if len(s["salt"]) >= 5
    }

    tier_stats: dict[str, dict[str, Any]] = defaultdict(_new_bucket)
    lang_stats: dict[str, dict[str, Any]] = defaultdict(_new_bucket)
    tpl_stats: dict[str, dict[str, Any]] = defaultdict(_new_bucket)
    per_image_rows: list[dict[str, Any]] = []
    t_start = time.time()

    for idx, rec in enumerate(selected, start=1):
        img_id = rec["image_id"]
        gt = gt_by_id[img_id]
        tier = rec["difficulty"]
        lang = rec["language"]
        tpl = rec["template_id"]
        img_path = out_dir / rec["file_name"]

        t1 = time.time()
        ocr_lines_with_conf = ocr.extract_lines(img_path)
        lat = time.time() - t1

        ocr_texts = [t for t, _ in ocr_lines_with_conf]
        full_ocr_upper = " ".join(normalize_ocr_text(t) for t in ocr_texts)
        full_ocr_lower = full_ocr_upper.lower()
        buckets_to_update = (tier_stats[tier], lang_stats[lang], tpl_stats[tpl])

        for b in buckets_to_update:
            b["latency_sum"] += lat

        if gt["is_hard_negative"]:
            false_hits = [s for s in known_salts if re.search(rf"\b{re.escape(s)}\b", full_ocr_lower)]
            for b in buckets_to_update:
                b["hn_images"] += 1
                if false_hits:
                    b["hn_false_positives"] += 1
            per_image_rows.append({
                "image_id": img_id,
                "difficulty": tier,
                "language": lang,
                "template_id": tpl,
                "is_hard_negative": True,
                "cer": 0.0,
                "salt_recall_fuzzy": 1.0 if not false_hits else 0.0,
                "dosage_exact_acc": 1.0,
                "latency_sec": round(lat, 2),
                "missed_dosages": "",
                "worst_error": f"HN_FALSE_SALT:{false_hits[:3]}" if false_hits else "",
            })
        else:
            target_boxes = [b for b in rec["boxes"] if b["tag"] in {"BRAND", "SALT"}]
            img_edits = 0
            img_chars = 0
            worst_pair = ""
            worst_cer = 0.0

            for tb in target_boxes:
                gt_line = normalize_ocr_text(tb["transcription"])
                if not gt_line:
                    continue
                allow_strip = tb["tag"] == "BRAND"
                _, matched = best_line_match(gt_line, ocr_texts, allow_prefix_strip=allow_strip)
                ed = levenshtein_distance(gt_line, matched)
                img_edits += ed
                img_chars += len(gt_line)
                line_cer = ed / max(1, len(gt_line))
                if line_cer > worst_cer:
                    worst_cer = line_cer
                    worst_pair = f"GT='{gt_line}' | OCR='{matched}'"

            img_salts_total = 0
            img_salts_exact = 0
            img_salts_fuzzy = 0
            img_dos_total = 0
            img_dos_exact = 0
            missed_dosages: list[str] = []

            ocr_dosages = set(extract_dosage_tokens_from_text(full_ocr_upper))
            words_ocr = re.findall(r"[A-Z0-9\-]+", full_ocr_upper)
            compact_ocr_lower = full_ocr_lower.replace(" ", "")

            for med in gt["medications"]:
                for s in med["salts"]:
                    salt_name = s["salt"].upper()
                    img_salts_total += 1

                    if salt_name in full_ocr_upper:
                        img_salts_exact += 1
                        img_salts_fuzzy += 1
                    else:
                        salt_words = salt_name.split()
                        if len(salt_words) == 1:
                            sim = max((SequenceMatcher(None, salt_name, w).ratio() for w in words_ocr), default=0.0)
                        else:
                            sim, _ = best_line_match(salt_name, ocr_texts)
                        if sim >= fuzzy_thresh:
                            img_salts_fuzzy += 1

                    strength = s.get("strength", "N/A")
                    if strength and strength.upper() != "N/A":
                        norm_st = normalize_dosage_token(strength)
                        if norm_st:
                            img_dos_total += 1
                            if dosage_present_in_ocr(norm_st, ocr_dosages, compact_ocr_lower):
                                img_dos_exact += 1
                            else:
                                missed_dosages.append(f"{s['salt']}:{norm_st}")

            for b in buckets_to_update:
                b["rx_images"] += 1
                b["edit_dist_sum"] += img_edits
                b["char_len_sum"] += img_chars
                b["brand_salt_lines"] += len(target_boxes)
                b["salts_total"] += img_salts_total
                b["salts_exact"] += img_salts_exact
                b["salts_fuzzy"] += img_salts_fuzzy
                b["dosages_total"] += img_dos_total
                b["dosages_exact"] += img_dos_exact

            img_cer = img_edits / max(1, img_chars)
            per_image_rows.append({
                "image_id": img_id,
                "difficulty": tier,
                "language": lang,
                "template_id": tpl,
                "is_hard_negative": False,
                "cer": round(img_cer, 4),
                "salt_recall_fuzzy": round(img_salts_fuzzy / max(1, img_salts_total), 4),
                "dosage_exact_acc": round(img_dos_exact / max(1, img_dos_total), 4) if img_dos_total else 1.0,
                "latency_sec": round(lat, 2),
                "missed_dosages": "; ".join(missed_dosages),
                "worst_error": worst_pair if worst_cer > 0.12 else "",
            })

        if idx % 10 == 0 or idx == len(selected):
            el = time.time() - t_start
            eta = (len(selected) - idx) * (el / idx)
            print(f"  [{idx:>3}/{len(selected)}] avg {el/idx:.2f}s/img | elapsed {el:.1f}s | ETA {eta:.1f}s")

    overall_raw = _new_bucket()
    for d in tier_stats.values():
        for k, v in d.items():
            overall_raw[k] += v

    t23_raw = _new_bucket()
    for t in ("tier2_smartphone", "tier3_hard"):
        if t in tier_stats:
            for k, v in tier_stats[t].items():
                t23_raw[k] += v

    summary_by_tier = {t: _summarize_bucket(tier_stats[t]) for t in ("tier1_clean", "tier2_smartphone", "tier3_hard") if t in tier_stats}
    summary_by_lang = {lg: _summarize_bucket(lang_stats[lg]) for lg in ("en", "en_gu", "en_hi") if lg in lang_stats}
    summary_by_tpl = {tp: _summarize_bucket(tpl_stats[tp]) for tp in sorted(tpl_stats.keys()) if tpl_stats[tp]["rx_images"] > 0}
    overall_summary = _summarize_bucket(overall_raw)
    t23_summary = _summarize_bucket(t23_raw) if t23_raw["rx_images"] > 0 else overall_summary

    passes_gate = (
        t23_summary["salt_recall_fuzzy_85"] >= 0.95
        and t23_summary["dosage_exact_acc"] >= 0.95
        and t23_summary["brand_salt_char_acc"] >= 0.95
    )

    verdict = (
        "PASS (>= 95% on Tier 2+3): Pre-trained PaddleOCR generalizes cleanly. "
        "Skip OCR fine-tuning; run --sample-n 0 (on Colab T4) for your full 750-image thesis table "
        "and redirect compute to the NLP filter & v2.0 dermatology vision model."
        if passes_gate else
        "FINE-TUNE RECOMMENDED (< 95% on Tier 2+3): Run --export-rec-crops and fine-tune "
        "PaddleOCR's Recognizer on Google Colab T4 for a Before-vs-After Chapter 9 upgrade."
    )

    output = {
        "run_tag": run_tag,
        "split": split,
        "only_tier": only_tier,
        "sample_size": len(selected),
        "paddleocr_mode": ocr.mode,
        "total_elapsed_sec": round(time.time() - t_start, 1),
        "overall": overall_summary,
        "tier2_plus_tier3": t23_summary,
        "by_tier": summary_by_tier,
        "by_language": summary_by_lang,
        "by_template": summary_by_tpl,
        "decision_passes_95_gate": passes_gate,
        "decision_verdict": verdict,
    }

    suffix = f"{run_tag}_{split}_{only_tier or 'all'}_{len(selected)}"
    res_json = out_dir / "manifests" / f"ocr_{suffix}.json"
    res_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    err_csv = out_dir / "manifests" / f"ocr_errors_{suffix}.csv"
    sorted_rows = sorted(per_image_rows, key=lambda r: (r["dosage_exact_acc"], -r["cer"]))
    with err_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sorted_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sorted_rows)

    def print_table(title: str, rows: dict[str, dict[str, Any]]) -> None:
        print(f"\n--- {title} ---")
        print(f"{'Group':<20} {'Rx':>4} {'CER':>8} {'CharAcc':>9} {'SaltEx':>8} {'SaltFz85':>9} {'DosageEx':>9} {'Sec/Img':>8}")
        print("-" * 80)
        for name, m in rows.items():
            print(
                f"{name:<20} {m['rx_images']:>4} {m['brand_salt_cer']*100:>7.2f}% "
                f"{m['brand_salt_char_acc']*100:>8.2f}% {m['salt_recall_exact']*100:>7.2f}% "
                f"{m['salt_recall_fuzzy_85']*100:>8.2f}% {m['dosage_exact_acc']*100:>8.2f}% "
                f"{m['avg_latency_sec']:>7.2f}s"
            )

    print(f"\n================ STRATIFIED PADDLEOCR REPORT ({run_tag.upper()}) ================")
    print_table("By Difficulty Tier", {**summary_by_tier, "TIER2+3": t23_summary, "OVERALL": overall_summary})
    print_table("By Language", summary_by_lang)
    print_table("By Template", summary_by_tpl)
    print(f"\n  Hard-Negative Zero-Hallucination Rate : {overall_summary['hn_zero_hallucination_rate']*100:.1f}% ({overall_summary['hn_images']} docs)")
    print(f"  Verdict : {verdict}")
    print(f"  Saved   : {res_json}")
    print(f"  Errors  : {err_csv}")
    return output


# ---------------------------------------------------------------------------
# MODE 3: Optional Colab Fine-Tuning Crop Exporter (BRAND + SALT Lines)
# ---------------------------------------------------------------------------
def perspective_crop(img: np.ndarray, points: list[list[int]]) -> np.ndarray | None:
    pts = np.array(points, dtype=np.float32)
    w_top = float(np.linalg.norm(pts[1] - pts[0]))
    w_bot = float(np.linalg.norm(pts[2] - pts[3]))
    h_left = float(np.linalg.norm(pts[3] - pts[0]))
    h_right = float(np.linalg.norm(pts[2] - pts[1]))
    cw = max(8, int(round(max(w_top, w_bot))))
    ch = max(8, int(round(max(h_left, h_right))))
    dst = np.array([[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    crop = cv2.warpPerspective(img, M, (cw, ch), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return crop if crop.size > 0 else None


def export_recognition_crops(out_dir: Path, tags: set[str]) -> None:
    warped_by_id = load_jsonl_by_id(out_dir / "manifests" / "warped_boxes.jsonl")
    crops_root = out_dir / "paddleocr_rec_crops"
    crops_root.mkdir(parents=True, exist_ok=True)

    handles = {
        sp: (crops_root / f"rec_gt_{sp}.txt").open("w", encoding="utf-8")
        for sp in ("train", "val", "test")
    }
    counts = Counter()
    t0 = time.time()

    try:
        for img_id in sorted(warped_by_id.keys()):
            rec = warped_by_id[img_id]
            sp = rec["split"]
            target_boxes = [b for b in rec["boxes"] if b["tag"] in tags]
            if not target_boxes:
                continue

            img = cv2.imread(str(out_dir / rec["file_name"]), cv2.IMREAD_COLOR)
            if img is None:
                continue

            sp_dir = crops_root / sp
            sp_dir.mkdir(parents=True, exist_ok=True)
            for idx, b in enumerate(target_boxes):
                crop = perspective_crop(img, b["points"])
                if crop is None:
                    continue
                crop_name = f"{img_id}_{b['tag'].lower()}_{idx:02d}.jpg"
                cv2.imwrite(str(sp_dir / crop_name), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                handles[sp].write(f"{sp}/{crop_name}\t{b['transcription']}\n")
                counts[sp] += 1
    finally:
        for h in handles.values():
            h.close()

    print(f"[DONE] Exported {sum(counts.values()):,} recognition crops in {time.time()-t0:.1f}s -> {crops_root}")
    print(f"       Split crop counts: {dict(counts)}")


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Part 5: Dataset Auditor & Stratified PaddleOCR Benchmark")
    parser.add_argument("--out-dir", type=Path, default=Path("synthetic_rx_dataset"))
    parser.add_argument("--audit", action="store_true", help="Run full 5,000-image integrity audit")
    parser.add_argument("--fast-audit", action="store_true", help="Skip pixel decoding during audit")
    parser.add_argument("--benchmark", action="store_true", help="Run stratified PaddleOCR benchmark")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--only-tier", choices=["tier1_clean", "tier2_smartphone", "tier3_hard"], default=None,
                        help="Restrict benchmark to a single difficulty tier (fast targeted verification)")
    parser.add_argument("--sample-n", type=int, default=100, help="Number of images to benchmark (0 = entire split)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--det-model-dir", default=None, help="Optional custom PaddleOCR detection inference dir")
    parser.add_argument("--rec-model-dir", default=None, help="Optional custom PaddleOCR recognition inference dir")
    parser.add_argument("--tag", default="baseline", help="Label for output report files (e.g. baseline, finetuned)")
    parser.add_argument("--export-rec-crops", action="store_true", help="Export rectified BRAND/SALT line crops for Colab fine-tuning")
    args = parser.parse_args()

    if not (args.audit or args.benchmark or args.export_rec_crops):
        parser.print_help()
        return

    if args.audit:
        run_audit(args.out_dir, verify_pixels=not args.fast_audit)
    if args.benchmark:
        run_benchmark(
            args.out_dir,
            split=args.split,
            sample_n=args.sample_n,
            seed=args.seed,
            only_tier=args.only_tier,
            det_model_dir=args.det_model_dir,
            rec_model_dir=args.rec_model_dir,
            run_tag=args.tag,
        )
    if args.export_rec_crops:
        export_recognition_crops(args.out_dir, tags={"BRAND", "SALT"})


if __name__ == "__main__":
    main()
