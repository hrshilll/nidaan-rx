#!/usr/bin/env python3
"""
Nidaan-Rx Task 1 — Part 1: Synthetic Prescription Payload & Ground-Truth Sampler.

Reads 1mg_dermatology_parsed.csv and outputs:
  1. manifests/payloads.jsonl      (full render payloads for Jinja2/Playwright)
  2. manifests/ground_truth.jsonl  (Format 1 evaluation manifest)
  3. manifests/ground_truth.csv    (flat summary table for quick inspection)
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import random
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from faker import Faker

# ---------------------------------------------------------------------------
# Stratified Distribution Targets
# ---------------------------------------------------------------------------
SPLIT_WEIGHTS = [
    ("train", 0.70),
    ("val", 0.15),
    ("test", 0.15),
]

TEMPLATE_WEIGHTS = [
    ("T1_emr_bordered", 0.35),
    ("T2_modern_hospital", 0.20),
    ("T3_compact_a5", 0.20),
    ("T4_dense_derm", 0.15),
    ("T5_minimal_legacy", 0.10),
]

LANGUAGE_WEIGHTS = [
    ("en", 0.45),
    ("en_gu", 0.35),
    ("en_hi", 0.20),
]

DIFFICULTY_WEIGHTS = [
    ("tier1_clean", 0.15),
    ("tier2_smartphone", 0.55),
    ("tier3_hard", 0.30),
]

# ---------------------------------------------------------------------------
# Dermatology Filtering & Normalization Constants
# ---------------------------------------------------------------------------
NON_DERM_KEYWORDS = [
    "eye drop", "ear drop", "nasal", "vaginal", "suppository", "inhaler",
    "transhaler", "accuhaler", "transcaps", "respules", "rotacap",
    "mouth paint", "gargle", "liquifilm", "pellets", "ophthalmic",
    "otological", "pessary", "eardrop", "eyedrop", "drops", "spray",
    "injection", "infusion", "syrup", "kit", "eye ointment", "vaginobact",
    "ear ", " eye",
]

NON_DERM_SUBCATS = {
    "Gynaecological Antibiotics Sulphonamides",
    "Bronchodilators",
}

VEHICLE_STRIP_RE = re.compile(
    r"\b(TABLET|CAPSULE|CREAM|GEL|OINTMENT|LOTION|SHAMPOO|SOAP|POWDER|"
    r"SUSPENSION|SOLUTION|FOAM|SERUM|CONDITIONER|FACEWASH|SUNSCREEN|MOISTURIZER)\b",
    flags=re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Realistic Dermatology Clinical Pools (100% Synthetic)
# ---------------------------------------------------------------------------
CLINIC_PREFIXES = [
    "DermaCare", "SkinCraft", "Aura Skin & Hair", "ClearDerm", "Epithelia",
    "Radiance Dermatology", "Apex Skin Studio", "Lotus Dermaclinic",
    "Zenith Skin & Laser", "Nirmal Skin Care", "Veda Derm Clinic",
]

DOCTOR_QUALIFICATIONS = [
    "MD - Dermatology, Venereology & Leprosy, M.B.B.S.",
    "MBBS, DVD (Dermatology & Cosmetology)",
    "MD (Dermatology), Dermatosurgeon & Aesthetic Dermatologist",
    "MBBS, DNB (Dermatology & Venereology)",
]

EMR_FOOTERS = [
    "Powered by MedSygn",
    "Powered by ClinicOS",
    "Generated via DermFlow EMR",
    "Powered by HealthBridge Rx",
    "E-Prescription System v2.4",
]

COMPLAINTS_AND_DIAGNOSES = [
    ("hair loss, pimples", "Acne vulgaris, MPHL"),
    ("hair loss, macular pigmentations on face", "FPHL, Melasma"),
    ("hair loss, dandruff", "FPHL, Seborrheic dermatitis"),
    ("itchy circular rash on trunk", "Tinea corporis"),
    ("redness, itching and dry scaly patches", "Atopic dermatitis"),
    ("recurrent hives and itching after cold exposure", "Chronic urticaria"),
    ("dark patches on cheeks and forehead", "Melasma, Post-inflammatory hyperpigmentation"),
    ("flaking scalp, itching", "Seborrheic dermatitis, Scalp folliculitis"),
    ("painful nodules on jawline and chin", "Nodulocystic acne"),
    ("itching in groin folds, redness", "Tinea cruris"),
    ("dry cracked skin on hands and forearms", "Irritant contact dermatitis, Xerosis"),
    ("silvery scaly plaques on elbows and knees", "Psoriasis vulgaris"),
]

INVESTIGATION_BLOCKS = [
    {
        "title": "COMPLETE BLOOD COUNT- CBC (.)",
        "details": (
            "ERYTHROCYTE SEDIMENTATION RATE (ESR), PLATELET COUNT, HEMOGLOBIN (HB), "
            "HEMATOCRIT (HCT), RED BLOOD CELL COUNT (RBC), WHITE BLOOD CELL COUNT (WBC), "
            "MEAN CORPUSCULAR VOLUME (MCV), MEAN CORPUSCULAR HEMOGLOBIN (MCH), "
            "MEAN CORPUSCULAR HEMOGLOBIN CONCENTRATION (MCHC), RED CELL DISTRIBUTION WIDTH (RDW), "
            "NEUTROPHILS, LYMPHOCYTES, MONOCYTES, EOSINOPHILS, BASOPHILS"
        ),
    },
    {
        "title": "THYROID FUNCTION TEST (TFT)",
        "details": "THYROID STIMULATING HORMONE (TSH), TOTAL THYROXINE (T4), TOTAL TRIIODOTHYRONINE (T3)",
    },
    {
        "title": "VITAMIN LEVELS (B12 and D3)",
        "details": "VITAMIN B12, VITAMIN D (25-HYDROXY)",
    },
    {
        "title": "SERUM FERRITIN & IRON PROFILE",
        "details": "SERUM IRON, TOTAL IRON BINDING CAPACITY (TIBC), TRANSFERRIN SATURATION",
    },
    {
        "title": "FASTING LIPID & LIVER PANEL (LFT)",
        "details": "SGOT (AST), SGPT (ALT), TOTAL BILIRUBIN, SERUM TRIGLYCERIDES, HDL, LDL CHOLESTEROL",
    },
    {
        "title": "HbA1c & FASTING BLOOD GLUCOSE",
        "details": "GLYCOSYLATED HEMOGLOBIN (HbA1c), MEAN BLOOD GLUCOSE",
    },
]

# Multilingual table headers matching Indian EMR conventions
TABLE_HEADERS = {
    "en": {
        "drug": "Brand & Strength",
        "dose": "Dose",
        "freq": "Frequency",
        "instr": "Instruction",
        "duration": "Duration",
        "qty": "Qty.",
    },
    "en_gu": {
        "drug": "દવાઓ",
        "dose": "જથ્થો",
        "freq": "આવર્તન",
        "instr": "સૂચના",
        "duration": "સમયગાળો",
        "qty": "Qty.",
    },
    "en_hi": {
        "drug": "दवाइयाँ",
        "dose": "मात्रा",
        "freq": "आवृत्ति",
        "instr": "निर्देश",
        "duration": "अवधि",
        "qty": "Qty.",
    },
}

ORAL_VEHICLES = {"TABLET", "CAPSULE", "TABLET/CAPSULE", "SYRUP", "SACHET"}

TOPICAL_FREQ_NOTES = [
    "apply immediately after facewash",
    "apply 30 mins after moisturiser, apply for 15 to 20 mins and wash",
    "apply 1 hour after moisturizer, keep it for 30 mins and wash. try keeping overnight if skin tolerates",
    "apply 1ml/16 drops on scalp evenly",
    "apply for 5 minutes and wash thoroughly",
    "apply thin layer on affected area at night only; use SPF 30+ in morning",
    "apply 1-1.5 fingertip unit on scalp once daily, then increase to twice daily after 1 week",
]

ORAL_FREQ_NOTES = [
    "start with half tablet daily for 7 days and take 1 tablet daily",
    "start with 1 capsule alternate days for first 2 weeks and then take 1 capsule daily",
    "take with a full glass of water; avoid lying down for 30 minutes",
]


# ---------------------------------------------------------------------------
# Defensive CSV Parsing
# ---------------------------------------------------------------------------
def _find_col(columns: list[str], candidates: list[str]) -> str | None:
    lower_map = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def parse_entities_field(raw: Any) -> list[dict[str, str]]:
    """Parse JSON or Python-literal list of salt dicts and strip trailing ' na' artifacts."""
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return []
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, str):
        text = raw.strip()
        if not text or text in {"[]", "nan", "None", "null"}:
            return []
        try:
            items = json.loads(text)
        except json.JSONDecodeError:
            try:
                items = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return []
    else:
        return []

    cleaned: list[dict[str, str]] = []
    if not isinstance(items, list):
        return cleaned

    for entry in items:
        if isinstance(entry, dict):
            salt = str(entry.get("salt", entry.get("name", ""))).strip().lower()
            salt = re.sub(r"\s+", " ", salt)
            salt = re.sub(r"\s+na$", "", salt).strip()
            strength = str(entry.get("strength", entry.get("dosage", "N/A"))).strip()
            if not strength or strength.lower() in {"none", "nan", "na"}:
                strength = "N/A"
            if salt and salt != "na":
                cleaned.append({"salt": salt, "strength": strength})
        elif isinstance(entry, str) and entry.strip():
            salt = re.sub(r"\s+na$", "", entry.strip().lower()).strip()
            if salt and salt != "na":
                cleaned.append({"salt": salt, "strength": "N/A"})
    return cleaned


def normalize_vehicle(raw_vehicle: Any, product_name: str) -> str:
    val = str(raw_vehicle or "").strip().upper()
    name_upper = product_name.upper()

    if "TABLET/CAPSULE" in val:
        return "CAPSULE" if "CAP" in name_upper else "TABLET"
    if val and val not in {"NAN", "NONE", "ANY", "OTHER", "UNKNOWN"}:
        return val

    if "SUSPENSION" in name_upper:
        return "LOTION"

    for kw in [
        "SHAMPOO", "CONDITIONER", "FACEWASH", "FACE WASH", "SUNSCREEN",
        "MOISTURIZER", "MOISTURISER", "OINTMENT", "CREAM", "LOTION",
        "SERUM", "GEL", "FOAM", "SOAP", "POWDER", "CAPSULE", "TABLET",
    ]:
        if kw in name_upper:
            return kw.replace(" ", "")
    return "CREAM"


def load_drug_catalog(csv_path: Path) -> tuple[list[dict[str, Any]], dict[str, str | None]]:
    df = pd.read_csv(csv_path)
    cols = list(df.columns)

    col_map = {
        "product_name": _find_col(cols, ["product_name", "name", "brand_name", "medicine_name"]),
        "display_name": _find_col(cols, ["display_name", "clean_name"]),
        "product_price": _find_col(cols, ["product_price", "price", "mrp"]),
        "vehicle": _find_col(cols, ["vehicle", "form", "dosage_form"]),
        "parsed_entities": _find_col(cols, ["parsed_entities", "entities", "salts", "composition_json"]),
        "salt_composition": _find_col(cols, ["salt_composition"]),
        "sub_category": _find_col(cols, ["sub_category"]),
    }

    if not col_map["product_name"]:
        raise ValueError(f"Could not locate product_name column in {csv_path}. Found columns: {cols}")

    # Deduplicate rows so each prescription samples distinct products
    dedup_cols = [col_map["product_name"]]
    if col_map["salt_composition"]:
        dedup_cols.append(col_map["salt_composition"])
    df = df.drop_duplicates(subset=dedup_cols).copy()

    catalog: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        prod_name = str(row[col_map["product_name"]]).strip()
        if not prod_name or prod_name.lower() == "nan":
            continue

        # Exclude non-dermatology dosage forms (eye/ear/nasal/inhaler/vaginal/gargle)
        sub_cat = str(row[col_map["sub_category"]]).strip() if col_map["sub_category"] else ""
        if sub_cat in NON_DERM_SUBCATS or any(kw in prod_name.lower() for kw in NON_DERM_KEYWORDS):
            continue

        vehicle = normalize_vehicle(
            row[col_map["vehicle"]] if col_map["vehicle"] else "",
            prod_name,
        )
        salts = parse_entities_field(
            row[col_map["parsed_entities"]] if col_map["parsed_entities"] else None
        )
        if not salts:
            continue

        # Strip currency symbols (e.g. '₹5.89') before parsing float
        price_val = None
        if col_map["product_price"] and pd.notna(row[col_map["product_price"]]):
            price_str = re.sub(r"[^\d.]", "", str(row[col_map["product_price"]]))
            if price_str:
                try:
                    price_val = round(float(price_str), 2)
                except ValueError:
                    price_val = None

        display_name = (
            str(row[col_map["display_name"]]).strip()
            if col_map["display_name"] and pd.notna(row[col_map["display_name"]])
            else prod_name
        )

        # Avoid duplicated vehicle words (e.g. "TABLET NEOPRED" instead of "TABLET NEOPRED TABLET")
        brand_upper = re.sub(r"\s+", " ", prod_name.upper()).strip()
        core_name = re.sub(r"\s+", " ", VEHICLE_STRIP_RE.sub("", brand_upper)).strip(" -")
        raw_line = f"{vehicle} {core_name}" if core_name else brand_upper

        salt_line_parts = []
        for s in salts:
            s_name = s["salt"].upper()
            st = s["strength"].upper()
            if st != "N/A" and st not in s_name:
                salt_line_parts.append(f"{s_name} {st}")
            else:
                salt_line_parts.append(s_name)
        salt_line = ", ".join(salt_line_parts)

        catalog.append({
            "product_name": prod_name,
            "display_name": display_name,
            "brand": brand_upper,
            "raw_line": raw_line,
            "salt_line": salt_line,
            "vehicle": vehicle,
            "salts": salts,
            "product_price": price_val,
        })

    if not catalog:
        raise ValueError(f"No valid dermatology rows found in {csv_path}. Mapping: {col_map}")
    return catalog, col_map


# ---------------------------------------------------------------------------
# Stratified Allocation Helpers
# ---------------------------------------------------------------------------
def build_stratified_list(total: int, weights: list[tuple[str, float]], rng: random.Random) -> list[str]:
    result: list[str] = []
    allocated = 0
    for idx, (label, weight) in enumerate(weights):
        if idx == len(weights) - 1:
            count = total - allocated
        else:
            count = int(round(total * weight))
            allocated += count
        result.extend([label] * max(0, count))
    while len(result) < total:
        result.append(weights[0][0])
    result = result[:total]
    rng.shuffle(result)
    return result


# ---------------------------------------------------------------------------
# Multilingual Posology Generator
# ---------------------------------------------------------------------------
def build_posology(vehicle: str, lang: str, rng: random.Random) -> dict[str, Any]:
    is_oral = vehicle in ORAL_VEHICLES

    if is_oral:
        if lang == "en":
            dose = "1 Capsule" if vehicle == "CAPSULE" else "1 Tablet"
            freq_code, freq_label = rng.choice([
                ("✓-X-X", "Morning"),
                ("X-✓-X", "Afternoon"),
                ("X-X-✓", "Bedtime"),
                ("✓-X-✓", "Morning & Night"),
                ("Alternate Day", ""),
            ])
            instr = rng.choice(["After Food", "After Dinner", "After Breakfast", "Before Food"])
            dur_days = rng.choice([15, 30, 60, 90, 120])
            duration = f"{dur_days} Day(s)"
            qty = str(dur_days if "✓-X-✓" not in freq_code else dur_days * 2)
        elif lang == "en_gu":
            dose = "1 Capsule" if vehicle == "CAPSULE" else "1 ટેબ્લેટ"
            freq_code, freq_label = rng.choice([
                ("✓-X-X", "સવારે"),
                ("X-✓-X", "બપોરે"),
                ("X-X-✓", "રાત્રે સુવાના સમયે"),
                ("✓-X-✓", "સવારે - સાંજે"),
            ])
            instr = rng.choice(["જમ્યા પછી", "નાસ્તા પછી", "જમ્યા પહેલા"])
            dur_label, qty = rng.choice([
                ("30 દિવસ", "30"),
                ("90 દિવસ", "90"),
                ("120 દિવસ", "120"),
                ("3 મહિના", "90"),
                ("6 મહિના", "180"),
            ])
            duration = dur_label
        else:  # en_hi
            dose = "1 कैप्सूल" if vehicle == "CAPSULE" else "1 टैबलेट"
            freq_code, freq_label = rng.choice([
                ("✓-X-X", "सुबह"),
                ("X-✓-X", "दोपहर"),
                ("X-X-✓", "रात को सोने से पहले"),
                ("✓-X-✓", "सुबह - शाम"),
            ])
            instr = rng.choice(["खाने के बाद", "नाश्ते के बाद", "खाने से पहले"])
            dur_label, qty = rng.choice([
                ("30 दिन", "30"),
                ("60 दिन", "60"),
                ("90 दिन", "90"),
                ("3 महीने", "90"),
            ])
            duration = dur_label

        freq_note = rng.choice(ORAL_FREQ_NOTES) if rng.random() < 0.25 else None
    else:
        dose = "To Apply"
        if lang == "en":
            if vehicle in {"SHAMPOO", "CONDITIONER"}:
                freq_code, freq_label = ("Twice a Week", "")
            else:
                freq_code, freq_label = rng.choice([
                    ("✓-X-X", "Morning"),
                    ("X-X-✓", "Bedtime"),
                    ("✓-X-✓", "Twice a Day"),
                ])
            instr = rng.choice(["As Directed", "Apply on affected area", "Apply externally"])
            duration = rng.choice(["Till next visit", "30 Day(s)", "60 Day(s)"])
        elif lang == "en_gu":
            if vehicle in {"SHAMPOO", "CONDITIONER"}:
                freq_code, freq_label = ("સપ્તાહમાં બે વખત", "")
            else:
                freq_code, freq_label = rng.choice([
                    ("✓-X-X", "સવારે"),
                    ("X-X-✓", "રાત્રે સુવાના સમયે"),
                    ("✓-X-✓", "સવારે - સાંજે"),
                ])
            instr = "ડોક્ટરના માર્ગદર્શન મુજબ"
            duration = rng.choice(["આગામી મુલાકાત સુધી", "ઉપયોગ ચાલુ રાખો", "60 દિવસ", "8 અઠવાડિયા"])
        else:  # en_hi
            if vehicle in {"SHAMPOO", "CONDITIONER"}:
                freq_code, freq_label = ("सप्ताह में दो बार", "")
            else:
                freq_code, freq_label = rng.choice([
                    ("✓-X-X", "सुबह"),
                    ("X-X-✓", "रात को सोने से पहले"),
                    ("✓-X-✓", "सुबह - शाम"),
                ])
            instr = "डॉक्टर के निर्देशानुसार"
            duration = rng.choice(["अगली मुलाकात तक", "30 दिन", "60 दिन", "जारी रखें"])

        qty = "-"
        freq_note = rng.choice(TOPICAL_FREQ_NOTES) if rng.random() < 0.45 else None

    return {
        "dose": dose,
        "freq_code": freq_code,
        "freq_label": freq_label,
        "instruction": instr,
        "duration": duration,
        "qty": qty,
        "freq_note": freq_note,
    }


# ---------------------------------------------------------------------------
# Payload Assembly (Prescription + Hard Negative)
# ---------------------------------------------------------------------------
def build_doctor_and_patient(fake: Faker, rng: random.Random) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    doc_first = fake.first_name()
    doc_last = fake.last_name()
    doctor_name = f"Dr. {doc_first} {doc_last}"
    clinic_name = f"{rng.choice(CLINIC_PREFIXES)} by {doctor_name}"
    reg_no = f"G{rng.randint(10000, 99999)},G{rng.randint(10000, 99999)}"
    phone_clinic = f"9{rng.randint(100000000, 999999999)}"
    email_clinic = f"{doc_first.lower()}.{doc_last.lower()}.derm@example.com"
    city = rng.choice(["Ahmedabad", "Gandhinagar", "Surat", "Vadodara", "Rajkot", "Mumbai", "Pune", "Jaipur"])
    pincode = str(rng.randint(380001, 411060))
    addr_line1 = f"{rng.randint(101, 612)}, {fake.street_name()}"
    addr_line2 = f"Near {fake.city()} Cross Roads, {city}-{pincode}, India"

    gender = rng.choice(["Male", "Female"])
    title = "Mr." if gender == "Male" else rng.choice(["Ms.", "Mrs."])
    patient_name = f"{title} {fake.first_name_male() if gender == 'Male' else fake.first_name_female()} {fake.last_name()}"
    age_yr = rng.randint(14, 68)
    patient_phone = f"8{rng.randint(100000000, 999999999)}"
    uid = f"{rng.choice(['G', 'B', 'Y', 'A'])}{rng.randint(1000000, 9999999)}"

    visit_dt = datetime(2025, 1, 1) + timedelta(
        days=rng.randint(0, 680),
        hours=rng.randint(9, 20),
        minutes=rng.randint(0, 59),
    )

    doctor = {
        "name": doctor_name,
        "clinic_name": clinic_name,
        "reg_no": reg_no,
        "qualifications": rng.choice(DOCTOR_QUALIFICATIONS),
        "address_line1": addr_line1,
        "address_line2": addr_line2,
        "phone": phone_clinic,
        "email": email_clinic,
        "emr_footer": rng.choice(EMR_FOOTERS),
    }

    patient = {
        "name": patient_name,
        "age": f"{age_yr}yr",
        "gender": gender,
        "mobile": patient_phone,
        "uid": uid,
        "visit_date": visit_dt.strftime("%d/%m/%Y;%I:%M %p").lower(),
        "generated_on": (visit_dt + timedelta(minutes=rng.randint(2, 20))).strftime("%d/%m/%Y %I:%M %p"),
    }

    noise_tokens = [
        doctor_name,
        clinic_name,
        reg_no,
        phone_clinic,
        email_clinic,
        addr_line1,
        addr_line2,
        patient_name,
        patient_phone,
        uid,
    ]
    return doctor, patient, noise_tokens


def build_hard_negative_payload(
    image_id: str,
    split: str,
    language: str,
    difficulty: str,
    fake: Faker,
    rng: random.Random,
) -> dict[str, Any]:
    doctor, patient, noise_tokens = build_doctor_and_patient(fake, rng)
    doc_kind = rng.choice(["lab_report", "clinic_bill"])

    if doc_kind == "lab_report":
        lab_rows = [
            {"test": "Hemoglobin (Hb)", "value": f"{rng.uniform(11.0, 16.2):.1f}", "unit": "g/dL", "ref": "12.0 - 16.0"},
            {"test": "WBC Count", "value": str(rng.randint(4500, 10500)), "unit": "/cumm", "ref": "4000 - 11000"},
            {"test": "Platelet Count", "value": f"{rng.uniform(1.6, 4.0):.2f}", "unit": "lakhs/cumm", "ref": "1.5 - 4.5"},
            {"test": "TSH (Ultra-sensitive)", "value": f"{rng.uniform(0.8, 5.2):.2f}", "unit": "uIU/mL", "ref": "0.4 - 4.5"},
            {"test": "Vitamin B12", "value": str(rng.randint(180, 720)), "unit": "pg/mL", "ref": "211 - 911"},
            {"test": "Vitamin D (25-OH)", "value": f"{rng.uniform(14.0, 58.0):.1f}", "unit": "ng/mL", "ref": "30.0 - 100.0"},
        ]
        noise_tokens.extend([r["test"] for r in lab_rows])
        extra_content = {"doc_kind": "lab_report", "lab_rows": lab_rows}
    else:
        fee = rng.choice([500, 700, 800, 1000, 1200])
        proc_fee = rng.choice([0, 1500, 2500])
        bill_items = [{"item": "Dermatology Specialist Consultation", "amount": fee}]
        if proc_fee:
            bill_items.append({"item": rng.choice(["Chemical Peel Session", "Dermoscopy Evaluation", "Patch Test"]), "amount": proc_fee})
        noise_tokens.extend([b["item"] for b in bill_items])
        extra_content = {
            "doc_kind": "clinic_bill",
            "bill_items": bill_items,
            "total_inr": sum(b["amount"] for b in bill_items),
        }

    return {
        "image_id": image_id,
        "file_name": f"images/{image_id}.jpg",
        "split": split,
        "template_id": f"HN_{doc_kind}",
        "language": language,
        "difficulty": difficulty,
        "is_hard_negative": True,
        "doctor": doctor,
        "patient": patient,
        "headers": TABLE_HEADERS[language],
        "chief_complaints": None,
        "diagnosis": None,
        "medications": [],
        "investigations": [],
        "followup_days": None,
        "hard_negative_content": extra_content,
        "injected_noise_tokens": noise_tokens,
    }


def build_prescription_payload(
    image_id: str,
    split: str,
    template_id: str,
    language: str,
    difficulty: str,
    catalog: list[dict[str, Any]],
    fake: Faker,
    rng: random.Random,
) -> dict[str, Any]:
    doctor, patient, noise_tokens = build_doctor_and_patient(fake, rng)
    complaints, diagnosis = rng.choice(COMPLAINTS_AND_DIAGNOSES)

    # Number of drugs scales naturally with template layout
    if template_id == "T4_dense_derm":
        n_meds = rng.randint(7, 9)
    elif template_id == "T3_compact_a5":
        n_meds = rng.randint(2, 4)
    else:
        n_meds = rng.randint(3, 7)

    sampled_drugs = rng.sample(catalog, k=min(n_meds, len(catalog)))
    medications: list[dict[str, Any]] = []
    for idx, drug in enumerate(sampled_drugs, start=1):
        posology = build_posology(drug["vehicle"], language, rng)
        medications.append({
            "index": idx,
            "raw_line": drug["raw_line"],
            "salt_line": drug["salt_line"],
            "brand": drug["brand"],
            "product_name": drug["product_name"],
            "vehicle": drug["vehicle"],
            "salts": drug["salts"],
            "product_price": drug["product_price"],
            "posology": posology,
        })

    # Add realistic Investigation Advised noise block (70% of full prescriptions)
    investigations: list[dict[str, str]] = []
    if template_id != "T3_compact_a5" and rng.random() < 0.70:
        n_inv = rng.randint(1, 3)
        investigations = rng.sample(INVESTIGATION_BLOCKS, k=n_inv)
        for inv in investigations:
            noise_tokens.append(inv["title"])
            noise_tokens.extend([tok.strip() for tok in inv["details"].split(",")[:4]])

    return {
        "image_id": image_id,
        "file_name": f"images/{image_id}.jpg",
        "split": split,
        "template_id": template_id,
        "language": language,
        "difficulty": difficulty,
        "is_hard_negative": False,
        "doctor": doctor,
        "patient": patient,
        "headers": TABLE_HEADERS[language],
        "chief_complaints": complaints,
        "diagnosis": diagnosis,
        "medications": medications,
        "investigations": investigations,
        "followup_days": rng.choice([15, 30, 45, 60, 90]),
        "hard_negative_content": None,
        "injected_noise_tokens": noise_tokens,
    }


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Part 1: Synthetic Prescription Payload & Ground-Truth Sampler")
    parser.add_argument("--csv", type=Path, required=True, help="Path to 1mg_dermatology_parsed.csv")
    parser.add_argument("--out-dir", type=Path, default=Path("synthetic_rx_dataset"), help="Output dataset root")
    parser.add_argument("--total", type=int, default=5000, help="Total synthetic samples to generate")
    parser.add_argument("--hard-neg-ratio", type=float, default=0.05, help="Fraction of zero-drug hard negatives")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--inspect", action="store_true", help="Print detected columns + 3 parsed drugs and exit")
    args = parser.parse_args()

    catalog, col_map = load_drug_catalog(args.csv)
    print(f"[INFO] Loaded {len(catalog):,} valid unique dermatology drugs from {args.csv}")
    print(f"[INFO] Column mapping: {col_map}")

    if args.inspect:
        print("\n[INSPECT] First 3 parsed catalog records:")
        for sample in catalog[:3]:
            print(json.dumps(sample, indent=2, ensure_ascii=False))
        return

    rng = random.Random(args.seed)
    fake = Faker("en_IN")
    Faker.seed(args.seed)

    manifests_dir = args.out_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    splits = build_stratified_list(args.total, SPLIT_WEIGHTS, rng)
    templates = build_stratified_list(args.total, TEMPLATE_WEIGHTS, rng)
    languages = build_stratified_list(args.total, LANGUAGE_WEIGHTS, rng)
    difficulties = build_stratified_list(args.total, DIFFICULTY_WEIGHTS, rng)

    n_hard_neg = int(round(args.total * args.hard_neg_ratio))
    hard_neg_flags = [True] * n_hard_neg + [False] * (args.total - n_hard_neg)
    rng.shuffle(hard_neg_flags)

    payloads_path = manifests_dir / "payloads.jsonl"
    gt_jsonl_path = manifests_dir / "ground_truth.jsonl"
    gt_csv_path = manifests_dir / "ground_truth.csv"

    stats_split = Counter()
    stats_template = Counter()
    stats_lang = Counter()
    stats_diff = Counter()
    total_meds = 0

    with (
        payloads_path.open("w", encoding="utf-8") as f_pay,
        gt_jsonl_path.open("w", encoding="utf-8") as f_gt,
        gt_csv_path.open("w", encoding="utf-8", newline="") as f_csv,
    ):
        csv_writer = csv.DictWriter(
            f_csv,
            fieldnames=[
                "image_id", "file_name", "split", "template_id", "language",
                "difficulty", "is_hard_negative", "num_medications",
                "expected_salts", "expected_vehicles",
            ],
        )
        csv_writer.writeheader()

        for i in range(1, args.total + 1):
            image_id = f"rx_{i:04d}"
            sp = splits[i - 1]
            tpl = templates[i - 1]
            lang = languages[i - 1]
            diff = difficulties[i - 1]
            is_hn = hard_neg_flags[i - 1]

            if is_hn:
                payload = build_hard_negative_payload(image_id, sp, lang, diff, fake, rng)
            else:
                payload = build_prescription_payload(image_id, sp, tpl, lang, diff, catalog, fake, rng)

            # 1. Full payload (consumed by Part 2 & Part 3 Jinja2 + Playwright renderer)
            f_pay.write(json.dumps(payload, ensure_ascii=False) + "\n")

            # 2. Clean Format 1 evaluation ground truth
            gt_record = {
                "image_id": payload["image_id"],
                "file_name": payload["file_name"],
                "split": payload["split"],
                "template_id": payload["template_id"],
                "language": payload["language"],
                "difficulty": payload["difficulty"],
                "is_hard_negative": payload["is_hard_negative"],
                "medications": [
                    {
                        "raw_line": m["raw_line"],
                        "salt_line": m["salt_line"],
                        "brand": m["brand"],
                        "vehicle": m["vehicle"],
                        "salts": m["salts"],
                    }
                    for m in payload["medications"]
                ],
                "injected_noise_tokens": payload["injected_noise_tokens"],
            }
            f_gt.write(json.dumps(gt_record, ensure_ascii=False) + "\n")

            # 3. Flat CSV summary
            unique_salts = sorted({s["salt"] for m in payload["medications"] for s in m["salts"]})
            unique_vehicles = sorted({m["vehicle"] for m in payload["medications"]})
            csv_writer.writerow({
                "image_id": image_id,
                "file_name": payload["file_name"],
                "split": sp,
                "template_id": payload["template_id"],
                "language": lang,
                "difficulty": diff,
                "is_hard_negative": is_hn,
                "num_medications": len(payload["medications"]),
                "expected_salts": "|".join(unique_salts),
                "expected_vehicles": "|".join(unique_vehicles),
            })

            stats_split[sp] += 1
            stats_template[payload["template_id"]] += 1
            stats_lang[lang] += 1
            stats_diff[diff] += 1
            total_meds += len(payload["medications"])

    print(f"\n[DONE] Generated {args.total:,} payloads ({total_meds:,} total drug lines).")
    print(f"  Payloads : {payloads_path}")
    print(f"  GT JSONL : {gt_jsonl_path}")
    print(f"  GT CSV   : {gt_csv_path}")
    print(f"  Splits   : {dict(stats_split)}")
    print(f"  Languages: {dict(stats_lang)}")
    print(f"  Tiers    : {dict(stats_diff)}")
    print(f"  Templates: {dict(stats_template)}")


if __name__ == "__main__":
    main()
