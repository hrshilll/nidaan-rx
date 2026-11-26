# Nidaan-Rx v2.0 — AI-Supported Dermatology Triage, Clinical Safety & Jan Aushadhi Generic Matching System

**Nidaan-Rx v2.0** is a full-stack clinical decision-support platform built for Indian dermatology workflows. It combines multi-angle skin lesion computer vision (**PanDerm-Base** with Gated Attention-MIL), **Stage-2 Bayesian clinical refinement**, a deterministic **12-Gate Clinical Safety Engine**, and **PaddleOCR-powered prescription scanning** to match commercial dermatology formulations with verified **Jan Aushadhi (PMBJP)** generic alternatives.

> **Medical Disclaimer:** Nidaan-Rx is an academic research prototype and clinical decision-support system. It does **not** provide a medical diagnosis or replace in-person evaluation by a licensed dermatologist.

---

## Key Features

1. **Multi-View Vision Pipeline & Pre-Inference Quality Gate**
   - **Pre-Inference Quality Gate (<50 ms):** OpenCV/YCrCb/HSV checks for resolution (>=224x224), CLAHE-normalized Laplacian sharpness (calibrated for Fitzpatrick I-VI skin tones), flash glare, underexposure, and skin/scalp pixel coverage.
   - **PanDerm-Base + Gated Attention-MIL:** Pools up to 3 lesion angles with patient metadata (Fitzpatrick I-VI / Monk Skin Tone 1-10, anatomical site, duration, itch/pain scores, scaling, barrier status, triggers, and red flags).
   - **Mahalanobis OOD Detection:** Flags out-of-distribution inputs outside the calibrated embedding manifold and triggers a clinical hard-stop.

2. **Stage-2 Bayesian Clinical Refinement**
   - Refines Stage-1 vision probabilities using bounded likelihood ratios across anatomical-site compatibility, symptom severity (itch, pain/burning), lesion duration, surface morphology, and reported triggers.

3. **Deterministic 12-Gate Clinical Safety Engine (G1-G12)**
   - **G1-G3 (PMBJP Chemical Equivalence):** Verifies exact salt-set equality, strict numeric dosage-strength alignment, and compatible dosage vehicle against the Jan Aushadhi catalog.
   - **G4 / G4B / G4C (Hard Stops & Discordance Guards):** Suppresses treatment cards and escalates care on urgent red flags, OOD flags, close cross-family differentials, or asymptomatic visual-clinical contradictions.
   - **G5 (Steroid-on-Infection Block):** Blocks topical corticosteroids when fungal, viral, bacterial, or parasitic infections appear in the differential.
   - **G6-G9 (Patient-Specific Safety):** Enforces pediatric/age ceilings, pregnancy/lactation restrictions, systemic condition checks, allergy/active conflicts, longitudinal non-response escalation, and weight-based dosing limits.
   - **G10 (Bidirectional Site & Vehicle Gate):** Blocks potent topical steroids on delicate sites (face, eyelids, groin folds) and restricts scalp-only shampoos/lotions to hair-bearing sites (scalp, beard).
   - **G11-G12 (DDI & Photosensitivity Guard):** Screens oral medications for serious drug-drug interactions and enforces night-only + SPF 30+ precautions on photosensitizing actives.

4. **Prescription OCR & Jan Aushadhi (PMBJP) Matching**
   - Extracts medication names, strengths, and dosage forms from uploaded prescriptions via **PaddleOCR**, matching commercial 1mg dermatology SKUs (13,432 entries) against verified Jan Aushadhi generics (160 SKUs) with cost-savings estimates.
   - Includes rx_synth/ (Jinja2 + Playwright + Augraphy/OpenCV) to generate synthetic Indian prescription images for OCR benchmarking.

5. **Longitudinal Follow-Up & Clinical PDF Reports**
   - Persists authenticated patient cases and follow-up visits in SQLite and exports structured clinical triage audit PDFs.

---

## Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend** | Next.js 16, React, TypeScript, Tailwind CSS |
| **Backend API** | Python 3.12+, FastAPI, Uvicorn, Pydantic v2 |
| **Database & Auth** | SQLite, JWT Bearer Authentication, Password Hashing |
| **Vision & ML** | PyTorch, PanDerm-Base, Gated Attention-MIL, Mahalanobis OOD, OpenCV, NumPy, Pillow |
| **Clinical Engine** | Bayesian Knowledge Graph + Deterministic 12-Gate Safety Engine |
| **OCR & Generics** | PaddleOCR, Jan Aushadhi (PMBJP) & 1mg Reference Catalogs |
| **Synthetic Data & Eval** | rx_synth, pytest, HTTPX, Matplotlib, Seaborn |

---

## Project Structure

```text
Nidaan-Rx/
├── backend/
│   ├── app/
│   │   ├── api/                # FastAPI routers (auth, triage, Rx OCR, follow-ups, PDF export)
│   │   ├── core/               # Knowledge graph, Bayesian modifier, 12-gate safety engine, schemas
│   │   ├── services/           # Inference service, Rx scanner, PDF generator, SQLite persistence
│   │   ├── vision/             # PanDerm-Base extractor, quality gate, taxonomy mapper
│   │   └── main.py             # FastAPI entrypoint
│   ├── artifacts/              # Production v2 model artifact (nidaan_v2_panderm_base_stage4.pt)
│   ├── scripts/                # Training & calibration utilities
│   └── tests/                  # Pytest suites (API, quality gate, safety gates, regressions)
├── data/
│   ├── processed/              # Parsed PMBJP & 1mg dermatology catalogs + RL Q-table
│   └── raw/                    # Lightweight reference CSVs (PMJAY list, symptom dataset)
├── design-system/              # UI design specification
├── frontend/                   # Next.js 16 + TypeScript + Tailwind CSS web app
├── reports/                    # Evaluation summary and publication figures
├── rx_synth/                   # Synthetic Indian prescription dataset generator
├── scripts/                    # Report figure generation scripts
├── dev.sh                      # One-command macOS/Linux dev launcher
├── requirements.txt            # Cross-platform Python dependencies
└── README.md
```

---

## Prerequisites

- **Python 3.12** (recommended for PyTorch & PaddleOCR compatibility)
- **Node.js 20 LTS+** and **npm**
- **Git**

---

## Installation & Running on macOS (MacBook — zsh)

1. **Clone the repository**
```bash
git clone https://github.com/hrshilll/nidaan-rx.git
cd Nidaan-Rx
```

2. **Create and activate the Python virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. **Install Frontend dependencies**
```bash
cd frontend
npm install
cd ..
```

4. **Start the application**
   - **Quick launch (dev.sh):**
```bash
chmod +x dev.sh
./dev.sh
```
   - **Or run Backend & Frontend in two terminals:**
     - Terminal 1 (Backend — port 8000):
```bash
source .venv/bin/activate
PYTHONPATH=backend uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
     - Terminal 2 (Frontend — port 3000):
```bash
cd frontend
npm run dev
```

5. **Open in browser:** Web UI at http://localhost:3000 | API Docs at http://localhost:8000/docs

---

## Installation & Running on Windows 10 / 11 (PowerShell)

1. **Clone the repository**
```powershell
git clone https://github.com/hrshilll/nidaan-rx.git
cd Nidaan-Rx
```

2. **Create and activate the Python virtual environment**
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
   > If PowerShell blocks script execution, run Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass first, then activate .venv again.

3. **Install Frontend dependencies**
```powershell
cd frontend
npm install
cd ..
```

4. **Start Backend & Frontend in two PowerShell windows**
   - PowerShell Window 1 (Backend — port 8000):
```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "backend"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
   - PowerShell Window 2 (Frontend — port 3000):
```powershell
cd frontend
npm run dev
```

5. **Open in browser:** Web UI at http://localhost:3000 | API Docs at http://localhost:8000/docs

---

## Running Automated Tests

- **macOS (zsh):**
```bash
source .venv/bin/activate
PYTHONPATH=backend pytest backend/tests -v
```

- **Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "backend"
pytest backend/tests -v
```

---

## Notes on First Run & Excluded Files
- On first startup, the backend automatically initializes data/nidaan_v2_cases.db and generates a local JWT signing key at data/.auth_secret (both excluded by .gitignore).
- Heavy datasets (synthetic_rx_dataset/, data/raw/medicine_data.csv) and local presentation files (Media/) are excluded from Git; run the scripts in rx_synth/ to regenerate synthetic prescription images locally.
