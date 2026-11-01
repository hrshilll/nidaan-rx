# PRISM-Rx 💊
> **Clinical Decision Support & Generic Substitution Engine**

PRISM-Rx is an AI-powered pharmaceutical platform designed to bridge the gap between expensive commercial dermatology brands and affordable Jan Aushadhi generic substitutes in the Indian market. 

Built as an academic M.Sc. IT (AI/ML) project, it features a multi-task Deep Learning model for symptom analysis, OCR for prescription scanning, and strict deterministic safety gates to ensure clinical equivalence.

---

## ⚡ Key Features

* **🤖 AI Symptom Checker (Deep Learning):** A PyTorch Multi-Task Neural Network that predicts optimal Active Pharmaceutical Ingredients (APIs) and clinical vehicle form-factors based on skin type and condition severity.
* **🔍 Deterministic Matching Engine:** Evaluates commercial drugs against the PMBJP (Jan Aushadhi) database using strict safety gates ensuring exact subset/superset chemical matches, dosage alignment, and vehicle compatibility.
* **👁️ Prescription OCR Scanner:** Leverages PaddleOCR and Levenshtein distance fuzzy-matching to extract and autocorrect messy text from physical prescriptions or medicine boxes.
* **📋 Clinical Posology Guide:** Provides detailed usage instructions (how, when, and duration to apply) for predicted treatments.

## 🛠️ Technology Stack
* **Language:** Python 3.12+
* **Deep Learning:** PyTorch (Multi-Task DNN)
* **Computer Vision:** PaddleOCR, Pillow
* **NLP & Data Processing:** Pandas, TheFuzz (Levenshtein distance), Scikit-Learn
* **Frontend UI:** Streamlit (Neumorphic Design)

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.12+ installed. Create a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Installation
Install the required AI and data science libraries:
```bash
pip install torch pandas scikit-learn streamlit paddlepaddle paddleocr thefuzz Pillow
```

### 3. Data Preparation
The raw data files (`medicine_data.csv` and `PMJAY_list.csv`) must be placed in `data/raw/`.
Run the Data Engineering pipeline to clean and extract the components:
```bash
python src/build_dermatology_mvp.py
```

### 4. Run the Web Application
Launch the Streamlit UI:
```bash
streamlit run src/app.py
```

---

## 📁 Project Structure
```text
prism_rx/
│
├── data/
│   ├── processed/          # Cleaned, parsed JSON entities ready for ML
│   └── raw/                # Unstructured Kaggle and PMBJP CSVs
│
├── src/
│   ├── app.py                     # Main Streamlit Web UI and PyTorch Engine
│   └── build_dermatology_mvp.py   # Data Engineering and Parsing Pipeline
│
├── .gitignore
└── README.md
```

## ⚠️ Disclaimer
**PRISM-Rx is an academic AI prototype.** The Deep Learning predictions and OCR extractions are for demonstration purposes only. Always consult a registered dermatologist or pharmacist before making medical substitutions or dispensing medication.
