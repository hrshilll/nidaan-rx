# Nidaan-Rx 💊
> **Clinical Decision Support & Generic Substitution Engine**

Nidaan-Rx is an AI-powered pharmaceutical platform designed to bridge the gap between expensive commercial dermatology brands and affordable Jan Aushadhi (PMBJP) generic substitutes in the Indian market. 

Built as an academic M.Sc. IT (AI/ML) project, it features a multi-task Deep Learning model for symptom analysis, OCR for prescription scanning, a Reinforcement Learning agent for dynamic ranking, and strict deterministic safety gates to ensure clinical equivalence.

---

## ⚡ Key Features

* **🤖 AI Symptom Checker (Deep Learning):** A PyTorch Multi-Task Neural Network that predicts optimal Active Pharmaceutical Ingredients (APIs) and clinical vehicle form-factors based on skin type and condition severity.
* **🔍 Deterministic Matching Engine:** Evaluates commercial drugs against the PMBJP database using strict safety gates ensuring exact subset/superset chemical matches, dosage alignment, and vehicle compatibility.
* **👁️ Prescription OCR Scanner:** Leverages PaddleOCR and Levenshtein distance fuzzy-matching (with NLP noise filtering) to extract and autocorrect messy text from physical prescriptions or medicine boxes.
* **🧠 Contextual Ranking (Reinforcement Learning):** An Epsilon-Greedy Multi-Armed Bandit agent dynamically ranks generic substitute results based on live pharmacist selection behavior (Q-Learning).
* **📋 Clinical Posology Guide:** Provides detailed usage instructions (how, when, and duration to apply) for predicted treatments.

## 🛠️ Technology Stack
* **Language:** Python 3.12+
* **Deep Learning:** PyTorch (Multi-Task DNN, Cross-Entropy Loss, Adam Optimizer)
* **Reinforcement Learning:** Q-Learning (Epsilon-Greedy, $\epsilon=0.1$)
* **Computer Vision:** PaddleOCR v3.7+, Pillow
* **NLP & Data Processing:** Pandas, TheFuzz (Levenshtein distance), Regex (NER)
* **Frontend UI:** Streamlit (Neumorphic Design)

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.12+ installed. It is highly recommended to use a virtual environment.

### 2. Installation & Setup

**For Windows:**
Open PowerShell or Command Prompt and run:
```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**For macOS/Linux:**
Open Terminal and run:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Data Preparation
Ensure the raw data files (`medicine_data.csv` and `PMJAY_list.csv`) are placed in the `data/raw/` directory.
Run the Data Engineering pipeline to clean the data, normalize chemical esters to base salts, and extract JSON entities:
```bash
python src/build_dermatology_mvp.py
```

### 4. Run the Web Application
Launch the Streamlit UI (works on both Windows and macOS):
```bash
streamlit run src/app.py
```
*The application will open automatically in your default web browser at `http://localhost:8501`.*

---

## 📁 Project Structure
```text
nidaan_rx/
│
├── data/
│   ├── processed/          # Cleaned JSON entities and RL Q-Table state
│   └── raw/                # Unstructured Kaggle and PMBJP CSVs
│
├── src/
│   ├── app.py                     # Main Streamlit Web UI, PyTorch DNN, RL Agent, OCR
│   └── build_dermatology_mvp.py   # Data Engineering and Parsing Pipeline
│
├── .gitignore
├── requirements.txt
└── README.md
```

## 🎓 Academic Context (M.Sc. IT AI/ML)
This repository contains the codebase for a Master's Thesis project. It explicitly demonstrates the architectural combination of **Probabilistic AI** (Deep Learning classification and OCR feature extraction) and **Deterministic Logic** (Hard-coded safety gates for clinical verification). 

## ⚠️ Disclaimer
**Nidaan-Rx is an academic AI prototype.** The Deep Learning predictions and OCR extractions are for demonstration purposes only. Always consult a registered dermatologist or pharmacist before making medical substitutions or dispensing medication.
