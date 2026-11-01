import streamlit as st
import pandas as pd
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from paddleocr import PaddleOCR
from thefuzz import process
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. PAGE CONFIG & ADAPTIVE CSS
# ==========================================
st.set_page_config(page_title="PRISM-Rx", page_icon="💊", layout="wide")

st.markdown("""
    <style>
    .stButton > button {
        border-radius: 8px !important;
        border: 1px solid var(--primary-color) !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.2) !important;
    }
    .stTextInput>div>div>input, .stSelectbox>div>div>div, .stNumberInput>div>div>input {
        border-radius: 8px !important;
    }
    [data-testid="stMetric"] {
        background-color: var(--secondary-background-color);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    hr { border-top: 2px solid var(--secondary-background-color) !important; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. USAGE INSTRUCTIONS (POSOLOGY DICTIONARY)
# ==========================================
POSOLOGY_GUIDE = {
    "adapalene": "🌙 **When:** Once daily at bedtime.\n🧴 **How:** Apply a pea-sized amount to the entire face. Wash hands after.\n⚠️ **Note:** Causes sun sensitivity. Must use sunscreen during the day.",
    "clindamycin": "☀️🌙 **When:** Twice daily (Morning and Night).\n🧴 **How:** Apply a thin layer to the affected area after gently washing with a mild cleanser.\n⏳ **Duration:** Use consistently for 6-8 weeks for visible results.",
    "clotrimazole": "☀️🌙 **When:** Twice daily.\n🧴 **How:** Clean and thoroughly dry the area first. Apply enough to cover the affected area and surrounding skin.\n⏳ **Duration:** Continue for 2 weeks after symptoms disappear to prevent return.",
    "ketoconazole": "🚿 **When:** 2-3 times a week.\n🧴 **How:** Apply to wet scalp/skin, lather, and leave on for 3-5 minutes before rinsing thoroughly.\n⏳ **Duration:** Use for 2-4 weeks, then once a week for maintenance.",
    "hydrocortisone": "☀️🌙 **When:** 1-2 times daily.\n🧴 **How:** Apply a very thin layer only to the itchy/inflamed area and rub in gently.\n⚠️ **Note:** Do not use for more than 7-10 days without consulting a doctor to avoid skin thinning.",
    "clobetasol": "☀️🌙 **When:** 1-2 times daily (Highly potent).\n🧴 **How:** Apply a very thin layer. Do not cover with bandages unless directed by a doctor.\n⏳ **Duration:** Maximum 2 weeks consecutive use.",
    "salicylic acid": "☀️🌙 **When:** 1-2 times daily.\n🧴 **How:** Apply directly to scars/acne. May cause slight peeling or dryness initially.\n⚠️ **Note:** Avoid contact with eyes and mucous membranes.",
    "mometasone": "🌙 **When:** Once daily.\n🧴 **How:** Apply a few drops or a thin film to the affected skin.\n⏳ **Duration:** Use only until the flare-up is controlled.",
    "silver sulfadiazine": "☀️🌙 **When:** 1-2 times daily.\n🧴 **How:** Clean the burn wound. Apply a 3-5mm thick layer using a sterile applicator or gloved hand. Keep the burn covered with the cream at all times.",
    "povidone iodine": "☀️🌙 **When:** 1-3 times daily as needed.\n🧴 **How:** Clean the affected area. Apply a small amount to the cut/scrape. May be covered with a sterile bandage.",
    "azelaic acid": "☀️🌙 **When:** Twice daily.\n🧴 **How:** Gently massage a thin layer into the affected hyperpigmented or scarred skin.",
}

# ==========================================
# 3. DATA & AI MODEL LOADING
# ==========================================
@st.cache_data
def load_data():
    try:
        df_c = pd.read_csv("data/processed/1mg_dermatology_parsed.csv")
        df_p = pd.read_csv("data/processed/pmbjp_dermatology_parsed.csv")
        df_c['display_name'] = df_c['product_name'] + " (" + df_c['vehicle'] + ")"
        
        all_salts = set()
        for entities_json in df_p['parsed_entities']:
            try:
                for item in json.loads(entities_json):
                    all_salts.add(item['salt'].title())
            except:
                pass
        return df_c, df_p, sorted(list(all_salts))
    except FileNotFoundError:
        return pd.DataFrame(), pd.DataFrame(), []

df_comm, df_pmbjp, available_salts = load_data()

@st.cache_resource
def load_ocr_model():
    """Loads PaddleOCR into memory once."""
    return PaddleOCR(use_angle_cls=True, lang='en')



ocr_model = load_ocr_model()

# ==========================================
# 4. SAFETY GATES LOGIC
# ==========================================
def check_safety_gates(target_entities_json, pmbjp_entities_json, target_vehicle, pmbjp_vehicle):
    try:
        t_salts = json.loads(target_entities_json)
        p_salts = json.loads(pmbjp_entities_json)
    except:
        return False, "Data Parse Error"

    if not t_salts or not p_salts:
        return False, "No Active Ingredients Found"

    t_salt_names = {item['salt'] for item in t_salts}
    p_salt_names = {item['salt'] for item in p_salts}
    
    if t_salt_names != p_salt_names:
        return False, "Subset/Superset Error"

    for t, p in zip(t_salts, p_salts):
        s1, s2 = t['strength'].replace('%', ''), p['strength'].replace('%', '')
        if s1 != "N/A" and s2 != "N/A":
            try:
                if float(s1) != float(s2):
                    return False, f"Dosage Mismatch"
            except:
                if s1 != s2:
                    return False, f"Dosage Mismatch"

    # Allow "ANY" vehicle to bypass the vehicle safety gate for the OCR scanner
    if target_vehicle != "ANY" and target_vehicle.upper() != pmbjp_vehicle.upper():
        return True, f"⚠️ Vehicle Mismatch ({target_vehicle} vs {pmbjp_vehicle})"

    return True, "✅ Exact Clinical Match"

def scan_pmbjp(entities_json, vehicle, comm_price=0.0):
    results_found = False
    for _, generic in df_pmbjp.iterrows():
        is_safe, msg = check_safety_gates(entities_json, generic['parsed_entities'], vehicle, generic['vehicle'])
        if is_safe:
            results_found = True
            gov_price = float(generic['MRP'])
            savings = comm_price - gov_price if comm_price > 0 else 0
            
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown(f"**{generic['Generic Name']}**")
                st.caption(f"Code: `{generic['Drug Code']}` | {msg}")
            with col2:
                st.metric("Gov Price", f"₹{gov_price:.2f}")
            with col3:
                if comm_price > 0:
                    st.metric("Savings", f"₹{savings:.2f}", f"{(savings/comm_price)*100:.0f}%", delta_color="normal")
                else:
                    st.metric("Savings", "-", "N/A", delta_color="off")
            st.write("") 
    return results_found

# ==========================================
# 5. DEEP LEARNING MODEL (SYMPTOM CHECKER)
# ==========================================
class PrismMultiTaskDNN(nn.Module):
    def __init__(self, input_size, num_components, num_vehicles):
        super(PrismMultiTaskDNN, self).__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.component_head = nn.Linear(16, num_components)
        self.vehicle_head = nn.Linear(16, num_vehicles)

    def forward(self, x):
        shared_out = self.shared_layer(x)
        return self.component_head(shared_out), self.vehicle_head(shared_out)

@st.cache_resource
def load_and_train_dl_model():
    np.random.seed(42)
    torch.manual_seed(42)
    
    conditions = [
        "Acne (Pimples)", "Cystic Acne (Severe)", "Acne Scars / Dark Spots", 
        "Fungal Infection (Ringworm)", "Dandruff / Flaky Scalp", "Eczema (Atopic Dermatitis)", 
        "Psoriasis", "First-Degree Burn (Fire/Heat)", "Minor Cuts / Scrapes", 
        "Rosacea (Redness)", "Skin Lightening / Pigmentation"
    ]
    skin_types = ["Oily", "Dry", "Combination", "Normal"]
    
    data = []
    for _ in range(3000):
        cond = np.random.choice(conditions)
        skin = np.random.choice(skin_types)
        severity = np.random.randint(1, 11)
        
        if "Burn" in cond:
            comp, veh = "silver sulfadiazine", "CREAM"
        elif "Cuts" in cond:
            comp, veh = "povidone iodine", "OINTMENT"
        elif "Scars" in cond or "Pigmentation" in cond:
            comp, veh = "azelaic acid", "CREAM"
        elif "Acne (Pimples)" in cond:
            comp = "adapalene" if skin == "Oily" else "clindamycin"
            veh = "GEL" if skin == "Oily" else "CREAM"
        elif "Cystic" in cond:
            comp, veh = "clindamycin", "GEL"
        elif "Fungal" in cond:
            comp = "clotrimazole" if severity < 7 else "ketoconazole"
            veh = "POWDER" if skin == "Oily" else "CREAM"
        elif "Dandruff" in cond:
            comp, veh = "ketoconazole", "SHAMPOO"
        elif "Eczema" in cond or "Rosacea" in cond:
            comp = "hydrocortisone" if severity < 5 else "clobetasol"
            veh = "OINTMENT" if skin == "Dry" else "LOTION"
        elif "Psoriasis" in cond:
            comp = "salicylic acid" if severity < 6 else "mometasone"
            veh = "OINTMENT"
        else:
            comp, veh = "hydrocortisone", "CREAM"
            
        data.append([cond, skin, severity, comp, veh])
        
    df = pd.DataFrame(data, columns=["Condition", "Skin_Type", "Severity", "Component", "Vehicle"])
    
    le_cond, le_skin, le_comp, le_veh = LabelEncoder(), LabelEncoder(), LabelEncoder(), LabelEncoder()
    
    X = pd.DataFrame()
    X['Condition'] = le_cond.fit_transform(df['Condition'])
    X['Skin_Type'] = le_skin.fit_transform(df['Skin_Type'])
    X['Severity'] = df['Severity'] / 10.0
    
    y_comp = le_comp.fit_transform(df['Component'])
    y_veh = le_veh.fit_transform(df['Vehicle'])
    
    X_t = torch.tensor(X.values, dtype=torch.float32)
    yc_t = torch.tensor(y_comp, dtype=torch.long)
    yv_t = torch.tensor(y_veh, dtype=torch.long)
    
    model = PrismMultiTaskDNN(3, len(le_comp.classes_), len(le_veh.classes_))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(250):
        optimizer.zero_grad()
        c_out, v_out = model(X_t)
        loss = criterion(c_out, yc_t) + criterion(v_out, yv_t)
        loss.backward()
        optimizer.step()
        
    return model, le_cond, le_skin, le_comp, le_veh, conditions, skin_types

if df_comm.empty:
    st.error("Data files not found.")
    st.stop()

dl_model, le_cond, le_skin, le_comp, le_veh, all_conditions, all_skins = load_and_train_dl_model()

# ==========================================
# 6. UI LAYOUT & TABS
# ==========================================
st.markdown("<h1 style='text-align: center; color: var(--text-color);'>⚡ PRISM-Rx</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem; color: var(--text-color);'>Clinical Decision Support & Generic Substitution Engine</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🤖 Symptom Checker (AI)", "🔍 Brand Search", "🧪 Manual Entry", "📸 OCR Scan"])

# ----------------- TAB 1: SYMPTOM CHECKER (DEEP LEARNING) -----------------
with tab1:
    st.markdown("### Deep Learning Symptom Analysis")
    st.caption("PyTorch Multi-Task Neural Network predicts the optimal Active Ingredient and Vehicle.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        cond_input = st.selectbox("Diagnosis / Condition", sorted(all_conditions))
    with col2:
        skin_input = st.selectbox("Skin Type", sorted(all_skins))
    with col3:
        sev_input = st.slider("Severity Level (1 = Mild, 10 = Severe)", 1, 10, 5)
        
    if st.button("Predict Treatment (AI)", type="primary", key="btn_ai"):
        c_enc = le_cond.transform([cond_input])[0]
        s_enc = le_skin.transform([skin_input])[0]
        sev_norm = sev_input / 10.0
        
        input_tensor = torch.tensor([[c_enc, s_enc, sev_norm]], dtype=torch.float32)
        
        dl_model.eval()
        with torch.no_grad():
            comp_out, veh_out = dl_model(input_tensor)
            pred_comp = le_comp.inverse_transform([torch.argmax(comp_out).item()])[0]
            pred_veh = le_veh.inverse_transform([torch.argmax(veh_out).item()])[0]
            
        st.markdown("---")
        
        col_res1, col_res2 = st.columns(2)
        col_res1.success(f"**Predicted Active Ingredient:**\n### {pred_comp.title()}")
        col_res2.success(f"**Predicted Form Factor (Vehicle):**\n### {pred_veh.upper()}")
        
        st.info("### 📋 Application Instructions (Posology)\n" + POSOLOGY_GUIDE.get(pred_comp, "Apply as directed by physician."))
        
        st.markdown("### 🛒 Recommended Jan Aushadhi Matches")
        entities_json = json.dumps([{"salt": pred_comp.lower(), "strength": "N/A"}])
        
        found = scan_pmbjp(entities_json, pred_veh)
        if not found:
            st.warning(f"AI predicted '{pred_comp.title()} {pred_veh.upper()}', but no direct PMBJP generic match was found in the government inventory.")
            
        st.caption("⚠️ AI Prediction Disclaimer: Always consult a registered dermatologist before dispensing medication.")

# ----------------- TAB 2: BRAND SEARCH -----------------
with tab2:
    st.markdown("### Search Commercial Medicine")
    drug_list = [""] + list(df_comm['display_name'].unique())
    selected_drug = st.selectbox("Type to search branded medicine (e.g., Candid):", options=drug_list)
    
    if selected_drug:
        target = df_comm[df_comm['display_name'] == selected_drug].iloc[0]
        try:
            comm_price = float(str(target['product_price']).replace('₹', '').replace(',', ''))
        except:
            comm_price = 0.0

        st.info(f"**Target:** {target['product_name']} | **MRP:** ₹{comm_price} | **Base:** {target['vehicle']}")
        
        if st.button("Find Jan Aushadhi Match", key="btn_brand"):
            st.markdown("---")
            st.markdown("### 💊 Safe Generic Alternatives")
            if not scan_pmbjp(target['parsed_entities'], target['vehicle'], comm_price):
                st.error("No safe Jan Aushadhi matches found for this exact formulation.")

# ----------------- TAB 3: MANUAL ENTRY -----------------
with tab3:
    st.markdown("### Manual Composition Entry")
    col1, col2 = st.columns(2)
    with col1:
        salt_options = [""] + available_salts
        salt_name = st.selectbox("Select Active Ingredient", options=salt_options)
    with col2:
        strength = st.text_input("Strength (e.g., 1% or N/A)", value="N/A")
        
    vehicle = st.selectbox("Vehicle / Form Factor", ["CREAM", "OINTMENT", "GEL", "LOTION", "POWDER", "SHAMPOO", "SOAP"])
    
    if st.button("Search Generics", key="btn_manual"):
        if salt_name:
            st.markdown("---")
            entities = json.dumps([{"salt": salt_name.lower().strip(), "strength": strength.strip()}])
            if not scan_pmbjp(entities, vehicle):
                st.error("No safe matches found.")
        else:
            st.warning("Please select a salt name from the dropdown.")

# ----------------- TAB 4: OCR (AI VISION) -----------------
with tab4:
    st.markdown("### 👁️ AI Prescription & Medicine Scanner")
    st.caption("Upload a photo of a branded medicine box or a printed prescription. The AI will extract the text, autocorrect typos, and find the Jan Aushadhi generic equivalent.")
    
    uploaded_file = st.file_uploader("Upload Image (JPG/PNG)", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Convert uploaded file to a format PaddleOCR can read
        image = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(image)
        
        col_img, col_res = st.columns([1, 1])
        with col_img:
            st.image(image, caption="Uploaded Document", use_container_width=True)
            
        with col_res:
            with st.spinner("🧠 Initializing Neural Vision Network & Scanning..."):
                try:
                    result = ocr_model.predict(img_array)
                except:
                    result = ocr_model.ocr(img_array, cls=True)
                
            if not result:
                st.error("❌ No text detected in the image.")
            else:
                extracted_texts = []
                for res in result:
                    if hasattr(res, 'keys') or isinstance(res, dict):
                        if 'rec_texts' in res:
                            extracted_texts.extend(res['rec_texts'])
                        elif 'rec_text' in res:
                            extracted_texts.append(res['rec_text'])
                    elif hasattr(res, 'rec_texts'):
                        extracted_texts.extend(res.rec_texts)
                    else:
                        try:
                            for line in res:
                                extracted_texts.append(line[1][0])
                        except:
                            pass
                
                # UI FIX: Hide the raw text dump inside an expander
                with st.expander("🔍 View Raw OCR Text (For Debugging)", expanded=False):
                    st.write(extracted_texts)
                
                st.markdown("#### 🧠 AI Analysis & Matching")
                found_valid_drug = False
                
                # NLP FILTER: Only look at lines that might actually be medicines
                # We look for common medical keywords or chemical strings
                medical_keywords = ['cream', 'gel', 'ointment', 'lotion', 'shampoo', 'tablet', 'capsule', 'acid', 'mg', '%', 'w/w']
                
                potential_meds = []
                for text in extracted_texts:
                    text_lower = text.lower()
                    # Keep the line if it contains a medical keyword or is all uppercase (often used for brand names on scripts)
                    if any(kw in text_lower for kw in medical_keywords) or (text.isupper() and len(text) > 4):
                        potential_meds.append(text)

                if not potential_meds:
                    # If the filter is too strict, fall back to checking everything
                    potential_meds = extracted_texts

                # Scan filtered text against our known active ingredients database
                identified_drugs = []
                for text in potential_meds:
                    if len(text) > 4: 
                        best_match, confidence = process.extractOne(text.title(), available_salts)
                        
                        # We use 85% to prevent hallucinating wrong drugs from random text
                        if confidence >= 85 and best_match not in [d['name'] for d in identified_drugs]:
                            identified_drugs.append({"name": best_match, "confidence": confidence, "raw": text})
                            
                if identified_drugs:
                    found_valid_drug = True
                    for drug in identified_drugs:
                        st.success(f"**Identified Active Ingredient:** {drug['name']} *(Confidence: {drug['confidence']}%)*")
                        
                        entities_json = json.dumps([{"salt": drug['name'].lower(), "strength": "N/A"}])
                        
                        # We pass "ANY" for the vehicle to cast a wide net on the OCR results
                        with st.container():
                            scan_pmbjp(entities_json, vehicle="ANY")
                            st.markdown("---")
                
                if not found_valid_drug:
                    st.warning("No dermatology active ingredients recognized in this image. Ensure the medicine name is clearly visible.")
