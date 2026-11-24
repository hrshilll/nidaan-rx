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
import os
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. PAGE CONFIG & ADAPTIVE CSS
# ==========================================
st.set_page_config(page_title="Nidaan-Rx", page_icon="💊", layout="centered")

st.markdown("""
    <style>
    /* Layout: narrow column for readability */
    .block-container { max-width: 960px; padding-top: 2.5rem; padding-bottom: 4rem; }
    footer { visibility: hidden; }

    /* Header */
    .nx-title { font-size: 1.9rem; font-weight: 700; letter-spacing: -0.02em; margin: 0; color: var(--text-color); }
    .nx-sub   { font-size: 1rem; opacity: 0.65; margin: 0.25rem 0 1.75rem 0; color: var(--text-color); }

    /* Tabs: plain text with an underline for the active one */
    .stTabs [data-baseweb="tab-list"] { gap: 1.75rem; border-bottom: 1px solid rgba(128,128,128,0.25); }
    .stTabs [data-baseweb="tab"] { padding: 0.6rem 0; background: transparent; font-weight: 500; }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem; }

    /* Buttons: quiet, no lift or shadow */
    .stButton > button {
        border-radius: 8px !important;
        border: 1px solid rgba(128,128,128,0.35) !important;
        transition: border-color 0.15s ease-in-out !important;
    }
    .stButton > button:hover { border-color: var(--primary-color) !important; }
    .stButton > button[kind="primary"] { border-color: var(--primary-color) !important; }

    /* Inputs */
    .stTextInput>div>div>input, .stSelectbox>div>div>div, .stNumberInput>div>div>input { border-radius: 8px !important; }

    /* Metrics: flat and left-aligned instead of boxed cards */
    [data-testid="stMetric"] { background: transparent; padding: 0; border: none; text-align: left; }
    [data-testid="stMetricLabel"] { opacity: 0.65; font-size: 0.8rem; }
    [data-testid="stMetricValue"] { font-size: 1.3rem; font-weight: 600; }

    /* Bordered containers */
    [data-testid="stVerticalBlockBorderWrapper"] { border-radius: 10px; }

    /* Section headings */
    h3 { font-size: 1.25rem !important; font-weight: 600 !important; margin-bottom: 0.1rem !important; }
    h4 { font-size: 1.05rem !important; font-weight: 600 !important; }
    hr { border-top: 1px solid rgba(128,128,128,0.2) !important; margin: 1.5rem 0 !important; }
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

    if target_vehicle != "ANY" and target_vehicle.upper() != pmbjp_vehicle.upper():
        return True, f"⚠️ Vehicle Mismatch ({target_vehicle} vs {pmbjp_vehicle})"

    return True, "✅ Exact Clinical Match"

# ==========================================
# 5. REINFORCEMENT LEARNING ENGINE
# ==========================================
RL_FILE = "data/processed/rl_q_table.json"

def load_q_table():
    if os.path.exists(RL_FILE):
        with open(RL_FILE, "r") as f:
            return json.load(f)
    return {}

def update_q_table(drug_code, reward=1.0):
    q_table = load_q_table()
    if drug_code not in q_table:
        q_table[drug_code] = 0.0
    # Q-Learning Bellman equation simplification (alpha = 0.1)
    q_table[drug_code] = q_table[drug_code] + 0.1 * (reward - q_table[drug_code])
    
    with open(RL_FILE, "w") as f:
        json.dump(q_table, f)

# Handling the button click to train the RL Agent
if 'rl_feedback_given' not in st.session_state:
    st.session_state.rl_feedback_given = False

def handle_rl_click(drug_code, drug_name):
    update_q_table(drug_code, reward=1.0)
    st.session_state.rl_feedback_given = True
    st.toast(f"Saved your choice: {drug_name}", icon="✅")

def scan_pmbjp(entities_json, vehicle, comm_price=0.0, key_prefix=""):
    safe_matches = []
    
    # 1. Find all safe matches
    for _, generic in df_pmbjp.iterrows():
        is_safe, msg = check_safety_gates(entities_json, generic['parsed_entities'], vehicle, generic['vehicle'])
        if is_safe:
            gov_price = float(generic['MRP'])
            savings = comm_price - gov_price if comm_price > 0 else 0
            safe_matches.append({
                "code": str(generic['Drug Code']),
                "name": generic['Generic Name'],
                "price": gov_price,
                "savings": savings,
                "comm_price": comm_price,
                "msg": msg
            })
            
    if not safe_matches:
        return False
        
    # 2. RL Ranking (Epsilon-Greedy Strategy)
    q_table = load_q_table()
    
    # Epsilon = 0.1 means 10% of the time, the RL agent explores by randomizing the list
    # 90% of the time, it exploits its knowledge and sorts by the highest Q-value
    if np.random.rand() < 0.1:
        st.caption("Showing a varied order this time to learn your preferences.")
        np.random.shuffle(safe_matches)
    else:
        # Sort by Q-Value (default 0.0 if unseen), then by price ascending
        safe_matches = sorted(safe_matches, key=lambda x: (q_table.get(x['code'], 0.0), -x['price']), reverse=True)

    st.caption(f"{len(safe_matches)} matching generic{'s' if len(safe_matches) != 1 else ''} found")

    # 3. Display Results
    for match in safe_matches:
        q_score = q_table.get(match['code'], 0.0)
        with st.container(border=True):
            col1, col2, col3, col4 = st.columns([3, 1.2, 1.2, 1.3])
            with col1:
                st.markdown(f"**{match['name']}**")
                st.caption(f"Code `{match['code']}`  |  Preference score {q_score:.3f}")
                st.caption(match['msg'])
            with col2:
                st.metric("Gov price", f"₹{match['price']:.2f}")
            with col3:
                if match['comm_price'] > 0:
                    st.metric("You save", f"₹{match['savings']:.2f}", f"{(match['savings']/match['comm_price'])*100:.0f}%", delta_color="normal")
                else:
                    st.metric("You save", "—")
            with col4:
                # The feedback button that triggers the RL reward
                st.button(
                    "Select",
                    key=f"btn_{key_prefix}{match['code']}",
                    on_click=handle_rl_click,
                    args=(match['code'], match['name']),
                    help="Marks this formulation as your pick so similar results rank higher next time.",
                    use_container_width=True
                )
        
    return True

# ==========================================
# 6. DEEP LEARNING MODEL (SYMPTOM CHECKER)
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
    st.error("Data files not found. Add `1mg_dermatology_parsed.csv` and `pmbjp_dermatology_parsed.csv` to `data/processed/`, then reload.")
    st.stop()

dl_model, le_cond, le_skin, le_comp, le_veh, all_conditions, all_skins = load_and_train_dl_model()

# ==========================================
# 7. UI LAYOUT & TABS
# ==========================================
st.markdown(
    "<div class='nx-title'>Nidaan-Rx</div>"
    "<div class='nx-sub'>Clinical decision support and generic substitution</div>",
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4 = st.tabs(["Symptom checker", "Brand search", "Manual entry", "Scan label"])

# ----------------- TAB 1: SYMPTOM CHECKER (DEEP LEARNING) -----------------
with tab1:
    st.markdown("### Symptom checker")
    st.caption("A neural network suggests an active ingredient and form. Use it as a starting point, not a prescription.")

    col1, col2 = st.columns(2)
    with col1:
        cond_input = st.selectbox("Condition", sorted(all_conditions))
    with col2:
        skin_input = st.selectbox("Skin type", sorted(all_skins))
    sev_input = st.slider("Severity (1 = mild, 10 = severe)", 1, 10, 5)

    if st.button("Suggest treatment", type="primary", key="btn_ai"):
        st.session_state.t1 = {"cond": cond_input, "skin": skin_input, "sev": sev_input}

    q1 = st.session_state.get("t1")
    if q1:
        c_enc = le_cond.transform([q1["cond"]])[0]
        s_enc = le_skin.transform([q1["skin"]])[0]
        sev_norm = q1["sev"] / 10.0

        input_tensor = torch.tensor([[c_enc, s_enc, sev_norm]], dtype=torch.float32)

        dl_model.eval()
        with torch.no_grad():
            comp_out, veh_out = dl_model(input_tensor)
            pred_comp = le_comp.inverse_transform([torch.argmax(comp_out).item()])[0]
            pred_veh = le_veh.inverse_transform([torch.argmax(veh_out).item()])[0]

        st.markdown("---")
        st.caption(f"Results for: {q1['cond']}, {q1['skin'].lower()} skin, severity {q1['sev']}/10")

        with st.container(border=True):
            col_res1, col_res2 = st.columns(2)
            col_res1.metric("Active ingredient", pred_comp.title())
            col_res2.metric("Form", pred_veh.upper())

        with st.expander("How to apply", expanded=True):
            st.markdown(POSOLOGY_GUIDE.get(pred_comp, "Apply as directed by physician."))

        st.markdown("#### Jan Aushadhi matches")
        entities_json = json.dumps([{"salt": pred_comp.lower(), "strength": "N/A"}])

        found = scan_pmbjp(entities_json, pred_veh, key_prefix="t1_")
        if not found:
            st.warning(f"The model suggested {pred_comp.title()} ({pred_veh.upper()}), but there is no direct PMBJP generic for it in the government inventory.")

        st.caption("AI suggestion only. Consult a registered dermatologist before dispensing medication.")
    else:
        st.caption("Pick a condition, skin type and severity, then select Suggest treatment.")

# ----------------- TAB 2: BRAND SEARCH -----------------
with tab2:
    st.markdown("### Brand search")
    st.caption("Find Jan Aushadhi generics for a branded medicine.")

    drug_list = [""] + list(df_comm['display_name'].unique())
    selected_drug = st.selectbox(
        "Branded medicine",
        options=drug_list,
        format_func=lambda x: "Search by name, e.g. Candid" if x == "" else x
    )

    if selected_drug:
        target = df_comm[df_comm['display_name'] == selected_drug].iloc[0]
        try:
            comm_price = float(str(target['product_price']).replace('₹', '').replace(',', ''))
        except:
            comm_price = 0.0

        with st.container(border=True):
            m1, m2, m3 = st.columns([2, 1, 1])
            m1.metric("Medicine", target['product_name'])
            m2.metric("MRP", f"₹{comm_price:,.2f}")
            m3.metric("Form", target['vehicle'])

        if st.button("Find generic alternatives", type="primary", key="btn_brand"):
            st.session_state.t2 = selected_drug

        if st.session_state.get("t2") == selected_drug:
            st.markdown("---")
            st.markdown("#### Safe generic alternatives")
            if not scan_pmbjp(target['parsed_entities'], target['vehicle'], comm_price, key_prefix="t2_"):
                st.error("No safe Jan Aushadhi match exists for this exact formulation.")
    else:
        st.caption("Select a medicine to see its price and generic alternatives.")

# ----------------- TAB 3: MANUAL ENTRY -----------------
with tab3:
    st.markdown("### Manual entry")
    st.caption("Enter an active ingredient, strength and form to look up generics.")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        salt_options = [""] + available_salts
        salt_name = st.selectbox(
            "Active ingredient",
            options=salt_options,
            format_func=lambda x: "Select an ingredient" if x == "" else x
        )
    with col2:
        strength = st.text_input("Strength", value="N/A", help="For example 1% or N/A")
    with col3:
        vehicle = st.selectbox("Form", ["CREAM", "OINTMENT", "GEL", "LOTION", "POWDER", "SHAMPOO", "SOAP", "TABLET/CAPSULE", "ANY"])

    if st.button("Search generics", type="primary", key="btn_manual"):
        st.session_state.t3 = {"salt": salt_name, "strength": strength, "vehicle": vehicle}

    q3 = st.session_state.get("t3")
    if q3:
        if q3["salt"]:
            st.markdown("---")
            entities = json.dumps([{"salt": q3["salt"].lower().strip(), "strength": q3["strength"].strip()}])
            if not scan_pmbjp(entities, q3["vehicle"], key_prefix="t3_"):
                st.error("No safe matches found. Try a different strength or form.")
        else:
            st.warning("Select an active ingredient to search.")

# ----------------- TAB 4: OCR (AI VISION) -----------------
with tab4:
    st.markdown("### Scan label")
    st.caption("Upload a photo of a medicine box or printed prescription. Text is read from the image, typos are corrected, and generic equivalents are shown.")

    uploaded_file = st.file_uploader("Upload image (JPG or PNG)", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")

    if uploaded_file is None:
        st.caption("Upload a clear, well-lit photo with the medicine name visible.")
    else:
        image = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(image)
        
        col_img, col_res = st.columns([1, 1.4])
        with col_img:
            st.image(image, caption="Uploaded document", use_container_width=True)
            
        with col_res:
            # Run OCR once per uploaded file so selecting a result doesn't re-scan the image
            file_key = f"{uploaded_file.name}-{uploaded_file.size}"
            if st.session_state.get("ocr_key") != file_key:
                with st.spinner("Reading text from image..."):
                    try:
                        result = ocr_model.predict(img_array)
                    except:
                        result = ocr_model.ocr(img_array, cls=True)

                texts = []
                if result:
                    for res in result:
                        if hasattr(res, 'keys') or isinstance(res, dict):
                            if 'rec_texts' in res:
                                texts.extend(res['rec_texts'])
                            elif 'rec_text' in res:
                                texts.append(res['rec_text'])
                        elif hasattr(res, 'rec_texts'):
                            texts.extend(res.rec_texts)
                        else:
                            try:
                                for line in res:
                                    texts.append(line[1][0])
                            except:
                                pass
                st.session_state.ocr_key = file_key
                st.session_state.ocr_empty = not result
                st.session_state.ocr_texts = texts

            extracted_texts = st.session_state.ocr_texts

            if st.session_state.ocr_empty:
                st.error("No text detected. Try a sharper or better-lit photo.")
            else:
                with st.expander("Raw text from image", expanded=False):
                    st.write(extracted_texts)
                
                found_valid_drug = False
                
                medical_keywords = ['cream', 'gel', 'ointment', 'lotion', 'shampoo', 'tablet', 'capsule', 'acid', 'mg', '%', 'w/w']
                
                potential_meds = []
                for text in extracted_texts:
                    text_lower = text.lower()
                    if any(kw in text_lower for kw in medical_keywords) or (text.isupper() and len(text) > 4):
                        potential_meds.append(text)

                if not potential_meds:
                    potential_meds = extracted_texts

                identified_drugs = []
                for text in potential_meds:
                    if len(text) > 4: 
                        best_match, confidence = process.extractOne(text.title(), available_salts)
                        
                        if confidence >= 85 and best_match not in [d['name'] for d in identified_drugs]:
                            identified_drugs.append({"name": best_match, "confidence": confidence, "raw": text})
                            
                if identified_drugs:
                    found_valid_drug = True
                    for drug in identified_drugs:
                        st.markdown(f"#### {drug['name']}")
                        st.caption(f"Identified active ingredient, {drug['confidence']}% confidence")
                        
                        entities_json = json.dumps([{"salt": drug['name'].lower(), "strength": "N/A"}])
                        
                        scan_pmbjp(entities_json, vehicle="ANY", key_prefix=f"t4_{drug['name']}_")
                        st.markdown("---")
                
                if not found_valid_drug:
                    st.warning("No dermatology active ingredient recognized. Make sure the medicine name is clearly visible.")