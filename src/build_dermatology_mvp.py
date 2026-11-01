import pandas as pd
import re
import json

# ==========================================
# 1. CONFIGURATION
# ==========================================
TATA_1MG_FILE = "data/raw/medicine_data.csv" 
PMBJP_FILE = "data/raw/PMJAY_list.csv"      

TATA_COMPOSITION_COL = "salt_composition" 
TATA_NAME_COL = "product_name"               

PMBJP_COMPOSITION_COL = "Generic Name" 
PMBJP_CODE_COL = "Drug Code"           

# ==========================================
# 2. DICTIONARIES & FILTERS
# ==========================================
DERM_SALTS = [
    "clotrimazole", "ketoconazole", "itraconazole", "fluconazole", "miconazole",
    "terbinafine", "luliconazole", "sertaconazole", "amorolfine",
    "betamethasone", "mometasone", "clobetasol", "hydrocortisone", "fluticasone",
    "desonide", "beclomethasone", "halobetasol",
    "adapalene", "clindamycin", "tretinoin", "benzoyl peroxide", "salicylic acid",
    "azelaic acid", "isotretinoin",
    "permethrin", "fusidic acid", "mupirocin", "neomycin", "gentamicin",
    "nadifloxacin", "silver sulfadiazine", "povidone iodine", "calamine"
]

VEHICLE_MAP = {
    "CREAM": ["cream", "crm", "emulsion"],
    "OINTMENT": ["ointment", "oint"],
    "GEL": ["gel", "jelly"],
    "LOTION": ["lotion", "lot", "solution", "soln", "topical solution"],
    "POWDER": ["powder", "dusting powder"],
    "SOAP": ["soap", "bar"],
    "SHAMPOO": ["shampoo", "scalp"]
}

# Words that get stuck inside the PMBJP generic names that we MUST delete
VEHICLE_WORDS = [
    "cream", "ointment", "gel", "lotion", "powder", "soap", "shampoo", 
    "tablet", "tablets", "capsule", "injection", "suspension", "syrup", 
    "suppository", "vaginal", "eye drops", "ear drops", "solution", "topical"
]

# Normalizes complex chemical names to their pure base
SYNONYM_MAP = {
    "beclometasone": "beclomethasone",
    "beclomethasone dipropionate": "beclomethasone",
    "clobetasol propionate": "clobetasol",
    "mometasone furoate": "mometasone",
    "betamethasone valerate": "betamethasone",
    "betamethasone dipropionate": "betamethasone",
    "clindamycin phosphate": "clindamycin",
    "terbinafine hydrochloride": "terbinafine",
    "gentamicin sulphate": "gentamicin",
    "neomycin sulphate": "neomycin",
    "miconazole nitrate": "miconazole"
}

# ==========================================
# 3. PARSING LOGIC
# ==========================================
def extract_vehicle(text):
    text_lower = str(text).lower()
    for canonical_vehicle, keywords in VEHICLE_MAP.items():
        for kw in keywords:
            if re.search(rf'\b{kw}\b', text_lower):
                return canonical_vehicle
    return "UNKNOWN"

def parse_composition(comp_text):
    if pd.isna(comp_text):
        return json.dumps([])
    
    clean = str(comp_text).lower()
    clean = re.sub(r'\b(ip|bp|usp|eq\.?\s*to|equivalent\s*to|w/w|w/v|v/v)\b', '', clean)
    
    parts = re.split(r'\+|\band\b|,|&', clean)
    parsed_components = []
    
    for part in parts:
        strength_match = re.search(r'(\d+(?:\.\d+)?\s*(?:%|mg|mcg|g|iu|ml))', part)
        strength = strength_match.group(1).replace(" ", "") if strength_match else "N/A"
        
        salt_raw = re.sub(r'(\d+(?:\.\d+)?\s*(?:%|mg|mcg|g|iu|ml))', '', part)
        salt_raw = re.sub(r'[\(\)\[\]\-\/]', ' ', salt_raw)
        
        # FIX: Scrub all vehicle words out of the salt string!
        for word in VEHICLE_WORDS:
            salt_raw = re.sub(rf'\b{word}s?\b', '', salt_raw)
            
        salt_raw = re.sub(r'\s+', ' ', salt_raw).strip()
        
        # FIX: Force chemical esters down to their base salt
        for wrong, right in SYNONYM_MAP.items():
            if wrong in salt_raw:
                salt_raw = right
                
        if salt_raw:
            parsed_components.append({"salt": salt_raw.strip(), "strength": strength})
            
    parsed_components = sorted(parsed_components, key=lambda x: x["salt"])
    return json.dumps(parsed_components)

# ==========================================
# 4. EXECUTION
# ==========================================
def process_datasets():
    print("Loading raw datasets...")
    df_1mg = pd.read_csv(TATA_1MG_FILE)
    df_pmbjp = pd.read_csv(PMBJP_FILE)

    pattern = '|'.join(DERM_SALTS)
    df_1mg_derm = df_1mg[df_1mg[TATA_COMPOSITION_COL].str.lower().str.contains(pattern, na=False)].copy()
    df_pmbjp_derm = df_pmbjp[df_pmbjp[PMBJP_COMPOSITION_COL].str.lower().str.contains(pattern, na=False)].copy()

    df_1mg_derm['parsed_entities'] = df_1mg_derm[TATA_COMPOSITION_COL].apply(parse_composition)
    df_1mg_derm['vehicle'] = df_1mg_derm[TATA_NAME_COL].astype(str) + " " + df_1mg_derm[TATA_COMPOSITION_COL].astype(str)
    df_1mg_derm['vehicle'] = df_1mg_derm['vehicle'].apply(extract_vehicle)

    df_pmbjp_derm['parsed_entities'] = df_pmbjp_derm[PMBJP_COMPOSITION_COL].apply(parse_composition)
    df_pmbjp_derm['vehicle'] = df_pmbjp_derm[PMBJP_COMPOSITION_COL].apply(extract_vehicle)

    out_1mg = "data/processed/1mg_dermatology_parsed.csv"
    out_pmbjp = "data/processed/pmbjp_dermatology_parsed.csv"
    
    df_1mg_derm.to_csv(out_1mg, index=False)
    df_pmbjp_derm.to_csv(out_pmbjp, index=False)
    print("✅ SUCCESS! Clean files saved.")

if __name__ == "__main__":
    process_datasets()
