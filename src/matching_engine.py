import pandas as pd
import json

# ==========================================
# 1. LOAD THE PROCESSED DATA
# ==========================================
COMMERCIAL_FILE = "data/processed/1mg_dermatology_parsed.csv"
PMBJP_FILE = "data/processed/pmbjp_dermatology_parsed.csv"

print("Loading PRISM-Rx Knowledge Base...")
try:
    df_comm = pd.read_csv(COMMERCIAL_FILE)
    df_pmbjp = pd.read_csv(PMBJP_FILE)
except FileNotFoundError:
    print("❌ Error: Could not find the processed CSV files. Run build_dermatology_mvp.py first.")
    exit()

# ==========================================
# 2. THE CLINICAL SAFETY GATES
# ==========================================
def check_safety_gates(target_entities_json, pmbjp_entities_json, target_vehicle, pmbjp_vehicle):
    try:
        t_salts = json.loads(target_entities_json)
        p_salts = json.loads(pmbjp_entities_json)
    except:
        return False, "Data Parse Error"

    if not t_salts or not p_salts:
        return False, "No Active Ingredients Found"

    # GATE 1: SUPERSET / SUBSET CHECK
    t_salt_names = {item['salt'] for item in t_salts}
    p_salt_names = {item['salt'] for item in p_salts}
    
    if t_salt_names != p_salt_names:
        return False, "Failed: Salt Composition Mismatch (Subset/Superset Error)"

    # GATE 2: DOSAGE CHECK
    for t, p in zip(t_salts, p_salts):
        if t['strength'] != p['strength'] and t['strength'] != "N/A" and p['strength'] != "N/A":
            return False, f"Failed: Dosage Mismatch ({t['strength']} vs {p['strength']})"

    # GATE 3: VEHICLE CHECK
    if target_vehicle != pmbjp_vehicle:
        return True, f"⚠️ Warning: Vehicle Mismatch ({target_vehicle} vs {pmbjp_vehicle})"

    return True, "✅ Exact Clinical Match"

# ==========================================
# 3. CORE SEARCH FUNCTION
# ==========================================
def scan_pmbjp_database(target_entities, target_vehicle, comm_price_float=0.0):
    found_matches = []
    
    for index, generic in df_pmbjp.iterrows():
        is_safe, message = check_safety_gates(
            target_entities, 
            generic['parsed_entities'], 
            target_vehicle, 
            generic['vehicle']
        )
        
        if is_safe:
            found_matches.append({
                "code": generic['Drug Code'],
                "name": generic['Generic Name'],
                "price": generic['MRP'],
                "status": message
            })

    if not found_matches:
        print("\n❌ No safe Jan Aushadhi generic matches found for this specific formulation.")
    else:
        print(f"\n✅ Found {len(found_matches)} Safe Jan Aushadhi Match(es):\n")
        for m in found_matches:
            savings = comm_price_float - float(m['price']) if comm_price_float > 0 else 0
            savings_pct = (savings / comm_price_float * 100) if comm_price_float > 0 else 0
            
            print(f"   💊 [Code: {m['code']}] {m['name']}")
            print(f"   🛡️  {m['status']}")
            if comm_price_float > 0:
                print(f"   💰 Gov Price: ₹{m['price']} (Savings: ₹{savings:.2f} / {savings_pct:.0f}%)\n")
            else:
                print(f"   💰 Gov Price: ₹{m['price']}\n")

# ==========================================
# 4. USER EXPERIENCE MODES
# ==========================================
def mode_brand_search():
    query = input("\n🔍 Enter branded medicine name (e.g., 'Candid'): ")
    matches = df_comm[df_comm['product_name'].str.lower().str.contains(query.lower(), na=False)]
    
    if matches.empty:
        print(f"❌ Could not find '{query}' in commercial database.")
        return

    # THE FIX: Disambiguation Menu
    print(f"\nFound {len(matches)} variations. Please select the exact match:")
    for i, row in enumerate(matches.head(10).itertuples(), 1):
        print(f"  {i}. {row.product_name} (Vehicle: {row.vehicle})")
    
    if len(matches) > 10:
        print("  ... (showing top 10 results)")

    try:
        selection = int(input("\nSelect number (0 to cancel): "))
        if selection == 0 or selection > len(matches):
            return
        target = matches.iloc[selection - 1]
    except ValueError:
        print("Invalid selection.")
        return

    # Extract price safely
    try:
        comm_price_float = float(str(target['product_price']).replace('₹', '').replace(',', ''))
    except:
        comm_price_float = 0.0

    print(f"\n" + "="*50)
    print(f"🏥 TARGET: {target['product_name']} | MRP: ₹{comm_price_float}")
    print(f"🧪 APIs: {target['parsed_entities']} | {target['vehicle']}")
    print("="*50)

    scan_pmbjp_database(target['parsed_entities'], target['vehicle'], comm_price_float)


def mode_manual_entry():
    print("\n" + "="*50)
    print("🧪 MANUAL COMPOSITION ENTRY")
    print("="*50)
    
    salts = []
    while True:
        salt_name = input("Enter Active Ingredient / Salt (e.g., 'clotrimazole'): ").strip().lower()
        strength = input("Enter Strength (e.g., '1%', '50mg', or 'N/A'): ").strip().lower()
        
        salts.append({"salt": salt_name, "strength": strength})
        
        more = input("Add another salt to this formulation? (y/n): ").strip().lower()
        if more != 'y':
            break

    # Sort salts alphabetically (Crucial for the subset/superset gate to work!)
    salts = sorted(salts, key=lambda x: x["salt"])
    entities_json = json.dumps(salts)

    print("\nSelect Vehicle / Form Factor:")
    print(" 1. CREAM\n 2. OINTMENT\n 3. GEL\n 4. LOTION\n 5. POWDER\n 6. SHAMPOO")
    
    vehicle_choices = {"1": "CREAM", "2": "OINTMENT", "3": "GEL", "4": "LOTION", "5": "POWDER", "6": "SHAMPOO"}
    v_choice = input("Select number: ").strip()
    vehicle = vehicle_choices.get(v_choice, "UNKNOWN")

    print(f"\n🔍 Searching PMBJP for: {entities_json} in {vehicle} base...")
    scan_pmbjp_database(entities_json, vehicle, comm_price_float=0.0)


# ==========================================
# MAIN MENU LOOP
# ==========================================
if __name__ == "__main__":
    while True:
        print("\n" + "#"*50)
        print("⚡ PRISM-Rx: Clinical Decision Engine ⚡")
        print("#"*50)
        print(" 1. Search by Commercial Brand Name")
        print(" 2. Manual Custom Entry (Salt, Strength, Vehicle)")
        print(" 3. Exit")
        
        choice = input("\nSelect Mode (1-3): ").strip()
        
        if choice == '1':
            mode_brand_search()
        elif choice == '2':
            mode_manual_entry()
        elif choice == '3':
            print("Exiting PRISM-Rx. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
