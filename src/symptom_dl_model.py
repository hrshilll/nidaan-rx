import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. SYNTHETIC DATASET GENERATOR
# ==========================================
def generate_medical_dataset(filename="data/raw/symptom_dataset.csv", num_samples=1000):
    """Generates training data mapping conditions to chemical components and vehicles."""
    np.random.seed(42)
    
    conditions = ["Acne", "Fungal Infection", "Eczema", "Psoriasis", "Dandruff"]
    skin_types = ["Oily", "Dry", "Combination", "Normal"]
    
    data = []
    for _ in range(num_samples):
        cond = np.random.choice(conditions)
        skin = np.random.choice(skin_types)
        severity = np.random.randint(1, 11) # 1 to 10
        
        # Clinical Rules Engine (for generating realistic training data)
        if cond == "Acne":
            component = "adapalene" if skin == "Oily" else "clindamycin"
            vehicle = "GEL" if skin == "Oily" else "CREAM"
        elif cond == "Fungal Infection":
            component = "clotrimazole" if severity < 7 else "ketoconazole"
            vehicle = "POWDER" if skin == "Oily" else "CREAM"
        elif cond == "Eczema":
            component = "hydrocortisone" if severity < 5 else "clobetasol"
            vehicle = "OINTMENT" if skin == "Dry" else "LOTION"
        elif cond == "Psoriasis":
            component = "salicylic acid" if severity < 6 else "mometasone"
            vehicle = "OINTMENT"
        elif cond == "Dandruff":
            component = "ketoconazole"
            vehicle = "SHAMPOO"
            
        data.append([cond, skin, severity, component, vehicle])
        
    df = pd.DataFrame(data, columns=["Condition", "Skin_Type", "Severity", "Component", "Vehicle"])
    df.to_csv(filename, index=False)
    print(f"✅ Generated {num_samples} rows of training data at {filename}")
    return df

# ==========================================
# 2. DATA PREPROCESSING
# ==========================================
# We use LabelEncoders to convert text (Acne, Oily) into numbers (0, 1) for the Neural Network.
le_cond = LabelEncoder()
le_skin = LabelEncoder()
le_comp = LabelEncoder()
le_veh = LabelEncoder()

def preprocess_data(df):
    X = pd.DataFrame()
    X['Condition'] = le_cond.fit_transform(df['Condition'])
    X['Skin_Type'] = le_skin.fit_transform(df['Skin_Type'])
    X['Severity'] = df['Severity'] / 10.0  # Normalize severity between 0 and 1
    
    y_comp = le_comp.fit_transform(df['Component'])
    y_veh = le_veh.fit_transform(df['Vehicle'])
    
    return torch.tensor(X.values, dtype=torch.float32), \
           torch.tensor(y_comp, dtype=torch.long), \
           torch.tensor(y_veh, dtype=torch.long)

# ==========================================
# 3. PYTORCH DEEP LEARNING ARCHITECTURE
# ==========================================
class PrismMultiTaskDNN(nn.Module):
    def __init__(self, input_size, num_components, num_vehicles):
        super(PrismMultiTaskDNN, self).__init__()
        # Shared Layers (Learns the general relationship of skin conditions)
        self.shared_layer = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        # Multi-Task Heads: Network splits to predict two different things simultaneously
        self.component_head = nn.Linear(16, num_components)
        self.vehicle_head = nn.Linear(16, num_vehicles)

    def forward(self, x):
        shared_out = self.shared_layer(x)
        comp_pred = self.component_head(shared_out)
        veh_pred = self.vehicle_head(shared_out)
        return comp_pred, veh_pred

# ==========================================
# 4. TRAINING LOOP
# ==========================================
def train_model(model, X_train, y_comp_train, y_veh_train, epochs=150):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print("\n🧠 Training PRISM-Rx Deep Learning Model...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        comp_out, veh_out = model(X_train)
        
        # Calculate loss for both tasks and combine them
        loss_comp = criterion(comp_out, y_comp_train)
        loss_veh = criterion(veh_out, y_veh_train)
        total_loss = loss_comp + loss_veh
        
        # Backward pass (Backpropagation)
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"   Epoch [{epoch+1}/{epochs}] | Loss: {total_loss.item():.4f}")
            
    print("✅ Model Training Complete.")
    return model

# ==========================================
# 5. INFERENCE (PREDICTION)
# ==========================================
def predict_treatment(model, condition, skin_type, severity):
    model.eval()
    with torch.no_grad():
        # Encode user input
        try:
            c_enc = le_cond.transform([condition])[0]
            s_enc = le_skin.transform([skin_type])[0]
            sev_norm = severity / 10.0
        except ValueError:
            return "Error: Unknown Condition or Skin Type entered."
            
        input_tensor = torch.tensor([[c_enc, s_enc, sev_norm]], dtype=torch.float32)
        
        # Predict
        comp_out, veh_out = model(input_tensor)
        
        comp_idx = torch.argmax(comp_out).item()
        veh_idx = torch.argmax(veh_out).item()
        
        predicted_component = le_comp.inverse_transform([comp_idx])[0]
        predicted_vehicle = le_veh.inverse_transform([veh_idx])[0]
        
        return predicted_component, predicted_vehicle

# ==========================================
# 6. EXECUTION
# ==========================================
if __name__ == "__main__":
    import os
    os.makedirs("data/raw", exist_ok=True)
    
    # 1. Generate and load data
    df = generate_medical_dataset()
    X, y_comp, y_veh = preprocess_data(df)
    
    # 2. Initialize Model
    num_comps = len(le_comp.classes_)
    num_vehs = len(le_veh.classes_)
    model = PrismMultiTaskDNN(input_size=3, num_components=num_comps, num_vehicles=num_vehs)
    
    # 3. Train Model
    trained_model = train_model(model, X, y_comp, y_veh)
    
    # 4. Interactive Testing
    print("\n" + "="*50)
    print("🩺 PRISM-Rx SYMPTOM CHECKER (AI INFERENCE)")
    print("="*50)
    print("Available Conditions: Acne, Fungal Infection, Eczema, Psoriasis, Dandruff")
    print("Available Skin Types: Oily, Dry, Combination, Normal")
    
    while True:
        print("\n---")
        cond = input("Enter Condition (or 'exit'): ").strip()
        if cond.lower() == 'exit': break
        skin = input("Enter Skin Type: ").strip()
        sev = float(input("Enter Severity (1-10): ").strip())
        
        result = predict_treatment(trained_model, cond, skin, sev)
        if isinstance(result, str):
            print(result)
        else:
            comp, veh = result
            print(f"\n🤖 AI Prediction:")
            print(f"   Recommended Active Ingredient: [{comp.upper()}]")
            print(f"   Recommended Form Factor:       [{veh.upper()}]")
            print(f"   👉 Next Step: Pass '{comp}' and '{veh}' to the PMBJP Database Search!")
