# ═══════════════════════════════════════════════════════════════════════
#  FERTILIZER RECOMMENDATION MODEL — INDIA
#  Crops    : Rice, Wheat, Maize, Sugarcane
#  Model    : Random Forest Regressor (NPK doses) +
#             Multi-output Classifier (Lime/Gypsum/Zinc flags)
#  Accuracy : N MAE=3.5 kg/ha R²=0.979 | P MAE=1.8 | K MAE=0.9
#  Source   : ICAR-CRRI, ICAR-IISS, ICAR-IIMR, ICAR-IISR guidelines
#  Author   : Satyendra (DSAstra Project)
# ═══════════════════════════════════════════════════════════════════════


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 1 — Imports                                                   │
# └─────────────────────────────────────────────────────────────────────┘
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import (mean_absolute_error, r2_score,
                              mean_squared_error, accuracy_score)

print("All imports successful")


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 2 — Load Dataset                                              │
# └─────────────────────────────────────────────────────────────────────┘
# Upload fertilizer_dataset.csv to Colab first

CSV_PATH = 'fertilizer_dataset.csv'
df = pd.read_csv(CSV_PATH)

print(f"Dataset shape  : {df.shape}")
print(f"Crops          : {df['Crop'].unique().tolist()}")
print(f"Varieties      : {df['Variety'].unique().tolist()}")
print(f"Organic inputs : {df['Organic_Input'].unique().tolist()}")
print(f"Previous crops : {df['Prev_Crop'].unique().tolist()}")
print(f"\nTarget ranges:")
for col in ['N_recommended','P_recommended','K_recommended']:
    print(f"  {col}: {df[col].min():.0f} – {df[col].max():.0f} kg/ha | mean={df[col].mean():.1f}")
print(f"\nSpecial flags:")
print(f"  Needs Lime:   {df['Needs_Lime'].mean()*100:.1f}% of fields")
print(f"  Needs Gypsum: {df['Needs_Gypsum'].mean()*100:.1f}% of fields")
print(f"  Needs Zinc:   {df['Needs_Zinc'].mean()*100:.1f}% of fields")
print(f"\nFirst 3 rows:")
print(df.head(3).to_string(index=False))


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 3 — Feature & Target Definition                               │
# └─────────────────────────────────────────────────────────────────────┘
# ── Encode categorical features ──────────────────────────────────────
le_crop    = LabelEncoder()
le_variety = LabelEncoder()
le_organic = LabelEncoder()
le_prev    = LabelEncoder()

df['Crop_enc']     = le_crop.fit_transform(df['Crop'])
df['Variety_enc']  = le_variety.fit_transform(df['Variety'])
df['Organic_enc']  = le_organic.fit_transform(df['Organic_Input'])
df['PrevCrop_enc'] = le_prev.fit_transform(df['Prev_Crop'])

FEATURES = [
    'Crop_enc',          # which crop (0=Maize, 1=Rice, 2=Sugarcane, 3=Wheat)
    'Zone_ID',           # ICAR zone 1-15
    'Season',            # 0=Kharif, 1=Rabi, 2=Zaid
    'Nitrogen_soil',     # current N in soil kg/ha — from sensor
    'Phosphorus_soil',   # current P in soil kg/ha — from sensor
    'Potassium_soil',    # current K in soil kg/ha — from sensor
    'pH',                # soil pH — from sensor
    'Irrigation',        # 0=rainfed, 1=irrigated
    'Variety_enc',       # HYV=0, Traditional=1
    'Organic_enc',       # organic input type
    'PrevCrop_enc',      # previous crop
    'Farm_Size_ha',      # farm size in hectares
]

REG_TARGETS  = ['N_recommended', 'P_recommended', 'K_recommended']
FLAG_TARGETS = ['Needs_Lime', 'Needs_Gypsum', 'Needs_Zinc']

print(f"Features ({len(FEATURES)}): {FEATURES}")
print(f"Regression targets: {REG_TARGETS}")
print(f"Flag targets: {FLAG_TARGETS}")

print(f"\nLabel encoding:")
print(f"  Crop:    {dict(zip(le_crop.classes_, le_crop.transform(le_crop.classes_)))}")
print(f"  Variety: {dict(zip(le_variety.classes_, le_variety.transform(le_variety.classes_)))}")
print(f"  Organic: {dict(zip(le_organic.classes_, le_organic.transform(le_organic.classes_)))}")
print(f"  PrevCrop:{dict(zip(le_prev.classes_, le_prev.transform(le_prev.classes_)))}")


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 4 — Scale & Split                                             │
# └─────────────────────────────────────────────────────────────────────┘
scaler   = MinMaxScaler()
X        = df[FEATURES]
X_scaled = scaler.fit_transform(X)

y_reg  = df[REG_TARGETS]
y_flag = df[FLAG_TARGETS]

(X_train, X_test,
 y_reg_train, y_reg_test,
 y_flag_train, y_flag_test) = train_test_split(
    X_scaled, y_reg, y_flag,
    test_size=0.2, random_state=42
)

print(f"Train: {len(X_train)} | Test: {len(X_test)}")
print(f"Features scaled to [0,1]")


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 5 — Train NPK Regressor                                       │
# └─────────────────────────────────────────────────────────────────────┘
print("Training NPK regressor...")

npk_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    max_features='sqrt',
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
npk_model.fit(X_train, y_reg_train)
print("NPK regressor trained")


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 6 — Train Special Flag Classifier                             │
# └─────────────────────────────────────────────────────────────────────┘
print("Training flag classifier (Lime/Gypsum/Zinc)...")

base_clf  = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
flag_model = MultiOutputClassifier(base_clf)
flag_model.fit(X_train, y_flag_train)
print("Flag classifier trained")


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 7 — Evaluation                                                │
# └─────────────────────────────────────────────────────────────────────┘
y_reg_pred  = npk_model.predict(X_test)
y_flag_pred = flag_model.predict(X_test)

print("=" * 50)
print("  NPK REGRESSOR RESULTS")
print("=" * 50)
for i, target in enumerate(REG_TARGETS):
    mae = mean_absolute_error(y_reg_test[target], y_reg_pred[:, i])
    r2  = r2_score(y_reg_test[target], y_reg_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_reg_test[target], y_reg_pred[:, i]))
    print(f"  {target:20s}: MAE={mae:.2f} kg/ha | R²={r2:.4f} | RMSE={rmse:.2f}")

print(f"\n{'='*50}")
print(f"  FLAG CLASSIFIER RESULTS")
print(f"{'='*50}")
for i, flag in enumerate(FLAG_TARGETS):
    acc = accuracy_score(y_flag_test[flag], y_flag_pred[:, i])
    print(f"  {flag:20s}: Accuracy={acc*100:.1f}%")

# Train vs test gap check
y_reg_train_pred = npk_model.predict(X_train)
train_mae_n = mean_absolute_error(y_reg_train['N_recommended'], y_reg_train_pred[:,0])
test_mae_n  = mean_absolute_error(y_reg_test['N_recommended'],  y_reg_pred[:,0])
print(f"\nOverfitting check (N dose): Train MAE={train_mae_n:.2f} | Test MAE={test_mae_n:.2f}")


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 8 — Visualisations                                            │
# └─────────────────────────────────────────────────────────────────────┘
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Fertilizer Recommendation Model — Evaluation', fontsize=14, fontweight='bold')

# Actual vs Predicted for N, P, K
colors = ['steelblue', 'green', 'orange']
for i, (target, color) in enumerate(zip(REG_TARGETS, colors)):
    ax = axes[0][i]
    ax.scatter(y_reg_test[target], y_reg_pred[:, i], alpha=0.3, s=10, color=color)
    max_val = max(y_reg_test[target].max(), y_reg_pred[:, i].max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=1)
    mae = mean_absolute_error(y_reg_test[target], y_reg_pred[:, i])
    r2  = r2_score(y_reg_test[target], y_reg_pred[:, i])
    ax.set_title(f'{target}\nMAE={mae:.1f} kg/ha | R²={r2:.3f}')
    ax.set_xlabel('Actual (kg/ha)')
    ax.set_ylabel('Predicted (kg/ha)')

# Feature importance
imp = pd.Series(npk_model.estimators_[0].feature_importances_
                if hasattr(npk_model, 'estimators_')
                else npk_model.feature_importances_,
                index=FEATURES).sort_values(ascending=False)

# Average importance across N/P/K sub-estimators
avg_imp = np.mean([npk_model.estimators_[i].feature_importances_
                   for i in range(3)], axis=0)
imp = pd.Series(avg_imp, index=FEATURES).sort_values()
imp.plot(kind='barh', ax=axes[1][0], color='steelblue', alpha=0.8)
axes[1][0].set_title('Feature Importance (avg N+P+K)')

# Residuals N
residuals = y_reg_test['N_recommended'].values - y_reg_pred[:, 0]
axes[1][1].hist(residuals, bins=40, color='steelblue', alpha=0.7)
axes[1][1].axvline(0, color='red', linestyle='--')
axes[1][1].set_title(f'N Residuals (std={residuals.std():.1f} kg/ha)')
axes[1][1].set_xlabel('Error (kg/ha)')

# Crop-wise N dose distribution
for crop in df['Crop'].unique():
    subset = df[df['Crop'] == crop]['N_recommended']
    axes[1][2].hist(subset, alpha=0.6, bins=30, label=crop)
axes[1][2].set_title('N Dose Distribution by Crop')
axes[1][2].legend()
axes[1][2].set_xlabel('N Recommended (kg/ha)')

plt.tight_layout()
plt.savefig('fertilizer_model_evaluation.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved fertilizer_model_evaluation.png")


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 9 — Sample Prediction Test                                    │
# └─────────────────────────────────────────────────────────────────────┘
print("=" * 50)
print("  SAMPLE PREDICTION TEST")
print("=" * 50)

def predict_fertilizer(crop, zone_id, season, n_soil, p_soil, k_soil,
                        ph, irrigation, variety, organic, prev_crop, farm_size):
    sample = {
        'Crop_enc':     le_crop.transform([crop])[0],
        'Zone_ID':      zone_id,
        'Season':       season,
        'Nitrogen_soil': n_soil,
        'Phosphorus_soil': p_soil,
        'Potassium_soil':  k_soil,
        'pH':           ph,
        'Irrigation':   irrigation,
        'Variety_enc':  le_variety.transform([variety])[0],
        'Organic_enc':  le_organic.transform([organic])[0],
        'PrevCrop_enc': le_prev.transform([prev_crop])[0],
        'Farm_Size_ha': farm_size,
    }
    X_in     = scaler.transform(pd.DataFrame([sample])[FEATURES])
    npk      = npk_model.predict(X_in)[0]
    flags    = flag_model.predict(X_in)[0]
    return {
        'N_per_ha': round(npk[0]),
        'P_per_ha': round(npk[1]),
        'K_per_ha': round(npk[2]),
        'N_total':  round(npk[0] * farm_size),
        'P_total':  round(npk[1] * farm_size),
        'K_total':  round(npk[2] * farm_size),
        'Needs_Lime':   bool(flags[0]),
        'Needs_Gypsum': bool(flags[1]),
        'Needs_Zinc':   bool(flags[2]),
    }

# Test 1 — Punjab Wheat farmer
r1 = predict_fertilizer(
    crop='Wheat', zone_id=6, season=1,
    n_soil=195, p_soil=28, k_soil=185, ph=7.8,
    irrigation=1, variety='HYV', organic='FYM_10t',
    prev_crop='Cereal', farm_size=2.0
)
print(f"\nTest 1: Punjab Wheat, 2ha, HYV, Irrigated, FYM available")
print(f"  N: {r1['N_per_ha']} kg/ha → {r1['N_total']} kg total")
print(f"  P: {r1['P_per_ha']} kg/ha → {r1['P_total']} kg total")
print(f"  K: {r1['K_per_ha']} kg/ha → {r1['K_total']} kg total")
print(f"  Lime needed: {r1['Needs_Lime']} | Gypsum: {r1['Needs_Gypsum']} | Zinc: {r1['Needs_Zinc']}")

# Test 2 — Kerala Rice farmer
r2 = predict_fertilizer(
    crop='Rice', zone_id=12, season=0,
    n_soil=175, p_soil=10, k_soil=110, ph=5.8,
    irrigation=0, variety='Traditional', organic='None',
    prev_crop='Fallow', farm_size=0.5
)
print(f"\nTest 2: Kerala Rice, 0.5ha, Traditional, Rainfed, No organic")
print(f"  N: {r2['N_per_ha']} kg/ha → {r2['N_total']} kg total")
print(f"  P: {r2['P_per_ha']} kg/ha → {r2['P_total']} kg total")
print(f"  K: {r2['K_per_ha']} kg/ha → {r2['K_total']} kg total")
print(f"  Lime needed: {r2['Needs_Lime']} | Gypsum: {r2['Needs_Gypsum']} | Zinc: {r2['Needs_Zinc']}")


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 10 — Save All Model Files                                     │
# └─────────────────────────────────────────────────────────────────────┘
os.makedirs('fertilizer_model_files', exist_ok=True)

with open('fertilizer_model_files/fertilizer_npk_model.pkl', 'wb') as f:
    pickle.dump(npk_model, f)
with open('fertilizer_model_files/fertilizer_flag_model.pkl', 'wb') as f:
    pickle.dump(flag_model, f)
with open('fertilizer_model_files/fertilizer_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

encoders = {
    'crop':     le_crop,
    'variety':  le_variety,
    'organic':  le_organic,
    'prev_crop': le_prev,
}
with open('fertilizer_model_files/fertilizer_encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

backend_config = {
    'features': FEATURES,
    'reg_targets': REG_TARGETS,
    'flag_targets': FLAG_TARGETS,
    'encodings': {
        'Crop':          list(le_crop.classes_),
        'Variety':       list(le_variety.classes_),
        'Organic_Input': list(le_organic.classes_),
        'Prev_Crop':     list(le_prev.classes_),
    },
    'model_metrics': {
        'N_MAE_kg_per_ha': round(mean_absolute_error(
            y_reg_test['N_recommended'], y_reg_pred[:,0]), 2),
        'N_R2': round(r2_score(y_reg_test['N_recommended'], y_reg_pred[:,0]), 4),
        'P_MAE_kg_per_ha': round(mean_absolute_error(
            y_reg_test['P_recommended'], y_reg_pred[:,1]), 2),
        'K_MAE_kg_per_ha': round(mean_absolute_error(
            y_reg_test['K_recommended'], y_reg_pred[:,2]), 2),
    },
    'scaler_params': {
        col: {'min': float(scaler.data_min_[i]),
              'max': float(scaler.data_max_[i])}
        for i, col in enumerate(FEATURES)
    },
    'icar_base_doses_kg_per_ha': {
        'Rice_HYV_irrigated':      {'N':100,'P':50,'K':50},
        'Rice_HYV_rainfed':        {'N':60, 'P':30,'K':30},
        'Wheat_HYV_irrigated':     {'N':120,'P':60,'K':40},
        'Wheat_HYV_rainfed':       {'N':80, 'P':40,'K':20},
        'Maize_HYV_irrigated':     {'N':150,'P':75,'K':50},
        'Maize_HYV_rainfed':       {'N':100,'P':50,'K':30},
        'Sugarcane_HYV_irrigated': {'N':250,'P':100,'K':120},
        'Sugarcane_HYV_rainfed':   {'N':200,'P':80, 'K':100},
    },
    'stage_schedule': {
        'Rice': [
            {'stage':'Basal','day_from_sowing':0,
             'N_fraction':0.50,'P_fraction':1.0,'K_fraction':0.50,
             'chemical_fertilizer':'DAP 108kg/ha + MOP 42kg/ha + Urea 55kg/ha',
             'natural_alternative':'FYM 10t/ha + Rock phosphate 200kg/ha'},
            {'stage':'Top dress 1','day_from_sowing':'25-30',
             'N_fraction':0.25,'P_fraction':0,'K_fraction':0.50,
             'chemical_fertilizer':'Urea 54kg/ha + MOP 42kg/ha',
             'natural_alternative':'Azolla 250kg/ha in standing water'},
            {'stage':'Top dress 2','day_from_sowing':'50-55',
             'N_fraction':0.25,'P_fraction':0,'K_fraction':0,
             'chemical_fertilizer':'Urea 54kg/ha',
             'natural_alternative':'Jeevamrut foliar spray 500L/ha'},
        ],
        'Wheat': [
            {'stage':'Basal','day_from_sowing':0,
             'N_fraction':0.50,'P_fraction':1.0,'K_fraction':1.0,
             'chemical_fertilizer':'DAP 130kg/ha + MOP 67kg/ha + Urea 52kg/ha',
             'natural_alternative':'FYM 15t/ha + Neem cake 500kg/ha'},
            {'stage':'Top dress 1','day_from_sowing':'21-25',
             'N_fraction':0.33,'P_fraction':0,'K_fraction':0,
             'chemical_fertilizer':'Urea 87kg/ha',
             'natural_alternative':'Vermicompost 5t/ha before irrigation'},
            {'stage':'Top dress 2','day_from_sowing':'40-45',
             'N_fraction':0.17,'P_fraction':0,'K_fraction':0,
             'chemical_fertilizer':'Urea 43kg/ha',
             'natural_alternative':'Azospirillum liquid 2L/ha'},
        ],
        'Maize': [
            {'stage':'Basal','day_from_sowing':0,
             'N_fraction':0.33,'P_fraction':1.0,'K_fraction':1.0,
             'chemical_fertilizer':'DAP 163kg/ha + MOP 83kg/ha + Urea 43kg/ha + ZnSO4 25kg/ha',
             'natural_alternative':'FYM 6t/ha + Rock phosphate 300kg/ha + Azospirillum seed treatment'},
            {'stage':'Top dress 1','day_from_sowing':'25-30',
             'N_fraction':0.33,'P_fraction':0,'K_fraction':0,
             'chemical_fertilizer':'Urea 108kg/ha',
             'natural_alternative':'Azotobacter 2L/ha soil drench'},
            {'stage':'Top dress 2','day_from_sowing':'45-50',
             'N_fraction':0.34,'P_fraction':0,'K_fraction':0,
             'chemical_fertilizer':'Urea 108kg/ha',
             'natural_alternative':'Vermicompost liquid 500L/ha foliar'},
        ],
        'Sugarcane': [
            {'stage':'Basal','day_from_sowing':0,
             'N_fraction':0.30,'P_fraction':1.0,'K_fraction':0.50,
             'chemical_fertilizer':'DAP 217kg/ha + MOP 100kg/ha + Urea 97kg/ha',
             'natural_alternative':'FYM 25t/ha + Pressmud cake 10t/ha'},
            {'stage':'Top dress 1','day_from_sowing':'30-45',
             'N_fraction':0.25,'P_fraction':0,'K_fraction':0.25,
             'chemical_fertilizer':'Urea 135kg/ha + MOP 50kg/ha',
             'natural_alternative':'Azotobacter+PSB 5kg/ha soil application'},
            {'stage':'Top dress 2','day_from_sowing':'90-120',
             'N_fraction':0.30,'P_fraction':0,'K_fraction':0.25,
             'chemical_fertilizer':'Urea 163kg/ha + MOP 50kg/ha',
             'natural_alternative':'Vermicompost 5t/ha + Jeevamrut 500L/ha'},
            {'stage':'Top dress 3','day_from_sowing':'150-180',
             'N_fraction':0.15,'P_fraction':0,'K_fraction':0,
             'chemical_fertilizer':'Urea 82kg/ha',
             'natural_alternative':'Panchagavya foliar 3% solution'},
        ],
    },
    'natural_alternatives': {
        'FYM_10t':      {'N_credit_kg_ha':50,'P_credit_kg_ha':25,'K_credit_kg_ha':60},
        'Vermicompost': {'N_credit_kg_ha':40,'P_credit_kg_ha':20,'K_credit_kg_ha':45},
        'GreenManure':  {'N_credit_kg_ha':60,'P_credit_kg_ha':10,'K_credit_kg_ha':20},
        'Azolla':       {'N_credit_kg_ha':30,'note':'Rice only — flooded field'},
        'Azospirillum': {'N_credit_kg_ha':25,'application':'seed treatment or 2L/ha'},
        'PSB':          {'P_credit_kg_ha':15,'note':'Mobilizes existing soil P'},
    }
}
with open('fertilizer_model_files/fertilizer_backend_config.json', 'w') as f:
    json.dump(backend_config, f, indent=2)

print("All fertilizer model files saved to fertilizer_model_files/")
print("  fertilizer_npk_model.pkl")
print("  fertilizer_flag_model.pkl")
print("  fertilizer_scaler.pkl")
print("  fertilizer_encoders.pkl")
print("  fertilizer_backend_config.json")


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 11 — Download All Files                                       │
# └─────────────────────────────────────────────────────────────────────┘
try:
    from google.colab import files
    import zipfile

    with zipfile.ZipFile('fertilizer_model_package.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        for fname in os.listdir('fertilizer_model_files'):
            zipf.write(f'fertilizer_model_files/{fname}', fname)
        if os.path.exists('fertilizer_model_evaluation.png'):
            zipf.write('fertilizer_model_evaluation.png')
        zipf.write(CSV_PATH, 'fertilizer_dataset.csv')

    files.download('fertilizer_model_package.zip')
    print("Download started — fertilizer_model_package.zip")

except ImportError:
    print("Files saved to fertilizer_model_files/ directory")
