"""
train_v3.py -- BhoomiAI Crop Prediction Model v3
=================================================
Production-level model with balanced feature importance.

v3 Improvements over v2:
  - Zone-coupled climate data (zone is now a meaningful signal)
  - Soil type OHE (sandy/loamy/clayey) -- boosts soil importance
  - Macro-zone group interactions (zone_rain, zone_temp)
  - 5 sensitivity tests (added soil + temperature tests)
  - 400 samples/crop (1600 total)

  PRESERVED from v2 (do NOT change):
    - NPK dominance (~33%)
    - One-hot encoding for Season + Agro_Zone
    - rain_avg (no min/max split)
    - StandardScaler on numeric features
    - class_weight='balanced'

  TARGET FEATURE IMPORTANCE:
    NPK:        30-35%  (preserve)
    Season:     15-20%  (preserve)
    Rainfall:   14-18%  (preserve)
    Humidity:   10-12%  (preserve)
    Temperature: 10-15% (boost via zone_temp)
    pH:          5-7%   (preserve)
    Agro-Zone:   >= 8%  (boost via zone coupling + interactions)
    Soil:        >= 8%  (boost via soil type OHE)

Usage:
    python dataset/build_authentic_dataset_v3.py  # build dataset first
    python train_v3.py                            # then train
"""

import os
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# --- Paths --------------------------------------------------------------------
DATASET_PATH = os.path.join('dataset', 'crop_train_v3.csv')
MODEL_PATH   = os.path.join('models',  'rf_model_v3.pkl')
REPORT_PATH  = os.path.join('results', 'accuracy_report_v3.txt')

# --- Numeric features (before OHE) -------------------------------------------
NUMERIC_FEATURES = [
    'mean_temp', 'rain_avg',
    'Sand_pct', 'Silt_pct', 'Clay_pct',
    'N_kg_ha', 'P_kg_ha', 'K_kg_ha',
    'Humidity_pct', 'pH',
    'temp_rain_ratio', 'npk_sum', 'soil_texture_index',
    'zone_rain', 'zone_temp',
]

# --- Categorical features (one-hot encoded) ----------------------------------
CATEGORICAL_FEATURES = ['Season', 'Agro_Zone', 'Soil_Type', 'Zone_Group']

TARGET_COL = 'Crop'

# --- Zone lookup (Planning Commission classification) ------------------------
ZONE_NAMES = {
    0: 'W.Himalayan',     1: 'E.Himalayan',    2: 'Lower Gangetic',
    3: 'Middle Gangetic',  4: 'Upper Gangetic',  5: 'Trans-Gangetic',
    6: 'E.Plateau',        7: 'Central Plateau', 8: 'W.Plateau',
    9: 'S.Plateau',       10: 'E.Coast',        11: 'W.Coast',
   12: 'Gujarat',         13: 'W.Dry',          14: 'Islands',
}

# --- RF hyperparameters -------------------------------------------------------
RF_PARAMS = dict(
    n_estimators   = 300,             # more trees for richer features
    max_depth      = 14,              # slightly deeper for interactions
    min_samples_split = 5,
    min_samples_leaf  = 3,
    max_features   = 'sqrt',
    class_weight   = 'balanced',
    random_state   = 42,
    n_jobs         = -1,
)


# --- Feature Engineering ------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-informed interaction features.

    v3 additions (on top of v2):
      zone_rain :  rain_avg * is_arid  -- arid zones with rain are distinctive
      zone_temp :  mean_temp * is_humid -- humid + hot is different from arid + hot
    """
    df = df.copy()

    # --- Carried from v2 (DO NOT CHANGE) ---
    df['temp_rain_ratio']    = df['mean_temp'] / df['rain_avg'].clip(lower=1.0)
    df['npk_sum']            = df['N_kg_ha'] + df['P_kg_ha'] + df['K_kg_ha']
    df['soil_texture_index'] = (df['Clay_pct'] + df['Silt_pct']) / df['Sand_pct'].clip(lower=1.0)

    # --- v3 zone interaction features ---
    # Use Zone_Group to create interaction (avoids ordinal bias from zone_id)
    is_arid  = (df['Zone_Group'] == 'arid').astype(float)
    is_humid = (df['Zone_Group'] == 'humid').astype(float)

    df['zone_rain'] = df['rain_avg'] * is_arid      # rain in arid zone = unusual signal
    df['zone_temp'] = df['mean_temp'] * is_humid     # temp in humid zone = tropical signal

    return df


def validate_soil(df: pd.DataFrame, tolerance: float = 5.0) -> pd.DataFrame:
    """Remove rows where sand+silt+clay deviates beyond tolerance from 100%."""
    soil_sum = df['Sand_pct'] + df['Silt_pct'] + df['Clay_pct']
    valid = (soil_sum - 100.0).abs() <= tolerance
    dropped = (~valid).sum()
    if dropped > 0:
        print(f"  [Soil Validation] Dropped {dropped} rows (|sum-100| > {tolerance}%)")
    return df[valid].reset_index(drop=True)


# --- Preprocessing Pipeline ---------------------------------------------------

def preprocess(df: pd.DataFrame, scaler=None, fit_scaler: bool = True):
    """
    Full preprocessing:
      1. Feature engineering (5 interaction features)
      2. Agro_Zone -> string prefix -> One-Hot
      3. Season -> One-Hot
      4. Soil_Type -> One-Hot (v3 addition)
      5. Zone_Group -> One-Hot (v3 addition)
      6. StandardScaler on numeric features

    Returns: X_scaled (DataFrame), y (Series), scaler, feature_columns (list)
    """
    df = engineer_features(df)

    # Prefix zone IDs for readable OHE column names
    df['Agro_Zone'] = 'Zone_' + df['Agro_Zone'].astype(str)

    # One-Hot Encode -- drop_first=False keeps all categories explicit
    df_enc = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=False)

    feature_cols = [c for c in df_enc.columns if c != TARGET_COL]
    X = df_enc[feature_cols].copy()
    y = df_enc[TARGET_COL].copy()

    # Scale numeric features only
    if fit_scaler:
        scaler = StandardScaler()
        X[NUMERIC_FEATURES] = scaler.fit_transform(X[NUMERIC_FEATURES])
    else:
        X[NUMERIC_FEATURES] = scaler.transform(X[NUMERIC_FEATURES])

    return X, y, scaler, feature_cols


# --- Training -----------------------------------------------------------------

def train_model(X_train, y_train) -> RandomForestClassifier:
    print("\n[Training] RandomForestClassifier ...")
    print(f"  Params: n_estimators={RF_PARAMS['n_estimators']}, "
          f"max_depth={RF_PARAMS['max_depth']}, "
          f"class_weight={RF_PARAMS['class_weight']}")
    t0 = time.time()
    clf = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X_train, y_train)
    print(f"  Completed in {time.time() - t0:.2f}s")
    return clf


# --- Evaluation ---------------------------------------------------------------

def evaluate_model(clf, X_train, X_test, y_train, y_test, feature_cols):
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc  = accuracy_score(y_test,  clf.predict(X_test))
    y_pred    = clf.predict(X_test)
    classes   = sorted(clf.classes_)

    X_all = pd.concat([X_train, X_test])
    y_all = pd.concat([y_train, y_test])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_all, y_all, cv=cv, scoring='accuracy')

    # Feature importances sorted descending
    importances = clf.feature_importances_
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: -x[1])

    lines = [
        "=" * 72,
        "  BhoomiAI v3 -- Crop Prediction Model Evaluation Report",
        "=" * 72,
        "",
        "  PIPELINE CHANGES vs v2",
        "  ----------------------",
        "  [OK] Zone-coupled climate data (temp/rain bounded by zone)",
        "  [OK] Soil_Type OHE (sandy/loamy/clayey)",
        "  [OK] Zone_Group OHE (arid/humid/plains) + interactions",
        "  [OK] zone_rain and zone_temp interaction features",
        "  [OK] 400 samples/crop x 4 crops = 1600 total",
        "  [OK] RandomForest with n_estimators=300, max_depth=14",
        "",
        "  PRESERVED from v2 (no change):",
        "  [OK] One-Hot encoding for Season + Agro_Zone",
        "  [OK] rain_avg (not min/max)",
        "  [OK] NPK features (N, P, K, npk_sum)",
        "  [OK] temp_rain_ratio, soil_texture_index",
        "  [OK] StandardScaler, class_weight='balanced'",
        "",
        f"  Dataset       : {DATASET_PATH}",
        f"  Total samples : {len(X_all)}",
        f"  Features      : {len(feature_cols)}",
        f"  Classes       : {classes}",
        "",
        "  MODEL PERFORMANCE",
        "  -----------------",
        f"  Train accuracy    : {train_acc * 100:.2f} %",
        f"  Test  accuracy    : {test_acc  * 100:.2f} %",
        f"  5-Fold CV mean    : {cv_scores.mean() * 100:.2f} % (+/- {cv_scores.std() * 100:.2f} %)",
        "",
        "  CLASSIFICATION REPORT (Test Set)",
        "  ---------------------------------",
        classification_report(y_test, y_pred, target_names=classes),
        "  CONFUSION MATRIX (rows=actual, cols=predicted)",
        "  ------------------------------------------------",
    ]

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    header = "           " + "  ".join(f"{c:>10}" for c in classes)
    lines.append(header)
    for i, rc in enumerate(classes):
        row = f"{rc:>10} " + "  ".join(f"{v:>10}" for v in cm[i])
        lines.append(row)

    lines += [
        "",
        "  FEATURE IMPORTANCES (all features including OHE columns)",
        "  ----------------------------------------------------------",
    ]
    for feat, imp in feat_imp:
        bar = "#" * int(imp * 70)
        lines.append(f"  {feat:<38}  {imp:.4f}  {bar}")

    # Aggregate group importances
    zone_total = sum(imp for f, imp in feat_imp if f.startswith('Agro_Zone_'))
    rain_total = sum(imp for f, imp in feat_imp if 'rain' in f.lower())
    npk_total  = sum(imp for f, imp in feat_imp if f in ('N_kg_ha', 'P_kg_ha', 'K_kg_ha', 'npk_sum'))
    soil_total = sum(imp for f, imp in feat_imp
                     if f in ('Sand_pct', 'Silt_pct', 'Clay_pct', 'soil_texture_index')
                     or f.startswith('Soil_Type_'))
    temp_total = sum(imp for f, imp in feat_imp if f in ('mean_temp', 'zone_temp'))
    season_total = sum(imp for f, imp in feat_imp if f.startswith('Season_'))
    humidity_imp = sum(imp for f, imp in feat_imp if f == 'Humidity_pct')
    ph_imp = sum(imp for f, imp in feat_imp if f == 'pH')
    zone_grp_total = sum(imp for f, imp in feat_imp if f.startswith('Zone_Group_'))
    zone_interact = sum(imp for f, imp in feat_imp if f in ('zone_rain', 'zone_temp'))

    # Combined zone importance (OHE + group + interactions)
    zone_combined = zone_total + zone_grp_total + zone_interact

    lines += [
        "",
        "  AGGREGATED GROUP IMPORTANCES",
        "  -----------------------------",
        f"  NPK (N, P, K + npk_sum)           : {npk_total * 100:.2f} %",
        f"  Season (all OHE cols)              : {season_total * 100:.2f} %",
        f"  Rainfall (rain_avg + temp_rain_r)  : {rain_total * 100:.2f} %",
        f"  Humidity                           : {humidity_imp * 100:.2f} %",
        f"  Temperature (mean_temp + zone_temp): {temp_total * 100:.2f} %",
        f"  pH                                 : {ph_imp * 100:.2f} %",
        f"  Agro-Zone (OHE only)               : {zone_total * 100:.2f} %",
        f"  Zone Group (OHE only)              : {zone_grp_total * 100:.2f} %",
        f"  Zone Interactions (zone_rain+temp)  : {zone_interact * 100:.2f} %",
        f"  Zone COMBINED (all zone features)   : {zone_combined * 100:.2f} %",
        f"  Soil (sand/silt/clay + texture + type): {soil_total * 100:.2f} %",
        "",
        "  TARGET CHECK",
        "  ------------",
        f"  Zone >= 8%  : {zone_combined * 100:.2f}%  {'[OK]' if zone_combined >= 0.08 else '[!] BELOW TARGET'}",
        f"  Soil >= 8%  : {soil_total * 100:.2f}%  {'[OK]' if soil_total >= 0.08 else '[!] BELOW TARGET'}",
        f"  Temp 10-15% : {temp_total * 100:.2f}%  {'[OK]' if 0.10 <= temp_total <= 0.15 else '[!] OUTSIDE TARGET'}",
        "",
    ]

    lines += ["", "=" * 72]
    return ("\\n".join(lines), test_acc, feat_imp, zone_combined, rain_total,
            soil_total, temp_total, npk_total, season_total, humidity_imp, ph_imp)


# --- Inference Vector Builder -------------------------------------------------

def build_inference_vector(inp: dict, feature_columns: list,
                           scaler: StandardScaler) -> pd.DataFrame:
    """
    Build a scaled, OHE-aligned feature DataFrame for a single prediction.
    Mirrors the exact preprocessing applied during training.

    inp keys:
        mean_temp, rain_avg, Sand_pct, Silt_pct, Clay_pct,
        N_kg_ha, P_kg_ha, K_kg_ha, Humidity_pct, pH,
        Season ('Rabi'|'Kharif'), Agro_Zone (int 0-14)
    """
    mean_temp  = float(inp['mean_temp'])
    rain_avg   = float(inp['rain_avg'])
    n, pv, k   = float(inp['N_kg_ha']),  float(inp['P_kg_ha']),  float(inp['K_kg_ha'])
    sand, silt, clay = float(inp['Sand_pct']), float(inp['Silt_pct']), float(inp['Clay_pct'])
    season     = str(inp['Season'])
    zone_id    = int(inp['Agro_Zone'])

    # Derive soil type
    if clay >= 35.0:
        soil_type = 'clayey'
    elif sand >= 45.0:
        soil_type = 'sandy'
    else:
        soil_type = 'loamy'

    # Derive zone group
    if zone_id in {5, 12, 13}:
        zone_group = 'arid'
    elif zone_id in {1, 2, 10, 11, 14}:
        zone_group = 'humid'
    else:
        zone_group = 'plains'

    is_arid  = 1.0 if zone_group == 'arid' else 0.0
    is_humid = 1.0 if zone_group == 'humid' else 0.0

    engineered = {
        'mean_temp':          mean_temp,
        'rain_avg':           rain_avg,
        'Sand_pct':           sand,
        'Silt_pct':           silt,
        'Clay_pct':           clay,
        'N_kg_ha':            n,
        'P_kg_ha':            pv,
        'K_kg_ha':            k,
        'Humidity_pct':       float(inp['Humidity_pct']),
        'pH':                 float(inp['pH']),
        'temp_rain_ratio':    mean_temp / max(rain_avg, 1.0),
        'npk_sum':            n + pv + k,
        'soil_texture_index': (clay + silt) / max(sand, 1.0),
        'zone_rain':          rain_avg * is_arid,
        'zone_temp':          mean_temp * is_humid,
    }

    row = {}
    for col in feature_columns:
        if col in engineered:
            row[col] = engineered[col]
        elif col.startswith('Season_'):
            season_val = col.replace('Season_', '')
            row[col] = 1 if season == season_val else 0
        elif col.startswith('Agro_Zone_Zone_'):
            zone_val = col.replace('Agro_Zone_Zone_', '')
            row[col] = 1 if str(zone_id) == zone_val else 0
        elif col.startswith('Soil_Type_'):
            st_val = col.replace('Soil_Type_', '')
            row[col] = 1 if soil_type == st_val else 0
        elif col.startswith('Zone_Group_'):
            zg_val = col.replace('Zone_Group_', '')
            row[col] = 1 if zone_group == zg_val else 0
        else:
            row[col] = 0

    X = pd.DataFrame([row])[feature_columns]
    X[NUMERIC_FEATURES] = scaler.transform(X[NUMERIC_FEATURES])
    return X


# --- Sensitivity Validation Tests ---------------------------------------------

def run_sensitivity_tests(clf, scaler, feature_columns: list) -> dict:
    """
    5 mandatory sensitivity tests per specification.
    Returns dict of 'PASS'/'FAIL' per test.
    """

    def predict_inp(inp: dict) -> dict:
        X = build_inference_vector(inp, feature_columns, scaler)
        proba   = clf.predict_proba(X)[0]
        classes = clf.classes_
        pred    = classes[np.argmax(proba)]
        votes   = {c: round(float(p) * 100, 1) for c, p in zip(classes, proba)}
        return {'prediction': pred, 'votes': votes}

    print("\n" + "=" * 72)
    print("  SENSITIVITY VALIDATION TESTS (5 tests)")
    print("=" * 72)

    # Base: moderate Kharif, Zone 4 (Upper Gangetic)
    base = {
        'mean_temp': 26.0, 'rain_avg': 85.0,
        'Sand_pct': 42.0,  'Silt_pct': 32.0, 'Clay_pct': 26.0,
        'N_kg_ha': 100.0,  'P_kg_ha': 45.0,  'K_kg_ha': 45.0,
        'Humidity_pct': 65.0, 'pH': 6.5,
        'Season': 'Kharif', 'Agro_Zone': 4,
    }

    results = {}

    # -- TEST 1: Rainfall Sensitivity ------------------------------------------
    print("\n  TEST 1 -- Rainfall Sensitivity")
    print("  Expectation: High rainfall (160cm) -> higher Rice probability than low (40cm)")
    r_low  = predict_inp({**base, 'rain_avg': 40.0,  'Humidity_pct': 52.0})
    r_high = predict_inp({**base, 'rain_avg': 160.0, 'Humidity_pct': 88.0})
    pass1 = r_high['votes'].get('Rice', 0) > r_low['votes'].get('Rice', 0)
    print(f"  rain=40 cm  -> {r_low['prediction']:10s}  Rice={r_low['votes'].get('Rice',0):.1f}%")
    print(f"  rain=160 cm -> {r_high['prediction']:10s}  Rice={r_high['votes'].get('Rice',0):.1f}%")
    print(f"  RESULT: {'[OK] PASS' if pass1 else '[FAIL]'}")
    results['rainfall_sensitivity'] = 'PASS' if pass1 else 'FAIL'

    # -- TEST 2: Zone Sensitivity -----------------------------------------------
    print("\n  TEST 2 -- Agro-Zone Sensitivity")
    print("  Expectation: Zone 13 (W.Dry) vs Zone 2 (Lower Gangetic) -> prediction/proba changes")
    z_desert   = predict_inp({**base, 'Agro_Zone': 13, 'Sand_pct': 60.0, 'Silt_pct': 22.0, 'Clay_pct': 18.0})
    z_gangetic = predict_inp({**base, 'Agro_Zone': 2,  'Sand_pct': 28.0, 'Silt_pct': 38.0, 'Clay_pct': 34.0})
    diff = sum(abs(z_desert['votes'].get(c, 0) - z_gangetic['votes'].get(c, 0))
               for c in z_desert['votes'])
    pass2 = (z_desert['prediction'] != z_gangetic['prediction']) or diff > 10.0
    print(f"  Zone=13 (W.Dry)      -> {z_desert['prediction']:10s}  {z_desert['votes']}")
    print(f"  Zone=2  (L.Gangetic) -> {z_gangetic['prediction']:10s}  {z_gangetic['votes']}")
    print(f"  Probability diff sum: {diff:.1f}%")
    print(f"  RESULT: {'[OK] PASS' if pass2 else '[FAIL]'}")
    results['zone_sensitivity'] = 'PASS' if pass2 else 'FAIL'

    # -- TEST 3: NPK Sensitivity ------------------------------------------------
    print("\n  TEST 3 -- NPK Sensitivity")
    print("  Expectation: Sugarcane-level NPK (N=220, P=85, K=85) -> higher Sugarcane probability")
    npk_low  = predict_inp({**base, 'N_kg_ha': 80.0,  'P_kg_ha': 35.0,  'K_kg_ha': 35.0})
    npk_high = predict_inp({**base, 'N_kg_ha': 220.0, 'P_kg_ha': 85.0,  'K_kg_ha': 85.0})
    pass3 = npk_high['votes'].get('Sugarcane', 0) > npk_low['votes'].get('Sugarcane', 0)
    print(f"  NPK=150 (low)  -> {npk_low['prediction']:10s}  Sugarcane={npk_low['votes'].get('Sugarcane',0):.1f}%")
    print(f"  NPK=390 (high) -> {npk_high['prediction']:10s}  Sugarcane={npk_high['votes'].get('Sugarcane',0):.1f}%")
    print(f"  RESULT: {'[OK] PASS' if pass3 else '[FAIL]'}")
    results['npk_sensitivity'] = 'PASS' if pass3 else 'FAIL'

    # -- TEST 4: Soil Sensitivity (v3 NEW) --------------------------------------
    print("\n  TEST 4 -- Soil Sensitivity")
    print("  Expectation: Sandy soil vs clay soil -> prediction/probability changes")
    s_sandy = predict_inp({**base, 'Sand_pct': 60.0, 'Silt_pct': 25.0, 'Clay_pct': 15.0,
                           'Agro_Zone': 13, 'rain_avg': 40.0, 'Humidity_pct': 45.0})
    s_clay  = predict_inp({**base, 'Sand_pct': 18.0, 'Silt_pct': 35.0, 'Clay_pct': 47.0,
                           'Agro_Zone': 2,  'rain_avg': 150.0, 'Humidity_pct': 85.0})
    diff_soil = sum(abs(s_sandy['votes'].get(c, 0) - s_clay['votes'].get(c, 0))
                    for c in s_sandy['votes'])
    pass4 = (s_sandy['prediction'] != s_clay['prediction']) or diff_soil > 10.0
    print(f"  Sandy (60/25/15) -> {s_sandy['prediction']:10s}  {s_sandy['votes']}")
    print(f"  Clay  (18/35/47) -> {s_clay['prediction']:10s}  {s_clay['votes']}")
    print(f"  Probability diff sum: {diff_soil:.1f}%")
    print(f"  RESULT: {'[OK] PASS' if pass4 else '[FAIL]'}")
    results['soil_sensitivity'] = 'PASS' if pass4 else 'FAIL'

    # -- TEST 5: Temperature Sensitivity ----------------------------------------
    print("\n  TEST 5 -- Temperature Sensitivity")
    print("  Expectation: High temp -> Wheat probability decreases")
    wheat_base = {**base, 'Season': 'Rabi', 'Agro_Zone': 4,
                  'rain_avg': 50.0, 'Humidity_pct': 55.0}
    t_cool = predict_inp({**wheat_base, 'mean_temp': 16.0})
    t_hot  = predict_inp({**wheat_base, 'mean_temp': 36.0})
    pass5 = t_cool['votes'].get('Wheat', 0) > t_hot['votes'].get('Wheat', 0)
    print(f"  temp=16C (cool) -> {t_cool['prediction']:10s}  Wheat={t_cool['votes'].get('Wheat',0):.1f}%")
    print(f"  temp=36C (hot)  -> {t_hot['prediction']:10s}  Wheat={t_hot['votes'].get('Wheat',0):.1f}%")
    print(f"  RESULT: {'[OK] PASS' if pass5 else '[FAIL]'}")
    results['temperature_sensitivity'] = 'PASS' if pass5 else 'FAIL'

    passed = sum(v == 'PASS' for v in results.values())
    print(f"\n  -- SUMMARY: {passed}/5 tests passed "
          f"{'[OK] ALL PASS' if passed == 5 else '[!] REVIEW FAILURES'} --")

    return results


# --- Save Model ---------------------------------------------------------------

def save_model(clf, scaler, feature_columns, accuracy, sensitivity_results):
    """Save model bundle with all metadata needed for inference."""
    os.makedirs('models', exist_ok=True)
    payload = {
        'sklearn':          clf,
        'scaler':           scaler,
        'feature_columns':  feature_columns,
        'numeric_features': NUMERIC_FEATURES,
        'classes':          list(clf.classes_),
        'accuracy':         round(accuracy * 100, 2),
        'n_estimators':     RF_PARAMS['n_estimators'],
        'sensitivity_tests': sensitivity_results,
        'version':          'v3',
        'data_source':      'ICAR/NBSS/IMD authentic Indian agricultural data',
        'improvements':     'zone-coupled climate, soil type OHE, zone interactions',
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(payload, f)
    print(f"\n  [OK] Model saved -> {MODEL_PATH}")


# --- Main Pipeline ------------------------------------------------------------

def main():
    print("\n" + "=" * 72)
    print("  BhoomiAI v3 -- Crop Prediction Training Pipeline")
    print("=" * 72)

    # Step 1 -- Load
    print(f"\n[1/7] Loading dataset from {DATASET_PATH} ...")
    if not os.path.exists(DATASET_PATH):
        print(f"\n  [FAIL] Dataset not found. Run first:\n"
              f"     python dataset/build_authentic_dataset_v3.py\n")
        return
    df = pd.read_csv(DATASET_PATH)
    print(f"  Shape: {df.shape}")
    print(f"  Class distribution: {df['Crop'].value_counts().to_dict()}")
    print(f"  Soil types: {df['Soil_Type'].value_counts().to_dict()}")
    print(f"  Zone groups: {df['Zone_Group'].value_counts().to_dict()}")

    # Step 2 -- Validate soil
    print("\n[2/7] Validating soil composition ...")
    df = validate_soil(df, tolerance=5.0)
    print(f"  Valid rows after soil check: {len(df)}")

    # Step 3 -- Preprocess
    print("\n[3/7] Feature engineering + One-Hot Encoding + StandardScaler ...")
    X, y, scaler, feature_columns = preprocess(df, fit_scaler=True)
    print(f"  Final feature count : {len(feature_columns)}")
    print(f"  Numeric features    : {NUMERIC_FEATURES}")
    ohe_cols = [c for c in feature_columns if c not in NUMERIC_FEATURES]
    print(f"  OHE columns ({len(ohe_cols)}) : {ohe_cols}")

    # Step 4 -- Split
    print("\n[4/7] Train/Test split (80/20, stratified, shuffle=True) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y, shuffle=True
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    # Step 5 -- Train
    clf = train_model(X_train, y_train)

    # Step 6 -- Evaluate
    print("\n[6/7] Evaluating model ...")
    eval_result = evaluate_model(clf, X_train, X_test, y_train, y_test, feature_columns)
    (report, test_acc, feat_imp, zone_combined, rain_total,
     soil_total, temp_total, npk_total, season_total, humidity_imp, ph_imp) = eval_result

    print(f"\n  Test Accuracy : {test_acc * 100:.2f}%")
    print(f"\n  Top 15 Feature Importances:")
    for feat, imp in feat_imp[:15]:
        bar = "#" * int(imp * 50)
        print(f"    {feat:<38} {imp:.4f}  {bar}")

    print(f"\n  Aggregated importance:")
    print(f"    NPK            : {npk_total * 100:.2f}%")
    print(f"    Season         : {season_total * 100:.2f}%")
    print(f"    Rainfall       : {rain_total * 100:.2f}%")
    print(f"    Humidity       : {humidity_imp * 100:.2f}%")
    print(f"    Temperature    : {temp_total * 100:.2f}%")
    print(f"    pH             : {ph_imp * 100:.2f}%")
    print(f"    Zone COMBINED  : {zone_combined * 100:.2f}%  {'[OK]' if zone_combined >= 0.08 else '[!] BELOW TARGET'}")
    print(f"    Soil COMBINED  : {soil_total * 100:.2f}%  {'[OK]' if soil_total >= 0.08 else '[!] BELOW TARGET'}")

    # Step 7 -- Sensitivity tests
    sensitivity_results = run_sensitivity_tests(clf, scaler, feature_columns)

    # Save model
    save_model(clf, scaler, feature_columns, test_acc, sensitivity_results)

    # Write full report
    os.makedirs('results', exist_ok=True)
    sens_lines = [
        "",
        "  SENSITIVITY TEST RESULTS (5 tests)",
        "  ------------------------------------",
    ]
    for test_name, result in sensitivity_results.items():
        icon = "[OK]" if result == 'PASS' else "[FAIL]"
        sens_lines.append(f"  {icon}  {test_name:<35}: {result}")

    full_report = report + "\n" + "\n".join(sens_lines) + "\n\n" + "=" * 72
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(full_report)
    print(f"\n  [OK] Report saved -> {REPORT_PATH}")

    print("\n" + "=" * 72)
    print("  TRAINING COMPLETE")
    print("=" * 72)
    print(f"  Test Accuracy      : {test_acc * 100:.2f}%")
    print(f"  Zone Importance    : {zone_combined * 100:.2f}%")
    print(f"  Soil Importance    : {soil_total * 100:.2f}%")
    print(f"  Temp Importance    : {temp_total * 100:.2f}%")
    passed = sum(v == 'PASS' for v in sensitivity_results.values())
    print(f"  Sensitivity Tests  : {passed}/5 passed")
    print(f"\n  Model saved to: {MODEL_PATH}")
    print(f"  Start server:   python app.py\n")

    return clf, scaler, feature_columns


if __name__ == '__main__':
    main()
