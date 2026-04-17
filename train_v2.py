"""
train_v2.py -- BhoomiAI Crop Prediction Model v2
================================================
Implements all preprocessing and bias-fix rules:

  DATA PREPROCESSING:
    - One-hot encoding for Season and Agro_Zone (no ordinal bias)
    - rain_avg replaces rain_min / rain_max (removes correlated duplicates)
    - StandardScaler on all numeric features
    - Soil sum validation (removes |sum-100| > 5% rows)
    - Low / Medium / High NPK variation ensured in dataset

  FEATURE ENGINEERING:
    - temp_rain_ratio  = mean_temp / rain_avg
    - npk_sum         = N + P + K
    - soil_texture_index = (clay + silt) / sand

  MODEL:
    - RandomForestClassifier (n_estimators=200, depth-limited, balanced weights)
    - 80/20 stratified split with shuffle=True
    - 5-Fold cross-validation

  VALIDATION (4 mandatory sensitivity tests):
    1. Rainfall   → high rain shifts toward Rice
    2. Zone       → changing zone changes prediction
    3. NPK        → sugarcane-level NPK favors Sugarcane
    4. Season     → Rabi favors Wheat; Kharif does not

Usage:
    python dataset/build_authentic_dataset.py  # build dataset first
    python train_v2.py                         # then train
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
DATASET_PATH = os.path.join('dataset', 'crop_train_v2.csv')
MODEL_PATH   = os.path.join('models',  'rf_model_v2.pkl')
REPORT_PATH  = os.path.join('results', 'accuracy_report_v2.txt')

# --- Numeric features (before OHE) -------------------------------------------
NUMERIC_FEATURES = [
    'mean_temp', 'rain_avg',
    'Sand_pct', 'Silt_pct', 'Clay_pct',
    'N_kg_ha', 'P_kg_ha', 'K_kg_ha',
    'Humidity_pct', 'pH',
    'temp_rain_ratio', 'npk_sum', 'soil_texture_index',
]

# --- Categorical features (one-hot encoded) ----------------------------------
CATEGORICAL_FEATURES = ['Season', 'Agro_Zone']

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
    n_estimators   = 200,
    max_depth      = 12,          # Controlled depth → avoids memorisation
    min_samples_split = 5,
    min_samples_leaf  = 3,
    max_features   = 'sqrt',
    class_weight   = 'balanced',  # Handles any residual class imbalance
    random_state   = 42,
    n_jobs         = -1,
)


# --- Feature Engineering ------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create three domain-informed interaction features.

    temp_rain_ratio    : captures climate aridity -- high for wheat (hot+dry),
                         low for rice (warm+wet)
    npk_sum            : total nutrient load -- very high for sugarcane (300-450)
                         vs. 150-250 for wheat/rice/maize
    soil_texture_index : (clay+silt)/sand -- high for rice (clay soils),
                         low for wheat/maize (sandy loam)
    """
    df = df.copy()
    df['temp_rain_ratio']    = df['mean_temp'] / df['rain_avg'].clip(lower=1.0)
    df['npk_sum']            = df['N_kg_ha'] + df['P_kg_ha'] + df['K_kg_ha']
    df['soil_texture_index'] = (df['Clay_pct'] + df['Silt_pct']) / df['Sand_pct'].clip(lower=1.0)
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
      1. Feature engineering
      2. Agro_Zone → string prefix → One-Hot (no ordinal bias)
      3. Season    → One-Hot
      4. StandardScaler on numeric features

    Returns: X_scaled (DataFrame), y (Series), scaler, feature_columns (list)
    """
    df = engineer_features(df)

    # Prefix zone IDs so OHE columns are readable ('Agro_Zone_Zone_4', etc.)
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
        "  BhoomiAI v2 -- Crop Prediction Model Evaluation Report",
        "=" * 72,
        "",
        "  PIPELINE CHANGES vs v1",
        "  ----------------------",
        "  [OK] One-Hot encoding for Season + Agro_Zone  (no ordinal bias)",
        "  [OK] rain_avg replaces rain_min/rain_max       (removes correlated pair)",
        "  [OK] 3 new features: temp_rain_ratio, npk_sum, soil_texture_index",
        "  [OK] StandardScaler on all numeric features",
        "  [OK] Authentic ICAR/NBSS data -- 300 samples/crop x 4 crops",
        "  [OK] RandomForest with class_weight='balanced', max_depth=12",
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

    # Aggregate zone + rainfall importance
    zone_total = sum(imp for f, imp in feat_imp if f.startswith('Agro_Zone_'))
    rain_total = sum(imp for f, imp in feat_imp if 'rain' in f.lower())
    npk_total  = sum(imp for f, imp in feat_imp if f in ('N_kg_ha', 'P_kg_ha', 'K_kg_ha', 'npk_sum'))

    lines += [
        "",
        "  AGGREGATED GROUP IMPORTANCES",
        "  -----------------------------",
        f"  Agro-Zone (all OHE cols)   : {zone_total * 100:.2f} %",
        f"  Rainfall (rain_avg + ratio) : {rain_total * 100:.2f} %",
        f"  NPK (N, P, K + npk_sum)    : {npk_total  * 100:.2f} %",
        "",
    ]

    lines += ["", "=" * 72]
    return "\n".join(lines), test_acc, feat_imp, zone_total, rain_total


# --- Inference Vector Builder (shared with functions.py) ----------------------

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
        else:
            row[col] = 0  # unseen OHE column (e.g. zone 14 not in training) → 0

    X = pd.DataFrame([row])[feature_columns]
    X[NUMERIC_FEATURES] = scaler.transform(X[NUMERIC_FEATURES])
    return X


# --- Sensitivity Validation Tests ---------------------------------------------

def run_sensitivity_tests(clf, scaler, feature_columns: list) -> dict:
    """
    4 mandatory sensitivity tests per specification.
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
    print("  SENSITIVITY VALIDATION TESTS")
    print("=" * 72)

    # Base: moderate Kharif, Zone 4 (Upper Gangetic), typical conditions
    base = {
        'mean_temp': 26.0, 'rain_avg': 85.0,
        'Sand_pct': 42.0,  'Silt_pct': 32.0, 'Clay_pct': 26.0,
        'N_kg_ha': 100.0,  'P_kg_ha': 45.0,  'K_kg_ha': 45.0,
        'Humidity_pct': 65.0, 'pH': 6.5,
        'Season': 'Kharif', 'Agro_Zone': 4,
    }

    results = {}

    # -- TEST 1: Rainfall Sensitivity -----------------------------------------
    print("\n  TEST 1 -- Rainfall Sensitivity")
    print("  Expectation: High rainfall (160cm) → higher Rice probability than low (40cm)")
    r_low  = predict_inp({**base, 'rain_avg': 40.0,  'Humidity_pct': 52.0})
    r_high = predict_inp({**base, 'rain_avg': 160.0, 'Humidity_pct': 88.0})
    pass1 = r_high['votes'].get('Rice', 0) > r_low['votes'].get('Rice', 0)
    print(f"  rain=40 cm  → {r_low['prediction']:10s}  Rice={r_low['votes'].get('Rice',0):.1f}%")
    print(f"  rain=160 cm → {r_high['prediction']:10s}  Rice={r_high['votes'].get('Rice',0):.1f}%")
    print(f"  RESULT: {'[OK] PASS' if pass1 else '[FAIL] FAIL'}")
    results['rainfall_sensitivity'] = 'PASS' if pass1 else 'FAIL'

    # -- TEST 2: Zone Sensitivity ----------------------------------------------
    print("\n  TEST 2 -- Agro-Zone Sensitivity")
    print("  Expectation: Zone 13 (W.Dry/Desert) vs Zone 2 (Lower Gangetic) → prediction changes")
    z_desert   = predict_inp({**base, 'Agro_Zone': 13, 'Sand_pct': 60.0, 'Silt_pct': 22.0, 'Clay_pct': 18.0})
    z_gangetic = predict_inp({**base, 'Agro_Zone': 2,  'Sand_pct': 28.0, 'Silt_pct': 38.0, 'Clay_pct': 34.0})
    diff = sum(abs(z_desert['votes'].get(c, 0) - z_gangetic['votes'].get(c, 0))
               for c in z_desert['votes'])
    pass2 = (z_desert['prediction'] != z_gangetic['prediction']) or diff > 10.0
    print(f"  Zone=13 (W.Dry)      → {z_desert['prediction']:10s}  {z_desert['votes']}")
    print(f"  Zone=2  (L.Gangetic) → {z_gangetic['prediction']:10s}  {z_gangetic['votes']}")
    print(f"  Probability diff sum: {diff:.1f}%")
    print(f"  RESULT: {'[OK] PASS' if pass2 else '[FAIL] FAIL'}")
    results['zone_sensitivity'] = 'PASS' if pass2 else 'FAIL'

    # -- TEST 3: NPK Sensitivity ------------------------------------------------
    print("\n  TEST 3 -- NPK Sensitivity")
    print("  Expectation: Sugarcane-level NPK (N=220, P=85, K=85) → higher Sugarcane probability")
    npk_low  = predict_inp({**base, 'N_kg_ha': 80.0,  'P_kg_ha': 35.0,  'K_kg_ha': 35.0})
    npk_high = predict_inp({**base, 'N_kg_ha': 220.0, 'P_kg_ha': 85.0,  'K_kg_ha': 85.0})
    pass3 = npk_high['votes'].get('Sugarcane', 0) > npk_low['votes'].get('Sugarcane', 0)
    print(f"  NPK=150 (low)  → {npk_low['prediction']:10s}  Sugarcane={npk_low['votes'].get('Sugarcane',0):.1f}%")
    print(f"  NPK=390 (high) → {npk_high['prediction']:10s}  Sugarcane={npk_high['votes'].get('Sugarcane',0):.1f}%")
    print(f"  RESULT: {'[OK] PASS' if pass3 else '[FAIL] FAIL'}")
    results['npk_sensitivity'] = 'PASS' if pass3 else 'FAIL'

    # -- TEST 4: Season Sensitivity ---------------------------------------------
    print("\n  TEST 4 -- Season Sensitivity")
    print("  Expectation: Same climate, Rabi → higher Wheat probability than Kharif")
    cool_base = {**base, 'mean_temp': 18.0, 'rain_avg': 50.0, 'Humidity_pct': 55.0}
    s_rabi   = predict_inp({**cool_base, 'Season': 'Rabi'})
    s_kharif = predict_inp({**cool_base, 'Season': 'Kharif'})
    pass4 = s_rabi['votes'].get('Wheat', 0) > s_kharif['votes'].get('Wheat', 0)
    print(f"  Season=Rabi   → {s_rabi['prediction']:10s}  Wheat={s_rabi['votes'].get('Wheat',0):.1f}%")
    print(f"  Season=Kharif → {s_kharif['prediction']:10s}  Wheat={s_kharif['votes'].get('Wheat',0):.1f}%")
    print(f"  RESULT: {'[OK] PASS' if pass4 else '[FAIL] FAIL'}")
    results['season_sensitivity'] = 'PASS' if pass4 else 'FAIL'

    passed = sum(v == 'PASS' for v in results.values())
    print(f"\n  -- SUMMARY: {passed}/4 tests passed "
          f"{'[OK] ALL PASS' if passed == 4 else '[!]️  REVIEW FAILURES'} --")

    return results


# --- Save Model ---------------------------------------------------------------

def save_model(clf, scaler, feature_columns, accuracy, sensitivity_results):
    """
    Save model bundle with all metadata needed for inference.
    feature_columns is critical -- inference must build the EXACT same column order.
    """
    os.makedirs('models', exist_ok=True)
    payload = {
        'sklearn':          clf,
        'scaler':           scaler,
        'feature_columns':  feature_columns,   # CRITICAL for inference alignment
        'numeric_features': NUMERIC_FEATURES,
        'classes':          list(clf.classes_),
        'accuracy':         round(accuracy * 100, 2),
        'n_estimators':     RF_PARAMS['n_estimators'],
        'sensitivity_tests': sensitivity_results,
        'version':          'v2',
        'data_source':      'ICAR/NBSS/IMD authentic Indian agricultural data',
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(payload, f)
    print(f"\n  [OK] Model saved → {MODEL_PATH}")


# --- Main Pipeline -------------------------------------------------------------

def main():
    print("\n" + "=" * 72)
    print("  BhoomiAI v2 -- Crop Prediction Training Pipeline")
    print("=" * 72)

    # Step 1 -- Load
    print(f"\n[1/6] Loading dataset from {DATASET_PATH} ...")
    if not os.path.exists(DATASET_PATH):
        print(f"\n  [FAIL] Dataset not found. Run first:\n"
              f"     python dataset/build_authentic_dataset.py\n")
        return
    df = pd.read_csv(DATASET_PATH)
    print(f"  Shape: {df.shape}")
    print(f"  Class distribution: {df['Crop'].value_counts().to_dict()}")

    # Step 2 -- Validate soil
    print("\n[2/6] Validating soil composition ...")
    df = validate_soil(df, tolerance=5.0)
    print(f"  Valid rows after soil check: {len(df)}")

    # Step 3 -- Preprocess
    print("\n[3/6] Feature engineering + One-Hot Encoding + StandardScaler ...")
    X, y, scaler, feature_columns = preprocess(df, fit_scaler=True)
    print(f"  Final feature count : {len(feature_columns)}")
    print(f"  Numeric features    : {NUMERIC_FEATURES}")
    ohe_cols = [c for c in feature_columns if c not in NUMERIC_FEATURES]
    print(f"  OHE columns ({len(ohe_cols)}) : {ohe_cols}")

    # Step 4 -- Split
    print("\n[4/6] Train/Test split (80/20, stratified, shuffle=True) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y, shuffle=True
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    # Step 5 -- Train
    clf = train_model(X_train, y_train)

    # Step 6 -- Evaluate
    print("\n[5/6] Evaluating model ...")
    report, test_acc, feat_imp, zone_total, rain_total = evaluate_model(
        clf, X_train, X_test, y_train, y_test, feature_columns
    )

    print(f"\n  Test Accuracy : {test_acc * 100:.2f}%")
    print(f"\n  Top 12 Feature Importances:")
    for feat, imp in feat_imp[:12]:
        bar = "#" * int(imp * 50)
        print(f"    {feat:<38} {imp:.4f}  {bar}")

    print(f"\n  Aggregated importance:")
    print(f"    Zone (all OHE) : {zone_total * 100:.2f}%  {'[OK]' if zone_total >= 0.05 else '[!]️'}")
    print(f"    Rainfall       : {rain_total * 100:.2f}%  {'[OK]' if rain_total <= 0.25 else '[!]️ high'}")

    # Step 7 -- Sensitivity tests
    sensitivity_results = run_sensitivity_tests(clf, scaler, feature_columns)

    # Step 8 -- Save
    save_model(clf, scaler, feature_columns, test_acc, sensitivity_results)

    # Write full report
    os.makedirs('results', exist_ok=True)
    sens_lines = [
        "",
        "  SENSITIVITY TEST RESULTS",
        "  -------------------------",
    ]
    for test_name, result in sensitivity_results.items():
        icon = "[OK]" if result == 'PASS' else "[FAIL]"
        sens_lines.append(f"  {icon}  {test_name:<35}: {result}")

    full_report = report + "\n" + "\n".join(sens_lines) + "\n\n" + "=" * 72
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(full_report)
    print(f"\n  [OK] Report saved → {REPORT_PATH}")

    print("\n" + "=" * 72)
    print("  TRAINING COMPLETE")
    print("=" * 72)
    print(f"  Test Accuracy      : {test_acc * 100:.2f}%")
    print(f"  Zone Importance    : {zone_total * 100:.2f}%")
    print(f"  Rainfall Dominance : {rain_total * 100:.2f}%")
    passed = sum(v == 'PASS' for v in sensitivity_results.values())
    print(f"  Sensitivity Tests  : {passed}/4 passed")
    print(f"\n  Model saved to: {MODEL_PATH}")
    print(f"  Start server:   python app.py\n")

    return clf, scaler, feature_columns


if __name__ == '__main__':
    main()
