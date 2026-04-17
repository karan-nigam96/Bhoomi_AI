"""
feature_engineering.py — Temperature Feature Engineering for BhoomiAI
======================================================================
Replaces 6 temperature-related features with 2 engineered features:
  - mean_temp = (Temp_min_C + Temp_max_C) / 2
  - gdd (Growing Degree Days) = mean_temp - base_temp

This simplifies the model while preserving predictive power.

Usage:
    python feature_engineering.py

Outputs:
    dataset/crop_train_gdd.csv — Dataset with engineered features
    models/rf_model_gdd.pkl    — Trained model with new features
    results/accuracy_report_gdd.txt — Evaluation report
"""

import os
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# ─── Configuration ────────────────────────────────────────────────────────────

# Base temperature for Growing Degree Days calculation (°C)
# Using 10°C as specified for rice crops
BASE_TEMP = 10.0

# Paths
INPUT_DATASET = os.path.join('dataset', 'crop_train_ml.csv')
OUTPUT_DATASET = os.path.join('dataset', 'crop_train_gdd.csv')
MODEL_PATH = os.path.join('models', 'rf_model_gdd.pkl')
RESULTS_DIR = 'results'
REPORT_PATH = os.path.join(RESULTS_DIR, 'accuracy_report_gdd.txt')

# Old temperature columns to remove
OLD_TEMP_COLS = [
    'Temp_min_C',
    'Temp_max_C',
    'Sow_temp_min',
    'Sow_temp_max',
    'Harvest_temp_min',
    'Harvest_temp_max',
]

# New engineered feature columns
NEW_TEMP_COLS = [
    'mean_temp',  # (Temp_min_C + Temp_max_C) / 2
    'gdd',        # Growing Degree Days = mean_temp - BASE_TEMP
]

# Final feature list for model training (after engineering)
FEATURE_COLS = [
    'mean_temp', 'gdd',                       # New engineered features
    'Rain_min_cm', 'Rain_max_cm',             # Rainfall features
    'Sand_pct', 'Clay_pct', 'Silt_pct',       # Soil composition
    'Nitrogen_N_kg_ha', 'Phosphorus_P_kg_ha', 'Potassium_K_kg_ha',  # NPK
    'Humidity_pct', 'pH',                     # Climate & soil pH
    'Season_code', 'Agro_Zone',               # Categorical features
]

TARGET_COL = 'Crop'

# Model hyperparameters
RF_PARAMS = dict(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
)


# ─── Feature Engineering Functions ────────────────────────────────────────────

def create_temperature_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered temperature features and remove old ones.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with original temperature columns
    
    Returns
    -------
    pd.DataFrame
        DataFrame with new engineered features (mean_temp, gdd)
        and old temperature columns removed
    """
    df = df.copy()
    
    # Calculate mean temperature: (min + max) / 2
    df['mean_temp'] = (df['Temp_min_C'] + df['Temp_max_C']) / 2
    
    # Calculate Growing Degree Days (GDD): mean_temp - base_temp
    # GDD represents the accumulated heat units for crop growth
    df['gdd'] = df['mean_temp'] - BASE_TEMP
    
    # Handle edge cases: GDD should be non-negative for most use cases
    # But we keep negative values as they indicate unfavorable conditions
    
    # Remove old temperature columns
    cols_to_drop = [col for col in OLD_TEMP_COLS if col in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    print(f"[Feature Engineering] Created: {NEW_TEMP_COLS}")
    print(f"[Feature Engineering] Removed: {cols_to_drop}")
    print(f"[Feature Engineering] mean_temp range: {df['mean_temp'].min():.2f} - {df['mean_temp'].max():.2f}")
    print(f"[Feature Engineering] GDD range: {df['gdd'].min():.2f} - {df['gdd'].max():.2f}")
    
    return df


def handle_missing_values(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Uses median imputation for numerical features to be robust to outliers.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    feature_cols : list
        List of feature column names to check
    
    Returns
    -------
    pd.DataFrame
        DataFrame with missing values handled
    """
    df = df.copy()
    
    # Check for missing values
    missing_counts = df[feature_cols].isnull().sum()
    total_missing = missing_counts.sum()
    
    if total_missing > 0:
        print(f"[Missing Values] Found {total_missing} missing values:")
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"    - {col}: {count} missing")
        
        # Impute with median for numerical columns
        for col in feature_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"    - {col}: filled with median = {median_val:.2f}")
    else:
        print("[Missing Values] No missing values found")
    
    return df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder columns with Crop first, then features in logical order.
    """
    # Define preferred column order
    preferred_order = [TARGET_COL] + FEATURE_COLS
    
    # Get any remaining columns not in preferred order
    remaining = [col for col in df.columns if col not in preferred_order]
    
    # Final column order
    final_order = [col for col in preferred_order if col in df.columns] + remaining
    
    return df[final_order]


# ─── Data Loading and Processing ──────────────────────────────────────────────

def load_and_engineer_data(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Load dataset, perform feature engineering, and save result.
    
    Parameters
    ----------
    input_path : str
        Path to input CSV file
    output_path : str
        Path to save engineered CSV file
    
    Returns
    -------
    pd.DataFrame
        Engineered dataset
    """
    print("\n" + "=" * 60)
    print("  STEP 1: Feature Engineering")
    print("=" * 60)
    
    # Load dataset
    print(f"\n[Loading] Reading from: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Dataset not found: {input_path}")
    
    df = pd.read_csv(input_path)
    print(f"[Loading] Original shape: {df.shape}")
    print(f"[Loading] Original columns: {df.columns.tolist()}")
    
    # Verify required columns exist
    required_temp_cols = ['Temp_min_C', 'Temp_max_C']
    missing = [c for c in required_temp_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required temperature columns: {missing}")
    
    # Perform feature engineering
    print("\n[Engineering] Creating temperature features...")
    df = create_temperature_features(df)
    
    # Handle missing values
    print("\n[Validation] Checking for missing values...")
    feature_cols_present = [col for col in FEATURE_COLS if col in df.columns]
    df = handle_missing_values(df, feature_cols_present + [TARGET_COL])
    
    # Reorder columns
    df = reorder_columns(df)
    
    # Save engineered dataset
    print(f"\n[Saving] Writing to: {output_path}")
    df.to_csv(output_path, index=False)
    print(f"[Saving] Final shape: {df.shape}")
    print(f"[Saving] Final columns: {df.columns.tolist()}")
    
    return df


# ─── Model Training Pipeline ──────────────────────────────────────────────────

def create_preprocessing_pipeline():
    """
    Create a preprocessing pipeline with imputation and scaling.
    
    Returns
    -------
    Pipeline
        sklearn pipeline for preprocessing
    """
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """
    Train Random Forest classifier.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    
    Returns
    -------
    RandomForestClassifier
        Trained model
    """
    print("\n[Training] Fitting Random Forest...")
    t0 = time.time()
    clf = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"[Training] Complete in {elapsed:.2f}s | Trees: {RF_PARAMS['n_estimators']}")
    return clf


def evaluate_model(clf, X_train, X_test, y_train, y_test, feature_names):
    """
    Evaluate model and generate report.
    
    Returns
    -------
    tuple
        (report_string, test_accuracy)
    """
    print("\n[Evaluation] Computing metrics...")
    
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    y_pred = clf.predict(X_test)
    classes = sorted(clf.classes_)
    
    # Cross-validation
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_all, y_all, cv=cv, scoring='accuracy')
    
    # Build report
    report_lines = [
        "=" * 62,
        "  BhoomiAI — GDD Feature Engineering Model Report",
        "=" * 62,
        "",
        "  FEATURE ENGINEERING SUMMARY",
        "  ----------------------------",
        f"  Base Temperature (GDD): {BASE_TEMP}°C",
        f"  Removed features: {OLD_TEMP_COLS}",
        f"  Added features: {NEW_TEMP_COLS}",
        "",
        f"  Dataset       : {OUTPUT_DATASET}",
        f"  Total samples : {len(X_all):,}",
        f"  Features      : {len(feature_names)}",
        f"  Classes       : {classes}",
        "",
        "  FINAL FEATURE LIST",
        "  -------------------",
    ]
    for i, feat in enumerate(feature_names, 1):
        report_lines.append(f"    {i:2d}. {feat}")
    
    report_lines += [
        "",
        "─" * 62,
        "  MODEL PERFORMANCE",
        "─" * 62,
        f"  Train accuracy        : {train_acc * 100:.2f} %",
        f"  Test  accuracy        : {test_acc * 100:.2f} %",
        f"  5-Fold CV accuracy    : {cv_scores.mean() * 100:.2f} % "
        f"(± {cv_scores.std() * 100:.2f} %)",
        "",
        "─" * 62,
        "  Classification Report (Test Set)",
        "─" * 62,
        classification_report(y_test, y_pred, target_names=classes),
        "─" * 62,
        "  Confusion Matrix",
        "─" * 62,
    ]
    
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    header = "         " + "  ".join(f"{c:>10}" for c in classes)
    report_lines.append(header)
    for i, row_cls in enumerate(classes):
        row = f"{row_cls:>8} " + "  ".join(f"{v:>10}" for v in cm[i])
        report_lines.append(row)
    
    # Feature importance
    report_lines += [
        "",
        "─" * 62,
        "  Feature Importances",
        "─" * 62,
    ]
    importances = clf.feature_importances_
    feat_imp = sorted(zip(feature_names, importances), key=lambda x: -x[1])
    for feat, imp in feat_imp:
        bar = "█" * int(imp * 50)
        report_lines.append(f"  {feat:<24} {imp:.4f}  {bar}")
    
    report_lines += ["", "=" * 62]
    return "\n".join(report_lines), test_acc


def save_model(clf, scaler, accuracy: float, feature_names: list):
    """
    Save trained model with metadata.
    """
    print("\n[Saving] Saving model...")
    os.makedirs('models', exist_ok=True)
    
    payload = {
        'sklearn': clf,
        'scaler': scaler,
        'classes': list(clf.classes_),
        'features': feature_names,
        'accuracy': round(accuracy * 100, 2),
        'n_estimators': RF_PARAMS['n_estimators'],
        'base_temp': BASE_TEMP,
        'feature_engineering': {
            'old_features': OLD_TEMP_COLS,
            'new_features': NEW_TEMP_COLS,
            'formula_mean_temp': '(Temp_min_C + Temp_max_C) / 2',
            'formula_gdd': 'mean_temp - BASE_TEMP',
        }
    }
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(payload, f)
    print(f"[Saving] Model saved → {MODEL_PATH}")


def save_report(report: str):
    """Save evaluation report."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"[Saving] Report saved → {REPORT_PATH}")


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def main():
    """
    Main pipeline: feature engineering, training, and evaluation.
    """
    print("\n" + "=" * 60)
    print("  BhoomiAI - Temperature Feature Engineering Pipeline")
    print("=" * 60)
    
    # Step 1: Feature Engineering
    df = load_and_engineer_data(INPUT_DATASET, OUTPUT_DATASET)
    
    # Step 2: Prepare Data
    print("\n" + "=" * 60)
    print("  STEP 2: Model Training")
    print("=" * 60)
    
    # Verify all feature columns exist
    missing_features = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
    
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    
    print(f"\n[Data] Feature matrix shape: {X.shape}")
    print(f"[Data] Target shape: {y.shape}")
    print(f"[Data] Classes: {sorted(set(y))}")
    
    # Split data
    print("\n[Split] Creating train/test split (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[Split] Train: {len(X_train)} | Test: {len(X_test)}")
    
    # Preprocessing: Scale features
    print("\n[Preprocessing] Scaling features with StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    clf = train_model(X_train_scaled, y_train)
    
    # Step 3: Evaluation
    print("\n" + "=" * 60)
    print("  STEP 3: Evaluation")
    print("=" * 60)
    
    report, test_acc = evaluate_model(
        clf, X_train_scaled, X_test_scaled, y_train, y_test, FEATURE_COLS
    )
    
    # Save outputs
    save_model(clf, scaler, test_acc, FEATURE_COLS)
    save_report(report)
    
    # Final summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"\n  ✅ Feature engineering complete!")
    print(f"  ✅ Replaced 6 temperature features with 2 engineered features")
    print(f"  ✅ Test Accuracy: {test_acc * 100:.2f}%")
    print(f"\n  📁 Output files:")
    print(f"      - Dataset: {OUTPUT_DATASET}")
    print(f"      - Model:   {MODEL_PATH}")
    print(f"      - Report:  {REPORT_PATH}")
    
    print("\n" + report)
    
    return df, clf, scaler


if __name__ == '__main__':
    main()
