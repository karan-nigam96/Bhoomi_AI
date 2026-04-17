"""
calculate_accuracy.py — BhoomiAI Model Trainer & Evaluator
===========================================================
Trains a Random Forest Classifier on the ICAR crop dataset,
evaluates accuracy on a held-out test split, saves the model,
and writes a results report.

Usage:
    python calculate_accuracy.py

Outputs:
    models/rf_model.pkl     — serialised sklearn model
    results/accuracy_report.txt  — full evaluation report
"""

import os
import pickle
import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder

# ─── Paths ────────────────────────────────────────────────────────────────────
DATASET_PATH = os.path.join('dataset', 'crop_train_ml.csv')  # Enhanced dataset with 18 features
MODEL_PATH   = os.path.join('models', 'rf_model.pkl')
RESULTS_DIR  = 'results'
REPORT_PATH  = os.path.join(RESULTS_DIR, 'accuracy_report.txt')

# ─── Model Hyper-parameters ───────────────────────────────────────────────────
RF_PARAMS = dict(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
)

FEATURE_COLS = [
    'Temp_min_C', 'Temp_max_C',
    'Rain_min_cm', 'Rain_max_cm',
    'Sow_temp_min', 'Sow_temp_max',
    'Harvest_temp_min', 'Harvest_temp_max',
    'Sand_pct', 'Clay_pct', 'Silt_pct',
    'Nitrogen_N_kg_ha', 'Phosphorus_P_kg_ha', 'Potassium_K_kg_ha',
    'Humidity_pct', 'pH',
    'Season_code', 'Agro_Zone',  # New features
]
TARGET_COL = 'Crop'


def load_data(path: str) -> pd.DataFrame:
    """Load and validate the dataset CSV."""
    print(f"[1/6] Loading dataset from '{path}' …")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}\n"
                                f"Place crop_train.csv inside the 'dataset/' folder.")
    df = pd.read_csv(path)
    print(f"      Rows: {len(df):,}  |  Columns: {list(df.columns)}")
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    print(f"      After dropping NaN rows: {len(df):,}")
    print(f"      Classes: {sorted(df[TARGET_COL].unique())}")
    return df


def split_data(df: pd.DataFrame):
    """Split into train/test with stratification."""
    print("[2/6] Splitting data (80 % train / 20 % test, stratified) …")
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train) -> RandomForestClassifier:
    """Train the Random Forest Classifier."""
    print("[3/6] Training Random Forest …")
    t0 = time.time()
    clf = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"      Training complete in {elapsed:.2f}s  |  Trees: {RF_PARAMS['n_estimators']}")
    return clf


def evaluate_model(clf, X_train, X_test, y_train, y_test, classes):
    """Compute accuracy metrics and return a formatted report string."""
    print("[4/6] Evaluating model …")

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc  = accuracy_score(y_test,  clf.predict(X_test))
    y_pred    = clf.predict(X_test)

    # 5-fold cross-validation on full data
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_all, y_all, cv=cv, scoring='accuracy')

    report_lines = [
        "=" * 62,
        "  BhoomiAI — Random Forest Accuracy Report",
        "=" * 62,
        "",
        f"  Dataset       : {DATASET_PATH}",
        f"  Total samples : {len(X_all):,}",
        f"  Features      : {len(FEATURE_COLS)}",
        f"  Classes       : {sorted(classes)}",
        "",
        f"  Train accuracy        : {train_acc * 100:.2f} %",
        f"  Test  accuracy        : {test_acc  * 100:.2f} %",
        f"  5-Fold CV accuracy    : {cv_scores.mean() * 100:.2f} % "
        f"(± {cv_scores.std() * 100:.2f} %)",
        "",
        "─" * 62,
        "  Classification Report (Test Set)",
        "─" * 62,
        classification_report(y_test, y_pred, target_names=sorted(classes)),
        "─" * 62,
        "  Confusion Matrix (rows = actual, cols = predicted)",
        "─" * 62,
    ]

    cm     = confusion_matrix(y_test, y_pred, labels=sorted(classes))
    header = "         " + "  ".join(f"{c:>10}" for c in sorted(classes))
    report_lines.append(header)
    for i, row_cls in enumerate(sorted(classes)):
        row = f"{row_cls:>8} " + "  ".join(f"{v:>10}" for v in cm[i])
        report_lines.append(row)

    report_lines += [
        "",
        "─" * 62,
        "  Feature Importances",
        "─" * 62,
    ]
    importances = clf.feature_importances_
    feat_imp = sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1])
    for feat, imp in feat_imp:
        bar = "█" * int(imp * 50)
        report_lines.append(f"  {feat:<28} {imp:.4f}  {bar}")

    report_lines += ["", "=" * 62]
    return "\n".join(report_lines), test_acc


def save_model(clf, accuracy: float):
    """Pickle the model along with metadata."""
    print("[5/6] Saving model …")
    os.makedirs('models', exist_ok=True)
    payload = {
        'sklearn': clf,
        'classes': list(clf.classes_),
        'features': FEATURE_COLS,
        'accuracy': round(accuracy * 100, 2),
        'n_estimators': RF_PARAMS['n_estimators'],
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(payload, f)
    print(f"      Saved → {MODEL_PATH}")


def save_report(report: str):
    """Write the accuracy report to the results directory."""
    print("[6/6] Saving report …")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"      Saved → {REPORT_PATH}")


def main():
    print("\n🌿  BhoomiAI — Model Training Pipeline\n")
    df = load_data(DATASET_PATH)
    X_train, X_test, y_train, y_test = split_data(df)
    clf = train_model(X_train, y_train)
    classes = clf.classes_
    report, test_acc = evaluate_model(clf, X_train, X_test, y_train, y_test, classes)
    save_model(clf, test_acc)
    save_report(report)
    print(f"\n✅  Done!  Test Accuracy: {test_acc * 100:.2f} %")
    print(report)


if __name__ == '__main__':
    main()
