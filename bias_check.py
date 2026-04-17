"""
bias_check.py — BhoomiAI quick bias diagnostics
------------------------------------------------
This script checks whether model performance / prediction distribution differs
across groups available in the dataset (Season_code, Agro_Zone).

It produces:
  - overall accuracy
  - accuracy by season and by agro-zone
  - prediction distribution by season and by agro-zone

Usage:
  python bias_check.py
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix


@dataclass(frozen=True)
class ModelBundle:
    name: str
    model_path: str
    dataset_path: str
    target_col: str
    feature_cols: list[str]
    season_col: str
    zone_col: str
    has_scaler: bool


def _load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else float("nan")


def _prediction_fn(bundle: ModelBundle, payload: dict, X: np.ndarray) -> np.ndarray:
    sk_model = payload.get("sklearn")
    if sk_model is None:
        raise RuntimeError(f"{bundle.name}: missing 'sklearn' in {bundle.model_path}")

    if bundle.has_scaler:
        scaler = payload.get("scaler")
        if scaler is None:
            raise RuntimeError(f"{bundle.name}: expected scaler but none found in {bundle.model_path}")
        X = scaler.transform(X)

    return sk_model.predict(X)


def _group_report(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, group_col: str) -> pd.DataFrame:
    rows = []
    for group, gdf in df.groupby(group_col):
        idx = gdf.index.to_numpy()
        yt = y_true[idx]
        yp = y_pred[idx]
        acc = accuracy_score(yt, yp) if len(yt) else float("nan")
        rows.append(
            {
                "group": group,
                "n": int(len(yt)),
                "accuracy": round(acc * 100, 2) if len(yt) else float("nan"),
            }
        )
    out = pd.DataFrame(rows).sort_values(["n", "group"], ascending=[False, True]).reset_index(drop=True)
    return out


def _prediction_distribution(df: pd.DataFrame, y_pred: np.ndarray, group_col: str) -> pd.DataFrame:
    # percent distribution per group
    tmp = df[[group_col]].copy()
    tmp["pred"] = y_pred
    counts = tmp.groupby([group_col, "pred"]).size().rename("count").reset_index()
    totals = tmp.groupby(group_col).size().rename("total").reset_index()
    merged = counts.merge(totals, on=group_col, how="left")
    merged["pct"] = (merged["count"] / merged["total"] * 100.0).round(2)
    merged = merged.sort_values([group_col, "pct"], ascending=[True, False]).reset_index(drop=True)
    return merged


def run_bundle(bundle: ModelBundle) -> None:
    print("\n" + "=" * 78)
    print(f"BIAS CHECK: {bundle.name}")
    print("=" * 78)

    if not os.path.exists(bundle.dataset_path):
        raise FileNotFoundError(f"Dataset not found: {bundle.dataset_path}")
    if not os.path.exists(bundle.model_path):
        raise FileNotFoundError(f"Model not found: {bundle.model_path}")

    df = pd.read_csv(bundle.dataset_path)
    needed = set(bundle.feature_cols + [bundle.target_col, bundle.season_col, bundle.zone_col])
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{bundle.name}: dataset missing columns: {missing}")

    payload = _load_pickle(bundle.model_path)
    X = df[bundle.feature_cols].to_numpy(dtype=float)
    y_true = df[bundle.target_col].to_numpy()

    y_pred = _prediction_fn(bundle, payload, X)

    overall_acc = accuracy_score(y_true, y_pred)
    print(f"Overall accuracy: {overall_acc * 100:.2f}%  (n={len(df)})")

    # Basic "bias smell test": max gap in group accuracies
    season_rep = _group_report(df, y_true, y_pred, bundle.season_col)
    zone_rep = _group_report(df, y_true, y_pred, bundle.zone_col)

    def _max_gap(rep: pd.DataFrame) -> float:
        accs = rep["accuracy"].dropna().to_numpy(dtype=float)
        return float(np.nanmax(accs) - np.nanmin(accs)) if len(accs) else float("nan")

    print(f"Accuracy gap by {bundle.season_col}: { _max_gap(season_rep):.2f} percentage points")
    print(f"Accuracy gap by {bundle.zone_col}  : { _max_gap(zone_rep):.2f} percentage points")

    print("\nAccuracy by season")
    print(season_rep.to_string(index=False))

    print("\nAccuracy by agro-zone")
    print(zone_rep.to_string(index=False))

    print("\nPrediction distribution by season (top rows)")
    dist_season = _prediction_distribution(df, y_pred, bundle.season_col)
    print(dist_season.head(20).to_string(index=False))

    print("\nPrediction distribution by agro-zone (top rows)")
    dist_zone = _prediction_distribution(df, y_pred, bundle.zone_col)
    print(dist_zone.head(30).to_string(index=False))

    # Confusion matrix summary
    labels = sorted(pd.unique(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_acc = _safe_div(np.trace(cm), np.sum(cm)) * 100.0
    print("\nConfusion matrix (rows=actual, cols=pred)")
    header = " " * 12 + " ".join(f"{c:>10}" for c in labels)
    print(header)
    for i, row_cls in enumerate(labels):
        row = " ".join(f"{v:>10}" for v in cm[i])
        print(f"{row_cls:>10}  {row}")
    print(f"\nCM-derived accuracy: {cm_acc:.2f}%")


def main() -> None:
    # Original 18-feature model
    rf_bundle = ModelBundle(
        name="RandomForest (18 features) — rf_model.pkl",
        model_path=os.path.join("models", "rf_model.pkl"),
        dataset_path=os.path.join("dataset", "crop_train_ml.csv"),
        target_col="Crop",
        feature_cols=[
            "Temp_min_C",
            "Temp_max_C",
            "Rain_min_cm",
            "Rain_max_cm",
            "Sow_temp_min",
            "Sow_temp_max",
            "Harvest_temp_min",
            "Harvest_temp_max",
            "Sand_pct",
            "Clay_pct",
            "Silt_pct",
            "Nitrogen_N_kg_ha",
            "Phosphorus_P_kg_ha",
            "Potassium_K_kg_ha",
            "Humidity_pct",
            "pH",
            "Season_code",
            "Agro_Zone",
        ],
        season_col="Season_code",
        zone_col="Agro_Zone",
        has_scaler=False,
    )

    # GDD engineered model (used by the Flask app)
    gdd_bundle = ModelBundle(
        name="RandomForest (GDD engineered) — rf_model_gdd.pkl",
        model_path=os.path.join("models", "rf_model_gdd.pkl"),
        dataset_path=os.path.join("dataset", "crop_train_gdd.csv"),
        target_col="Crop",
        feature_cols=[
            "mean_temp",
            "gdd",
            "Rain_min_cm",
            "Rain_max_cm",
            "Sand_pct",
            "Clay_pct",
            "Silt_pct",
            "Nitrogen_N_kg_ha",
            "Phosphorus_P_kg_ha",
            "Potassium_K_kg_ha",
            "Humidity_pct",
            "pH",
            "Season_code",
            "Agro_Zone",
        ],
        season_col="Season_code",
        zone_col="Agro_Zone",
        has_scaler=True,
    )

    for bundle in [rf_bundle, gdd_bundle]:
        try:
            run_bundle(bundle)
        except Exception as e:
            print("\n" + "-" * 78)
            print(f"Skipped {bundle.name}: {e}")
            print("-" * 78)


if __name__ == "__main__":
    main()

