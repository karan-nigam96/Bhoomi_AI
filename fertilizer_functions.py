"""
fertilizer_functions.py — BhoomiAI Fertilizer Recommendation Module

Two models:
  - fertilizer_npk_model.pkl  : RandomForestRegressor (N, P, K kg/ha)
  - fertilizer_flag_model.pkl : MultiOutputClassifier (Lime/Gypsum/Zinc flags)

Supporting files:
  - fertilizer_scaler.pkl     : StandardScaler (fitted on training data)
  - fertilizer_encoders.pkl   : LabelEncoders for Crop, Variety, Organic, PrevCrop
  - fertilizer_backend_config.json : stage schedule fractions + natural alternatives
"""

import os
import json
import pickle
import numpy as np
import pandas as pd

# ── Model file paths ──────────────────────────────────────────────────────────
_BASE = os.path.join(os.path.dirname(__file__), 'models', 'fertilizer_model')

_NPK_MODEL_PATH = os.path.join(_BASE, 'fertilizer_npk_model (1).pkl')
_FLAG_MODEL_PATH = os.path.join(_BASE, 'fertilizer_flag_model (1).pkl')
_SCALER_PATH = os.path.join(_BASE, 'fertilizer_scaler.pkl')
_ENCODERS_PATH = os.path.join(_BASE, 'fertilizer_encoders (1).pkl')
_CONFIG_PATH = os.path.join(_BASE, 'fertilizer_backend_config.json')

# Crop label encoding (must match training)
CROP_ENC = {'Maize': 0, 'Rice': 1, 'Sugarcane': 2, 'Wheat': 3}

# Variety encoding
VARIETY_ENC = {'HYV': 0, 'Traditional': 1}

# Organic input encoding
ORGANIC_ENC = {'FYM_10t': 0, 'FYM_5t': 1, 'GreenManure': 2, 'Vermicompost': 3, 'None': 4}

# Previous crop encoding
PREV_CROP_ENC = {'Cereal': 0, 'Fallow': 1, 'Legume': 2, 'Vegetable': 3}

# Feature order (must match scaler fit order)
FEATURE_ORDER = [
    'Crop_enc', 'Zone_ID', 'Season',
    'Nitrogen_soil', 'Phosphorus_soil', 'Potassium_soil',
    'pH', 'Irrigation',
    'Variety_enc', 'Organic_enc', 'PrevCrop_enc', 'Farm_Size_ha',
]


def load_fertilizer_models():
    """
    Load all 4 model components + JSON config at server startup.
    Returns a dict or None on failure.
    """
    try:
        with open(_NPK_MODEL_PATH, 'rb') as f:
            npk_model = pickle.load(f)
        with open(_FLAG_MODEL_PATH, 'rb') as f:
            flag_model = pickle.load(f)
        with open(_SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        with open(_CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Encoders file is optional — we use hardcoded dicts as fallback
        encoders = None
        if os.path.exists(_ENCODERS_PATH):
            try:
                with open(_ENCODERS_PATH, 'rb') as f:
                    encoders = pickle.load(f)
            except Exception:
                pass

        print(f"[BhoomiAI] Fertilizer models loaded: NPK regressor + flag classifier")
        return {
            'npk_model': npk_model,
            'flag_model': flag_model,
            'scaler': scaler,
            'encoders': encoders,
            'config': config,
        }
    except FileNotFoundError as e:
        print(f"[BhoomiAI] WARNING: Fertilizer model file not found: {e}")
        return None
    except Exception as e:
        print(f"[BhoomiAI] WARNING: Could not load fertilizer models: {e}")
        return None


def _encode_categorical(value, enc_dict, default=0):
    """Safe dictionary-based label encoding with fallback."""
    return enc_dict.get(str(value), default)


def predict_fertilizer(models, crop, zone_id, season,
                       n_soil, p_soil, k_soil, ph, irrigation,
                       variety, organic, prev_crop, farm_size):
    """
    Run the fertilizer recommendation pipeline.

    Parameters
    ----------
    models      : dict returned by load_fertilizer_models()
    crop        : str — 'Rice'|'Wheat'|'Maize'|'Sugarcane'
    zone_id     : int — 1-15 (ICAR zone from GPS/manual)
    season      : int — 0 Kharif | 1 Rabi | 2 Zaid
    n_soil      : float — soil Nitrogen kg/ha
    p_soil      : float — soil Phosphorus kg/ha
    k_soil      : float — soil Potassium kg/ha
    ph          : float — soil pH 4.5-9.2
    irrigation  : int   — 1 = canal/tubewell, 0 = rainfed
    variety     : str   — 'HYV' or 'Traditional'
    organic     : str   — 'FYM_10t'|'FYM_5t'|'GreenManure'|'Vermicompost'|'None'
    prev_crop   : str   — 'Cereal'|'Fallow'|'Legume'|'Vegetable'
    farm_size   : float — hectares

    Returns
    -------
    dict with keys: npk_per_ha, npk_total, special_treatments, stage_schedule
    """
    npk_model  = models['npk_model']
    flag_model = models['flag_model']
    scaler     = models['scaler']
    config     = models['config']

    # ── Encode categoricals ───────────────────────────────────────────────────
    crop_enc    = _encode_categorical(crop,      CROP_ENC,     default=3)  # Wheat
    variety_enc = _encode_categorical(variety,   VARIETY_ENC,  default=0)  # HYV
    organic_enc = _encode_categorical(organic,   ORGANIC_ENC,  default=4)  # None
    prev_enc    = _encode_categorical(prev_crop, PREV_CROP_ENC, default=0) # Cereal

    # ── Build feature row ─────────────────────────────────────────────────────
    sample = {
        'Crop_enc':        crop_enc,
        'Zone_ID':         zone_id,
        'Season':          season,
        'Nitrogen_soil':   n_soil,
        'Phosphorus_soil': p_soil,
        'Potassium_soil':  k_soil,
        'pH':              ph,
        'Irrigation':      irrigation,
        'Variety_enc':     variety_enc,
        'Organic_enc':     organic_enc,
        'PrevCrop_enc':    prev_enc,
        'Farm_Size_ha':    farm_size,
    }

    X_df = pd.DataFrame([sample])[FEATURE_ORDER]

    # ── Scale ────────────────────────────────────────────────────────────────
    try:
        X_scaled = scaler.transform(X_df)
    except Exception:
        X_scaled = X_df.values  # fallback: unscaled (rare)

    # ── Predict NPK (regression) ─────────────────────────────────────────────
    npk_pred = npk_model.predict(X_scaled)[0]
    N_per_ha = max(0, round(float(npk_pred[0])))
    P_per_ha = max(0, round(float(npk_pred[1])))
    K_per_ha = max(0, round(float(npk_pred[2])))

    # ── Predict flags (classification) ───────────────────────────────────────
    flag_pred = flag_model.predict(X_scaled)[0]
    needs_lime   = bool(flag_pred[0])
    needs_gypsum = bool(flag_pred[1])
    needs_zinc   = bool(flag_pred[2])

    # ── Build stage schedule ─────────────────────────────────────────────────
    schedule_config = config.get('stage_schedule', {})
    crop_schedule   = schedule_config.get(crop, [])

    schedule = []
    for i, stage in enumerate(crop_schedule):
        N_stage = max(0, round(N_per_ha * stage.get('N_fraction', 0)))
        P_stage = max(0, round(P_per_ha * stage.get('P_fraction', 0)))
        K_stage = max(0, round(K_per_ha * stage.get('K_fraction', 0)))

        day_val = stage.get('day_from_sowing', 0)
        timing  = f"Day {day_val}" if day_val != 0 else "Day 0 — at sowing"

        chem = stage.get('chemical_fertilizer', '')
        natural = stage.get('natural_alternative', '')

        # Append special treatment notes to basal stage
        special_notes = []
        if i == 0:
            if needs_zinc:
                special_notes.append("+ ZnSO₄ 25 kg/ha at basal stage")
            if needs_lime:
                special_notes.append("⚠ Apply Agricultural Lime 2-4 t/ha 3 weeks before sowing")
            if needs_gypsum:
                special_notes.append("⚠ Apply Gypsum 500 kg/ha before fertilizer application")

        schedule.append({
            'stage':       stage.get('stage', f'Stage {i+1}'),
            'timing':      timing,
            'N_kg_ha':     N_stage,
            'P_kg_ha':     P_stage,
            'K_kg_ha':     K_stage,
            'N_total':     round(N_stage * farm_size, 1),
            'P_total':     round(P_stage * farm_size, 1),
            'K_total':     round(K_stage * farm_size, 1),
            'chemical':    chem,
            'natural':     natural,
            'special_notes': special_notes,
        })

    return {
        'npk_per_ha': {'N': N_per_ha, 'P': P_per_ha, 'K': K_per_ha},
        'npk_total': {
            'N': round(N_per_ha * farm_size, 1),
            'P': round(P_per_ha * farm_size, 1),
            'K': round(K_per_ha * farm_size, 1),
        },
        'special_treatments': {
            'Needs_Lime':   needs_lime,
            'Needs_Gypsum': needs_gypsum,
            'Needs_Zinc':   needs_zinc,
        },
        'stage_schedule': schedule,
        'model_metrics': config.get('model_metrics', {}),
    }
