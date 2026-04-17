"""
functions.py — BhoomiAI helper functions
Handles model loading, prediction, and crop data utilities.
Supports original GDD model (v1) and improved v2 model.
v2 uses One-Hot encoding for Season/Zone, rain_avg, and
three engineered features.  Includes region-based soil defaults.
"""

import os
import pickle
import json
import numpy as np

ALLOWED_EXTENSIONS = {'csv', 'json', 'txt'}

# Base temperature for GDD calculation (°C)
BASE_TEMP = 10.0

# ─── Region-Based Default Soil Values ─────────────────────────────────────────
# Based on ICAR soil survey data for each agro-climatic zone
# Values represent typical soil composition (sand%, silt%, clay%)
# Used when user doesn't provide soil values

REGION_SOIL_DEFAULTS = {
    # Zone ID: {"sand": %, "silt": %, "clay": %}
    0: {"sand": 45, "silt": 30, "clay": 25, "name": "Western Himalayan Region"},       # Mountain soils
    1: {"sand": 40, "silt": 35, "clay": 25, "name": "Eastern Himalayan Region"},       # Humid hill soils
    2: {"sand": 25, "silt": 35, "clay": 40, "name": "Lower Gangetic Plains Region"},   # Heavy alluvial
    3: {"sand": 30, "silt": 40, "clay": 30, "name": "Middle Gangetic Plains Region"},  # Alluvial loam
    4: {"sand": 35, "silt": 40, "clay": 25, "name": "Upper Gangetic Plains Region"},   # Sandy loam alluvial
    5: {"sand": 40, "silt": 35, "clay": 25, "name": "Trans-Gangetic Plains Region"},   # Sandy alluvial
    6: {"sand": 35, "silt": 25, "clay": 40, "name": "Eastern Plateau and Hills"},      # Red laterite
    7: {"sand": 25, "silt": 30, "clay": 45, "name": "Central Plateau and Hills"},      # Black cotton soil
    8: {"sand": 20, "silt": 30, "clay": 50, "name": "Western Plateau and Hills"},      # Deep black soil
    9: {"sand": 35, "silt": 30, "clay": 35, "name": "Southern Plateau and Hills"},     # Red and black mix
    10: {"sand": 30, "silt": 40, "clay": 30, "name": "East Coast Plains and Hills"},   # Coastal alluvial
    11: {"sand": 40, "silt": 30, "clay": 30, "name": "West Coast Plains and Ghats"},   # Laterite
    12: {"sand": 35, "silt": 35, "clay": 30, "name": "Gujarat Plains and Hills"},      # Mixed soils
    13: {"sand": 55, "silt": 25, "clay": 20, "name": "Western Dry Region"},            # Desert sandy
    14: {"sand": 50, "silt": 30, "clay": 20, "name": "Islands Region"},                # Coastal sandy
}

# Default fallback soil values if region is unknown
DEFAULT_SOIL = {"sand": 33, "silt": 34, "clay": 33}


# ─── Soil Preprocessing Functions ─────────────────────────────────────────────

def get_soil_defaults_for_region(zone_id: int) -> dict:
    """
    Get default soil values for a given agro-climatic zone.
    
    Parameters
    ----------
    zone_id : int
        Agro-climatic zone ID (0-14)
    
    Returns
    -------
    dict : {"sand": float, "silt": float, "clay": float}
    """
    if zone_id in REGION_SOIL_DEFAULTS:
        defaults = REGION_SOIL_DEFAULTS[zone_id]
        return {"sand": defaults["sand"], "silt": defaults["silt"], "clay": defaults["clay"]}
    return DEFAULT_SOIL.copy()


def is_soil_value_missing(value) -> bool:
    """
    Check if a soil value is missing/empty/invalid.
    
    Returns True if value should be treated as missing.
    """
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == '':
        return True
    try:
        val = float(value)
        # Treat 0 or negative as potentially missing (user didn't fill)
        return val <= 0
    except (ValueError, TypeError):
        return True


def validate_soil_percentages(sand: float, silt: float, clay: float, tolerance: float = 15.0) -> tuple:
    """
    Validate and optionally normalize soil percentages to sum to 100%.
    
    Parameters
    ----------
    sand, silt, clay : float
        Soil composition percentages
    tolerance : float
        Acceptable deviation from 100% before rejection (default 15%)
    
    Returns
    -------
    tuple : (sand, silt, clay, is_valid, message)
    """
    total = sand + silt + clay
    
    if total == 0:
        return sand, silt, clay, False, "All soil values are zero"
    
    if abs(total - 100) <= tolerance:
        # Within tolerance - normalize to exactly 100%
        factor = 100 / total
        return (
            round(sand * factor, 2),
            round(silt * factor, 2),
            round(clay * factor, 2),
            True,
            f"Normalized from {total:.1f}% to 100%"
        )
    else:
        # Outside tolerance - reject or warn
        return sand, silt, clay, False, f"Soil percentages sum to {total:.1f}%, expected ~100%"


def preprocess_soil_inputs(
    sand, silt, clay,
    zone_id: int = None,
    auto_fill: bool = True,
    normalize: bool = True
) -> dict:
    """
    Preprocess soil inputs with region-based defaults and validation.
    
    This is the main preprocessing function that:
    1. Detects missing soil values
    2. Auto-fills from region defaults if enabled
    3. Validates and normalizes percentages
    
    Parameters
    ----------
    sand, silt, clay : any
        Soil values (can be None, empty string, or numeric)
    zone_id : int
        Agro-climatic zone ID for defaults lookup
    auto_fill : bool
        If True, fill missing values from region defaults
    normalize : bool
        If True, normalize percentages to sum to 100%
    
    Returns
    -------
    dict : {
        "sand": float,
        "silt": float, 
        "clay": float,
        "source": "user" | "default" | "mixed",
        "normalized": bool,
        "message": str
    }
    """
    # Check which values are missing
    sand_missing = is_soil_value_missing(sand)
    silt_missing = is_soil_value_missing(silt)
    clay_missing = is_soil_value_missing(clay)
    
    all_missing = sand_missing and silt_missing and clay_missing
    any_missing = sand_missing or silt_missing or clay_missing
    
    # Determine source and get values
    if all_missing and auto_fill:
        # All missing - use region defaults
        defaults = get_soil_defaults_for_region(zone_id)
        result = {
            "sand": defaults["sand"],
            "silt": defaults["silt"],
            "clay": defaults["clay"],
            "source": "default",
            "normalized": True,
            "message": f"Using defaults for zone {zone_id}"
        }
    elif any_missing and auto_fill:
        # Partial missing - fill gaps with defaults
        defaults = get_soil_defaults_for_region(zone_id)
        result = {
            "sand": float(sand) if not sand_missing else defaults["sand"],
            "silt": float(silt) if not silt_missing else defaults["silt"],
            "clay": float(clay) if not clay_missing else defaults["clay"],
            "source": "mixed",
            "normalized": False,
            "message": "Partial user input, gaps filled with defaults"
        }
    else:
        # All provided by user
        result = {
            "sand": float(sand) if not sand_missing else 0,
            "silt": float(silt) if not silt_missing else 0,
            "clay": float(clay) if not clay_missing else 0,
            "source": "user",
            "normalized": False,
            "message": "User-provided values"
        }
    
    # Validate and normalize
    if normalize and result["source"] != "default":
        sand_val, silt_val, clay_val, is_valid, msg = validate_soil_percentages(
            result["sand"], result["silt"], result["clay"]
        )
        if is_valid:
            result["sand"] = sand_val
            result["silt"] = silt_val
            result["clay"] = clay_val
            result["normalized"] = True
            result["message"] = msg
        else:
            result["message"] = msg
    
    return result

CROP_INFO = {
    'Wheat': {
        'icon': '🌾',
        'scientific': 'Triticum aestivum',
        'season': 'Rabi (Nov–Apr)',
        'temp': '10–26 °C',
        'rainfall': '300–750 mm',
        'ph': '6.0–7.5',
        'soil': 'Loamy / Sandy Loam',
        'npk': '120-60-40',
        'yield': '4–6 t/ha',
        'color': '#f4a261',
        'varieties': [
            'HD-3086 (Pusa Wheat) — 6.5 t/ha',
            'PBW-343 — Punjab favourite, 5.8 t/ha',
            'GW-496 — Gujarat, drought tolerant',
            'DBW-187 — Rust resistant, 5.5 t/ha',
        ],
        'description': 'Primary Rabi staple crop of the Indo-Gangetic plains.',
    },
    'Rice': {
        'icon': '🌾',
        'scientific': 'Oryza sativa',
        'season': 'Kharif (Jun–Nov)',
        'temp': '20–38 °C',
        'rainfall': '1000–2000 mm',
        'ph': '5.0–6.5',
        'soil': 'Clay / Heavy Loam',
        'npk': '120-60-60',
        'yield': '5–8 t/ha',
        'color': '#1976d2',
        'varieties': [
            'IR-64 — Widely adopted, 6.5 t/ha',
            'Pusa Basmati-1121 — Premium, 5.5 t/ha',
            'MTU-1010 — Andhra favourite, 7 t/ha',
            'Swarna (MTU-7029) — Flood tolerant',
        ],
        'description': 'Primary Kharif staple, requires high water availability.',
    },
    'Maize': {
        'icon': '🌽',
        'scientific': 'Zea mays',
        'season': 'Kharif & Rabi',
        'temp': '16–34 °C',
        'rainfall': '500–1200 mm',
        'ph': '5.5–7.0',
        'soil': 'Sandy Loam / Loam',
        'npk': '120-60-40',
        'yield': '5–10 t/ha',
        'color': '#ef6c00',
        'varieties': [
            'Pioneer 30V92 — Hybrid, 9.5 t/ha',
            'HQPM-1 — QPM protein-rich variety',
            'Vivek-QPM-9 — High altitude suitable',
            'DKC-9144 — Drought tolerant hybrid',
        ],
        'description': 'Versatile dual-season crop used for food, fodder and starch.',
    },
    'Sugarcane': {
        'icon': '🎋',
        'scientific': 'Saccharum officinarum',
        'season': 'Annual (Feb–Jan)',
        'temp': '18–35 °C',
        'rainfall': '750–1500 mm',
        'ph': '6.0–7.5',
        'soil': 'Deep Loam / Clay Loam',
        'npk': '250-115-115',
        'yield': '80–120 t/ha',
        'color': '#558b2f',
        'varieties': [
            'Co-0238 — UP favourite, 90 t/ha',
            'CoJ-64 — Punjab, early maturing',
            'CoM-0265 — Maharashtra, high sugar',
            'Co-86032 — Tamil Nadu, 110 t/ha',
        ],
        'description': 'Annual cash crop; main source of sugar and ethanol in India.',
    },
}

FEATURE_NAMES = [
    'Temp_min_C', 'Temp_max_C',
    'Rain_min_cm', 'Rain_max_cm',
    'Sow_temp_min', 'Sow_temp_max',
    'Harvest_temp_min', 'Harvest_temp_max',
    'Sand_pct', 'Clay_pct', 'Silt_pct',
    'Nitrogen_N_kg_ha', 'Phosphorus_P_kg_ha', 'Potassium_K_kg_ha',
    'Humidity_pct', 'pH',
    'Season_code', 'Agro_Zone',
]

# GDD model feature names (simplified with engineered features)
FEATURE_NAMES_GDD = [
    'mean_temp', 'gdd',
    'Rain_min_cm', 'Rain_max_cm',
    'Sand_pct', 'Clay_pct', 'Silt_pct',
    'Nitrogen_N_kg_ha', 'Phosphorus_P_kg_ha', 'Potassium_K_kg_ha',
    'Humidity_pct', 'pH',
    'Season_code', 'Agro_Zone',
]

SEASON_MAP = {'Rabi': 0, 'Kharif': 1}

CLASS_NAMES = ['Maize', 'Rice', 'Sugarcane', 'Wheat']


def load_model(model_path: str = 'models/rf_model.pkl') -> dict:
    """
    Load the trained Random Forest model from disk.
    Falls back to a default stub if the model file doesn't exist yet.
    """
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"[BhoomiAI] Model loaded from {model_path}")
        return model
    else:
        print(f"[BhoomiAI] Model not found at {model_path}. Run calculate_accuracy.py to train.")
        return {'sklearn': None, 'classes': CLASS_NAMES, 'accuracy': None}


def load_model_gdd(model_path: str = 'models/rf_model_gdd.pkl') -> dict:
    """
    Load the GDD-enhanced Random Forest model from disk.
    This model uses mean_temp and gdd instead of 6 temperature features.
    """
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"[BhoomiAI] GDD model loaded from {model_path}")
        return model
    else:
        print(f"[BhoomiAI] GDD model not found at {model_path}. Run feature_engineering.py to train.")
        return {'sklearn': None, 'scaler': None, 'classes': CLASS_NAMES, 'accuracy': None}


def load_zone_data(zone_path: str = 'dataset/agro_zones.json') -> dict:
    """Load agro-climatic zone mapping data."""
    if os.path.exists(zone_path):
        with open(zone_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def get_zone_from_district(state: str, district: str, zone_data: dict = None) -> dict:
    """
    Get agro-climatic zone information from state and district.
    
    Returns dict with zone_id, zone_name, or None if not found.
    """
    if zone_data is None:
        zone_data = load_zone_data()
    
    if zone_data and state in zone_data['state_district_mapping']:
        state_info = zone_data['state_district_mapping'][state]
        if district in state_info['districts']:
            return {
                'zone_id': state_info['zone_id'],
                'zone_name': state_info['zone_name']
            }
    return None


def predict_crop(model: dict, features: list, season: str = None, zone_id: int = None) -> dict:
    """
    Run inference and return prediction dict with individual crop confidence scores.

    Parameters
    ----------
    model    : dict returned by load_model()
    features : list of 18 floats in FEATURE_NAMES order (including season_code and zone)
             OR list of 16 floats (old format, season/zone will be appended)
    season   : 'Rabi' or 'Kharif' (used if features has only 16 elements)
    zone_id  : 0-14 agro-climatic zone ID (used if features has only 16 elements)

    Returns
    -------
    dict with keys: crop, confidence, votes (individual scores), features_used, season, zone
    """
    # Handle backward compatibility: if 16 features provided, append season and zone
    if len(features) == 16:
        season_code = SEASON_MAP.get(season, 1) if season else 1  # Default to Kharif
        zone_val = zone_id if zone_id is not None else 4  # Default to Upper Gangetic
        features = features + [season_code, zone_val]
    elif len(features) == 18:
        season_code = int(features[16])
        zone_val = int(features[17])
        season = 'Rabi' if season_code == 0 else 'Kharif'
    else:
        raise ValueError(f"Expected 16 or 18 features, got {len(features)}")

    arr = np.array(features).reshape(1, -1)

    sk_model = model.get('sklearn') if isinstance(model, dict) else model

    if sk_model is not None:
        # sklearn RandomForestClassifier
        proba = sk_model.predict_proba(arr)[0]
        classes = sk_model.classes_
        best_idx = int(np.argmax(proba))
        best_crop = classes[best_idx]
        
        # Convert probabilities to individual confidence scores
        votes = {}
        for cls, p in zip(classes, proba):
            individual_score = min(100, max(0, (p * 200)))
            votes[cls] = round(individual_score, 1)
        
        confidence = votes[best_crop]
    else:
        # Fallback: rule-based heuristic (no trained model yet)
        best_crop, confidence, votes = _heuristic_predict(features)

    # Load zone data for display
    zone_data = load_zone_data()
    zone_name = None
    if zone_data and 0 <= zone_val < len(zone_data['zones']):
        zone_name = zone_data['zones'][zone_val]['name']

    return {
        'crop': best_crop,
        'confidence': round(confidence, 1),
        'votes': votes,
        'features_used': dict(zip(FEATURE_NAMES, features)),
        'season': season,
        'zone_id': zone_val,
        'zone_name': zone_name,
    }


def predict_crop_gdd(model: dict, features: list, season: str = None, zone_id: int = None) -> dict:
    """
    Run inference using the GDD-enhanced model.
    
    Parameters
    ----------
    model    : dict returned by load_model_gdd()
    features : list of 14 floats in FEATURE_NAMES_GDD order
               [mean_temp, gdd, Rain_min_cm, Rain_max_cm, Sand_pct, Clay_pct, Silt_pct,
                Nitrogen_N_kg_ha, Phosphorus_P_kg_ha, Potassium_K_kg_ha,
                Humidity_pct, pH, Season_code, Agro_Zone]
    season   : 'Rabi' or 'Kharif'
    zone_id  : 0-14 agro-climatic zone ID

    Returns
    -------
    dict with keys: crop, confidence, votes, features_used, season, zone
    """
    if len(features) != 14:
        raise ValueError(f"Expected 14 features for GDD model, got {len(features)}")
    
    arr = np.array(features).reshape(1, -1)
    
    sk_model = model.get('sklearn') if isinstance(model, dict) else None
    scaler = model.get('scaler') if isinstance(model, dict) else None
    
    # Derive season and zone from features if not provided
    season_code = int(features[12])
    zone_val = int(features[13])
    if season is None:
        season = 'Rabi' if season_code == 0 else 'Kharif'
    if zone_id is None:
        zone_id = zone_val
    
    if sk_model is not None:
        # Scale features if scaler is available
        if scaler is not None:
            arr_scaled = scaler.transform(arr)
        else:
            arr_scaled = arr
        
        # sklearn RandomForestClassifier prediction
        proba = sk_model.predict_proba(arr_scaled)[0]
        classes = sk_model.classes_
        best_idx = int(np.argmax(proba))
        best_crop = classes[best_idx]
        
        # Convert probabilities to individual confidence scores
        votes = {}
        for cls, p in zip(classes, proba):
            individual_score = min(100, max(0, (p * 200)))
            votes[cls] = round(individual_score, 1)
        
        confidence = votes[best_crop]
    else:
        # Fallback: rule-based heuristic using GDD features
        best_crop, confidence, votes = _heuristic_predict_gdd(features)
    
    # Load zone data for display
    zone_data = load_zone_data()
    zone_name = None
    if zone_data and 0 <= zone_val < len(zone_data['zones']):
        zone_name = zone_data['zones'][zone_val]['name']
    
    return {
        'crop': best_crop,
        'confidence': round(confidence, 1),
        'votes': votes,
        'features_used': dict(zip(FEATURE_NAMES_GDD, features)),
        'season': season,
        'zone_id': zone_val,
        'zone_name': zone_name,
    }


def _heuristic_predict_gdd(features: list):
    """
    Rule-based fallback for GDD model when no sklearn model is available.
    Uses mean_temp and gdd instead of min/max temps.
    """
    mean_temp, gdd = features[0], features[1]
    rmin, rmax = features[2], features[3]
    humidity = features[10]
    ph = features[11]
    
    scores = {'Wheat': 0, 'Rice': 0, 'Maize': 0, 'Sugarcane': 0}
    
    # Wheat: cool, low rain (GDD 5-15)
    wheat_score = 0
    if 5 <= gdd <= 15 and rmin < 80:
        wheat_score += 3
    if 6.0 <= ph <= 7.5:
        wheat_score += 2
    scores['Wheat'] = min(100, wheat_score * 20)
    
    # Rice: warm, high rain (GDD > 15)
    rice_score = 0
    if gdd > 15 and rmax >= 100:
        rice_score += 3
    if humidity >= 70:
        rice_score += 2
    scores['Rice'] = min(100, rice_score * 20)
    
    # Maize: moderate (GDD 10-20)
    maize_score = 0
    if 10 <= gdd <= 20 and 50 <= rmin <= 120:
        maize_score += 3
    if 5.5 <= ph <= 7.0:
        maize_score += 1
    scores['Maize'] = min(100, maize_score * 25)
    
    # Sugarcane: warm, abundant rain (GDD > 12)
    cane_score = 0
    if gdd > 12 and rmax >= 120:
        cane_score += 3
    if humidity >= 60:
        cane_score += 1
    scores['Sugarcane'] = min(100, cane_score * 25)
    
    best = max(scores, key=scores.get)
    confidence = scores[best]
    
    return best, confidence, scores


def _heuristic_predict(features: list):
    """
    Simple rule-based fallback when no sklearn model is available.
    Returns individual confidence scores (0-100) for each crop.
    Based on ICAR agroclimatic guidelines.
    """
    tmin, tmax, rmin, rmax = features[0], features[1], features[2], features[3]
    humidity = features[14]
    ph = features[15]

    scores = {'Wheat': 0, 'Rice': 0, 'Maize': 0, 'Sugarcane': 0}

    # Wheat: cool, low rain, neutral pH (score out of 5 max)
    wheat_score = 0
    if 10 <= tmin <= 22 and rmin < 80:
        wheat_score += 3
    if 6.0 <= ph <= 7.5:
        wheat_score += 2
    scores['Wheat'] = min(100, wheat_score * 20)

    # Rice: hot, high rain, high humidity (score out of 5 max)
    rice_score = 0
    if tmax >= 28 and rmax >= 100:
        rice_score += 3
    if humidity >= 70:
        rice_score += 2
    scores['Rice'] = min(100, rice_score * 20)

    # Maize: moderate temp, moderate rain (score out of 4 max)
    maize_score = 0
    if 16 <= tmin <= 26 and 50 <= rmin <= 120:
        maize_score += 3
    if 5.5 <= ph <= 7.0:
        maize_score += 1
    scores['Maize'] = min(100, maize_score * 25)

    # Sugarcane: hot, abundant rain (score out of 4 max)
    cane_score = 0
    if tmax >= 30 and rmax >= 120:
        cane_score += 3
    if humidity >= 60:
        cane_score += 1
    scores['Sugarcane'] = min(100, cane_score * 25)

    # Find best crop
    best = max(scores, key=scores.get)
    confidence = scores[best]
    
    return best, confidence, scores


def get_crop_info() -> dict:
    """Return the static crop information dictionary."""
    return CROP_INFO


def allowed_file(filename: str) -> bool:
    """Check if uploaded file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ─── v2 Model — Improved Pipeline with OHE + Feature Engineering ──────────────
# Matches preprocessing in train_v2.py exactly.

# Numeric features used by v2 model (for StandardScaler alignment)
NUMERIC_FEATURES_V2 = [
    'mean_temp', 'rain_avg',
    'Sand_pct', 'Silt_pct', 'Clay_pct',
    'N_kg_ha', 'P_kg_ha', 'K_kg_ha',
    'Humidity_pct', 'pH',
    'temp_rain_ratio', 'npk_sum', 'soil_texture_index',
]


def load_model_v2(model_path: str = 'models/rf_model_v2.pkl') -> dict:
    """
    Load the v2 Random Forest model from disk.
    The bundle includes: sklearn model, StandardScaler, OHE feature_columns,
    and sensitivity test results.
    """
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"[BhoomiAI] v2 model loaded from {model_path} "
              f"(acc={model.get('accuracy','?')}%)")
        return model
    else:
        print(f"[BhoomiAI] v2 model not found at {model_path}. "
              f"Run: python train_v2.py")
        return {'sklearn': None, 'scaler': None, 'feature_columns': [],
                'numeric_features': NUMERIC_FEATURES_V2,
                'classes': CLASS_NAMES, 'accuracy': None, 'version': 'v2'}


def predict_crop_v2(model_v2: dict, inputs: dict) -> dict:
    """
    Run crop prediction using the v2 model.

    Applies the EXACT same preprocessing as train_v2.py:
      - Computes temp_rain_ratio, npk_sum, soil_texture_index
      - One-hot encodes Season and Agro_Zone
      - Scales numeric features with saved StandardScaler
      - Aligns to training feature_columns order

    Parameters
    ----------
    model_v2 : dict
        Output of load_model_v2().
    inputs : dict with keys:
        mean_temp, rain_avg, Sand_pct, Silt_pct, Clay_pct,
        N_kg_ha, P_kg_ha, K_kg_ha, Humidity_pct, pH,
        Season ('Rabi'|'Kharif'), Agro_Zone (int 0-14)

    Returns
    -------
    dict : {crop, confidence, votes, model_version}
    """
    import pandas as pd  # local import for minimal dependency surface

    clf            = model_v2.get('sklearn')
    scaler         = model_v2.get('scaler')
    feature_columns = model_v2.get('feature_columns', [])
    numeric_feats  = model_v2.get('numeric_features', NUMERIC_FEATURES_V2)

    if clf is None or not feature_columns:
        # Model not loaded — return helpful error instead of crashing
        return {
            'error': 'v2 model not loaded. Run: python train_v2.py',
            'crop': None, 'confidence': 0, 'votes': {},
            'model_version': 'v2-unavailable',
        }

    # ── Feature engineering ──────────────────────────────────────────────────
    mean_temp = float(inputs.get('mean_temp', 25))
    rain_avg  = float(inputs.get('rain_avg',  80))
    n   = float(inputs.get('N_kg_ha',      90))
    pv  = float(inputs.get('P_kg_ha',      45))
    k   = float(inputs.get('K_kg_ha',      45))
    sand = float(inputs.get('Sand_pct',    33))
    silt = float(inputs.get('Silt_pct',    33))
    clay = float(inputs.get('Clay_pct',    34))
    humidity = float(inputs.get('Humidity_pct', 65))
    ph       = float(inputs.get('pH',          6.5))
    season   = str(inputs.get('Season',  'Kharif'))
    zone_id  = int(inputs.get('Agro_Zone', 4))

    engineered = {
        'mean_temp':          mean_temp,
        'rain_avg':           rain_avg,
        'Sand_pct':           sand,
        'Silt_pct':           silt,
        'Clay_pct':           clay,
        'N_kg_ha':            n,
        'P_kg_ha':            pv,
        'K_kg_ha':            k,
        'Humidity_pct':       humidity,
        'pH':                 ph,
        'temp_rain_ratio':    mean_temp / max(rain_avg, 1.0),
        'npk_sum':            n + pv + k,
        'soil_texture_index': (clay + silt) / max(sand, 1.0),
    }

    # ── Build OHE-aligned row matching training column order ─────────────────
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
            row[col] = 0  # unseen zone or column → zero

    X = pd.DataFrame([row])[feature_columns]

    # ── Scale numeric features ───────────────────────────────────────────────
    X[numeric_feats] = scaler.transform(X[numeric_feats])

    # ── Predict ──────────────────────────────────────────────────────────────
    proba   = clf.predict_proba(X)[0]
    classes = clf.classes_
    best_idx  = int(np.argmax(proba))
    best_crop = classes[best_idx]

    votes = {}
    for cls, p in zip(classes, proba):
        individual_score = min(100, max(0, float(p) * 200))
        votes[cls] = round(individual_score, 1)

    confidence = votes[best_crop]

    # Load zone name for display
    zone_data = load_zone_data()
    zone_name = None
    if zone_data and 0 <= zone_id < len(zone_data.get('zones', [])):
        zone_name = zone_data['zones'][zone_id]['name']

    return {
        'crop':          best_crop,
        'confidence':    round(confidence, 1),
        'votes':         votes,
        'zone_id':       zone_id,
        'zone_name':     zone_name,
        'season':        season,
        'model_version': 'v2',
    }


# ─── v3 Model — Production-Level with Balanced Feature Importance ─────────────
# Matches preprocessing in train_v3.py exactly.
# Adds: Soil_Type OHE, Zone_Group OHE, zone_rain, zone_temp interactions

# Zone group definitions (must match train_v3.py / build_authentic_dataset_v3.py)
ZONE_GROUP_ARID   = {5, 12, 13}
ZONE_GROUP_HUMID  = {1, 2, 10, 11, 14}
ZONE_GROUP_PLAINS = {0, 3, 4, 6, 7, 8, 9}

NUMERIC_FEATURES_V3 = [
    'mean_temp', 'rain_avg',
    'Sand_pct', 'Silt_pct', 'Clay_pct',
    'N_kg_ha', 'P_kg_ha', 'K_kg_ha',
    'Humidity_pct', 'pH',
    'temp_rain_ratio', 'npk_sum', 'soil_texture_index',
    'zone_rain', 'zone_temp',
]


def load_model_v3(model_path: str = 'models/rf_model_v3.pkl') -> dict:
    """
    Load the v3 Random Forest model from disk.
    The bundle includes: sklearn model, StandardScaler, OHE feature_columns,
    sensitivity test results, and version metadata.
    """
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"[BhoomiAI] v3 model loaded from {model_path} "
              f"(acc={model.get('accuracy','?')}%)")
        return model
    else:
        print(f"[BhoomiAI] v3 model not found at {model_path}. "
              f"Run: python train_v3.py")
        return {'sklearn': None, 'scaler': None, 'feature_columns': [],
                'numeric_features': NUMERIC_FEATURES_V3,
                'classes': CLASS_NAMES, 'accuracy': None, 'version': 'v3'}


def predict_crop_v3(model_v3: dict, inputs: dict) -> dict:
    """
    Run crop prediction using the v3 model.

    Applies the EXACT same preprocessing as train_v3.py:
      - Computes temp_rain_ratio, npk_sum, soil_texture_index
      - Computes zone_rain, zone_temp (zone group interactions)
      - Derives Soil_Type (sandy/loamy/clayey) and Zone_Group (arid/humid/plains)
      - One-hot encodes Season, Agro_Zone, Soil_Type, Zone_Group
      - Scales numeric features with saved StandardScaler
      - Aligns to training feature_columns order

    Parameters
    ----------
    model_v3 : dict
        Output of load_model_v3().
    inputs : dict with keys:
        mean_temp, rain_avg, Sand_pct, Silt_pct, Clay_pct,
        N_kg_ha, P_kg_ha, K_kg_ha, Humidity_pct, pH,
        Season ('Rabi'|'Kharif'), Agro_Zone (int 0-14)

    Returns
    -------
    dict : {crop, confidence, votes, model_version, zone_id, zone_name, season}
    """
    import pandas as pd

    clf             = model_v3.get('sklearn')
    scaler          = model_v3.get('scaler')
    feature_columns = model_v3.get('feature_columns', [])
    numeric_feats   = model_v3.get('numeric_features', NUMERIC_FEATURES_V3)

    if clf is None or not feature_columns:
        return {
            'error': 'v3 model not loaded. Run: python train_v3.py',
            'crop': None, 'confidence': 0, 'votes': {},
            'model_version': 'v3-unavailable',
        }

    # ── Parse inputs ─────────────────────────────────────────────────────────
    mean_temp = float(inputs.get('mean_temp', 25))
    rain_avg  = float(inputs.get('rain_avg',  80))
    n    = float(inputs.get('N_kg_ha',      90))
    pv   = float(inputs.get('P_kg_ha',      45))
    k    = float(inputs.get('K_kg_ha',      45))
    sand = float(inputs.get('Sand_pct',     33))
    silt = float(inputs.get('Silt_pct',     33))
    clay = float(inputs.get('Clay_pct',     34))
    humidity = float(inputs.get('Humidity_pct', 65))
    ph       = float(inputs.get('pH',          6.5))
    season   = str(inputs.get('Season',  'Kharif'))
    zone_id  = int(inputs.get('Agro_Zone', 4))

    # ── Derive soil type ─────────────────────────────────────────────────────
    if clay >= 35.0:
        soil_type = 'clayey'
    elif sand >= 45.0:
        soil_type = 'sandy'
    else:
        soil_type = 'loamy'

    # ── Derive zone group ────────────────────────────────────────────────────
    if zone_id in ZONE_GROUP_ARID:
        zone_group = 'arid'
    elif zone_id in ZONE_GROUP_HUMID:
        zone_group = 'humid'
    else:
        zone_group = 'plains'

    is_arid  = 1.0 if zone_group == 'arid' else 0.0
    is_humid = 1.0 if zone_group == 'humid' else 0.0

    # ── Feature engineering ──────────────────────────────────────────────────
    engineered = {
        'mean_temp':          mean_temp,
        'rain_avg':           rain_avg,
        'Sand_pct':           sand,
        'Silt_pct':           silt,
        'Clay_pct':           clay,
        'N_kg_ha':            n,
        'P_kg_ha':            pv,
        'K_kg_ha':            k,
        'Humidity_pct':       humidity,
        'pH':                 ph,
        'temp_rain_ratio':    mean_temp / max(rain_avg, 1.0),
        'npk_sum':            n + pv + k,
        'soil_texture_index': (clay + silt) / max(sand, 1.0),
        'zone_rain':          rain_avg * is_arid,
        'zone_temp':          mean_temp * is_humid,
    }

    # ── Build OHE-aligned row matching training column order ─────────────────
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

    # ── Scale numeric features ───────────────────────────────────────────────
    X[numeric_feats] = scaler.transform(X[numeric_feats])

    # ── Predict ──────────────────────────────────────────────────────────────
    proba   = clf.predict_proba(X)[0]
    classes = clf.classes_
    best_idx  = int(np.argmax(proba))
    best_crop = classes[best_idx]

    votes = {}
    for cls, p in zip(classes, proba):
        individual_score = min(100, max(0, float(p) * 200))
        votes[cls] = round(individual_score, 1)

    confidence = votes[best_crop]

    # Load zone name for display
    zone_data = load_zone_data()
    zone_name = None
    if zone_data and 0 <= zone_id < len(zone_data.get('zones', [])):
        zone_name = zone_data['zones'][zone_id]['name']

    return {
        'crop':          best_crop,
        'confidence':    round(confidence, 1),
        'votes':         votes,
        'zone_id':       zone_id,
        'zone_name':     zone_name,
        'season':        season,
        'model_version': 'v3',
    }


# ─── New Model — 3-file bundle (crop_model + label_encoder + scaler) ──────────
# Features: Zone_ID, Season, Season_Temp, Temp_Range, N, P, K, pH,
#           Soil_Moisture, Irrigation
# Scaler  : MinMaxScaler
# Encoder : LabelEncoder (int → crop string)
# Season  : Rabi=0, Kharif=1, Zaid=2

NEW_MODEL_DIR        = 'newModel'
NEW_MODEL_FEATURE_NAMES = [
    'Zone_ID', 'Season', 'Season_Temp', 'Temp_Range',
    'Nitrogen', 'Phosphorus', 'Potassium', 'pH',
    'Soil_Moisture', 'Irrigation',
]

NEW_MODEL_SEASON_MAP = {'Rabi': 0, 'Kharif': 1, 'Zaid': 2}


def load_model_new(model_dir: str = NEW_MODEL_DIR) -> dict:
    """
    Load the new 3-file model bundle from disk.

    Files expected in model_dir:
        crop_model.pkl    — RandomForestClassifier
        label_encoder.pkl — LabelEncoder (int → crop name)
        scaler.pkl        — MinMaxScaler fitted on 10 features
    """
    clf_path = os.path.join(model_dir, 'crop_model.pkl')
    le_path  = os.path.join(model_dir, 'label_encoder.pkl')
    sc_path  = os.path.join(model_dir, 'scaler.pkl')

    if os.path.exists(clf_path) and os.path.exists(le_path) and os.path.exists(sc_path):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress sklearn version warnings
            with open(clf_path, 'rb') as f:
                clf = pickle.load(f)
            with open(le_path, 'rb') as f:
                le = pickle.load(f)
            with open(sc_path, 'rb') as f:
                scaler = pickle.load(f)
        n_est = getattr(clf, 'n_estimators', '?')
        print(f"[BhoomiAI] New model loaded from {model_dir} "
              f"(RF n_estimators={n_est}, 10-feature pipeline)")
        return {
            'sklearn':       clf,
            'label_encoder': le,
            'scaler':        scaler,
            'feature_names': NEW_MODEL_FEATURE_NAMES,
            'classes':       list(le.classes_),
            'version':       'new',
        }
    else:
        print(f"[BhoomiAI] New model files not found in {model_dir}.")
        return {
            'sklearn': None, 'label_encoder': None, 'scaler': None,
            'feature_names': NEW_MODEL_FEATURE_NAMES,
            'classes': CLASS_NAMES, 'version': 'new-unavailable',
        }


def predict_crop_new(model_new: dict, inputs: dict) -> dict:
    """
    Run crop prediction using the new 3-file model.

    Preprocessing pipeline:
        1. Map Season string → int  (Rabi=0, Kharif=1, Zaid=2)
        2. Derive temp_range = temp_max - temp_min  (or estimate from zone)
        3. Assemble 10-feature vector in exact scaler order
        4. MinMaxScaler.transform
        5. clf.predict / predict_proba
        6. LabelEncoder.inverse_transform → crop name

    Parameters
    ----------
    model_new : dict
        Output of load_model_new().
    inputs : dict with keys:
        mean_temp     — season mean temperature (°C)
        temp_min      — optional min temperature for temp_range derivation
        temp_max      — optional max temperature for temp_range derivation
        nitrogen      — N (kg/ha)
        phosphorus    — P (kg/ha)
        potassium     — K (kg/ha)
        ph            — soil pH
        soil_moisture — soil moisture % (11–61); fallback: humidity
        humidity      — used as soil_moisture proxy if soil_moisture absent
        irrigation    — 0 (rainfed) or 1 (irrigated)
        Season        — 'Rabi' | 'Kharif' | 'Zaid'
        Agro_Zone     — int 0-14  (model stores them as 1-14; we add 1 if needed)

    Returns
    -------
    dict : {crop, confidence, votes, zone_id, zone_name, season, model_version}
    """
    clf    = model_new.get('sklearn')
    le     = model_new.get('label_encoder')
    scaler = model_new.get('scaler')

    if clf is None or le is None or scaler is None:
        return {
            'error': 'New model not loaded. Check models/new_model/ directory.',
            'crop': None, 'confidence': 0, 'votes': {},
            'model_version': 'new-unavailable',
        }

    # ── Parse inputs ─────────────────────────────────────────────────────────
    season_str = str(inputs.get('Season', 'Kharif'))
    season_int = NEW_MODEL_SEASON_MAP.get(season_str, 1)

    # Zone_ID: the scaler was trained with IDs 1-14; our app uses 0-based zones.
    # Add 1 to convert (zone 0 → 1, zone 14 → 14 clamped)
    zone_id_app = int(inputs.get('Agro_Zone', 4))
    zone_id_model = max(1, min(14, zone_id_app + 1))

    mean_temp  = float(inputs.get('mean_temp', 25.0))

    # Temp_Range: prefer explicit min/max; otherwise use zone-based estimate
    # Temp_Range: prefer direct lookup value from ZONE_CLIMATE table;
    # fallback to explicit min/max; last resort season estimates
    if inputs.get('temp_range') not in (None, '', 0):
        temp_range = float(inputs['temp_range'])
    else:
        temp_min = inputs.get('temp_min')
        temp_max = inputs.get('temp_max')
        if temp_min is not None and temp_max is not None:
            temp_range = float(temp_max) - float(temp_min)
        else:
            # Estimate temp range from season: Rabi ≈ 14°C, Kharif ≈ 12°C, Zaid ≈ 16°C
            _season_range_defaults = {0: 14.0, 1: 12.0, 2: 16.0}
            temp_range = _season_range_defaults.get(season_int, 13.0)

    nitrogen   = float(inputs.get('N_kg_ha',       120.0))
    phosphorus = float(inputs.get('P_kg_ha',        25.0))
    potassium  = float(inputs.get('K_kg_ha',       120.0))
    ph         = float(inputs.get('pH',              6.5))
    irrigation = float(inputs.get('irrigation',      0.0))   # 0=rainfed, 1=irrigated

    # Soil_Moisture: use dedicated field, fallback to humidity proxy
    if inputs.get('soil_moisture') not in (None, '', 0):
        soil_moisture = float(inputs.get('soil_moisture'))
    else:
        # Map humidity (0-100%) to soil moisture approx range (11-61%)
        humidity = float(inputs.get('Humidity_pct', 50.0))
        soil_moisture = round(11.0 + (humidity / 100.0) * 50.0, 2)

    # ── Build feature vector (must match scaler column order exactly) ─────────
    X_raw = np.array([[
        zone_id_model,   # Zone_ID      (1-14)
        season_int,      # Season       (0/1/2)
        mean_temp,       # Season_Temp  (°C)
        temp_range,      # Temp_Range   (°C)
        nitrogen,        # Nitrogen     (kg/ha)
        phosphorus,      # Phosphorus   (kg/ha)
        potassium,       # Potassium    (kg/ha)
        ph,              # pH
        soil_moisture,   # Soil_Moisture(%)
        irrigation,      # Irrigation   (0/1)
    ]], dtype=float)

    # ── Scale ─────────────────────────────────────────────────────────────────
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        X_scaled = scaler.transform(X_raw)

    # ── Predict ──────────────────────────────────────────────────────────────
    proba      = clf.predict_proba(X_scaled)[0]
    pred_int   = int(np.argmax(proba))
    best_crop  = le.inverse_transform([pred_int])[0]

    # Map class indices to string names for votes dict
    votes = {}
    for idx, p in enumerate(proba):
        crop_name = le.inverse_transform([idx])[0]
        individual_score = min(100.0, max(0.0, float(p) * 200.0))
        votes[crop_name] = round(individual_score, 1)

    confidence = votes[best_crop]

    # Load zone name for display
    zone_data = load_zone_data()
    zone_name = None
    if zone_data and 0 <= zone_id_app < len(zone_data.get('zones', [])):
        zone_name = zone_data['zones'][zone_id_app]['name']

    return {
        'crop':           best_crop,
        'confidence':     round(confidence, 1),
        'votes':          votes,
        'zone_id':        zone_id_app,
        'zone_name':      zone_name,
        'season':         season_str,
        'model_version':  'new',
        'soil_moisture_used': round(soil_moisture, 1),
    }
