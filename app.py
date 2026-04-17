"""
BhoomiAI — Crop Recommendation System
Flask web application with Random Forest ML model (new version)

Model evolution:
  v1  — original GDD model
  v2  — OHE + engineered features (rain_avg, soil texture)
  v3  — zone-coupled, soil OHE, balanced feature importance
  new — 10-feature pipeline: Zone_ID, Season, Season_Temp, Temp_Range,
        N/P/K, pH, Soil_Moisture, Irrigation  (MinMaxScaler + LabelEncoder)
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import pickle
import os
import numpy as np
from functions import (
    predict_crop, predict_crop_gdd, load_model, load_model_gdd,
    load_model_v2, predict_crop_v2,
    load_model_v3, predict_crop_v3,
    load_model_new, predict_crop_new,
    get_crop_info, allowed_file, load_zone_data, get_zone_from_district,
    preprocess_soil_inputs, get_soil_defaults_for_region
)
from fertilizer_functions import load_fertilizer_models, predict_fertilizer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit
app.secret_key = 'bhoomiai_secret_2025'

# ── Zone Climate Lookup Table (21-year NASA averages) ──────────────────────────
# Keys: zone_id 1-15 (matching model training data)
# Season temps: kharif = monsoon max temp, rabi = winter min temp, zaid = average
ZONE_CLIMATE = {
    1:  {'kharif': 29.41, 'rabi': 12.10, 'zaid': 20.76, 'temp_range': 17.31},
    2:  {'kharif': 22.57, 'rabi':  8.58, 'zaid': 15.58, 'temp_range': 13.99},
    3:  {'kharif': 34.89, 'rabi': 17.58, 'zaid': 26.24, 'temp_range': 17.31},
    4:  {'kharif': 36.51, 'rabi': 16.45, 'zaid': 26.48, 'temp_range': 20.06},
    5:  {'kharif': 36.54, 'rabi': 14.91, 'zaid': 25.72, 'temp_range': 21.63},
    6:  {'kharif': 37.09, 'rabi': 14.69, 'zaid': 25.89, 'temp_range': 22.40},
    7:  {'kharif': 35.37, 'rabi': 17.56, 'zaid': 26.46, 'temp_range': 17.81},
    8:  {'kharif': 36.44, 'rabi': 15.97, 'zaid': 26.20, 'temp_range': 20.47},
    9:  {'kharif': 36.12, 'rabi': 16.82, 'zaid': 26.47, 'temp_range': 19.30},
    10: {'kharif': 36.72, 'rabi': 18.53, 'zaid': 27.62, 'temp_range': 18.19},
    11: {'kharif': 34.89, 'rabi': 23.20, 'zaid': 29.04, 'temp_range': 11.69},
    12: {'kharif': 32.08, 'rabi': 20.00, 'zaid': 26.04, 'temp_range': 12.08},
    13: {'kharif': 37.47, 'rabi': 18.12, 'zaid': 27.80, 'temp_range': 19.35},
    14: {'kharif': 38.42, 'rabi': 16.84, 'zaid': 27.63, 'temp_range': 21.58},
    15: {'kharif': 29.56, 'rabi': 26.61, 'zaid': 28.08, 'temp_range':  2.95},
}

# Load models and zone data at startup
MODEL_GDD = load_model_gdd()   # v1 legacy model (kept for /api/predict_legacy)
MODEL_V2  = load_model_v2()    # v2 model (fallback chain)
MODEL_V3  = load_model_v3()    # v3 model (fallback chain)
MODEL_NEW = load_model_new()   # new 3-file model (primary)
ZONE_DATA = load_zone_data()
FERT_MODELS = load_fertilizer_models()  # fertilizer NPK regressor + flag classifier

# Priority: new → v3 → v2
if MODEL_NEW.get('sklearn') is not None:
    ACTIVE_MODEL = MODEL_NEW
    ACTIVE_PREDICT = predict_crop_new
    print("[BhoomiAI] Using new model (primary)")
elif MODEL_V3.get('sklearn') is not None:
    ACTIVE_MODEL = MODEL_V3
    ACTIVE_PREDICT = predict_crop_v3
    print("[BhoomiAI] New model unavailable, falling back to v3")
else:
    ACTIVE_MODEL = MODEL_V2
    ACTIVE_PREDICT = predict_crop_v2
    print("[BhoomiAI] Falling back to v2 model")

# Base temperature for GDD calculation (kept for UI display only)
BASE_TEMP = 10.0


@app.route('/')
def index():
    """Home page with hero section."""
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Crop prediction page and endpoint (uses v3 production model, v2 fallback)."""
    if request.method == 'GET':
        return render_template('predict.html',
                               zones=ZONE_DATA['zones'] if ZONE_DATA else [],
                               states=list(ZONE_DATA['state_district_mapping'].keys()) if ZONE_DATA else [])

    # POST: handle JSON API call from frontend form
    data = request.get_json(silent=True) or request.form.to_dict()

    try:
        # ── NEW UI FLOW (zone_id 1-15, no user temp input) ───────────────────
        # Frontend sends: zone_id(1-15), season, nitrogen, phosphorus,
        # potassium, ph, soil_moisture, irrigation
        new_ui = data.get('new_ui', False)

        if new_ui:
            zone_id  = int(data.get('zone_id', 6))      # 1-15 from GPS/manual
            season   = str(data.get('season', 'Rabi'))  # 'Rabi'|'Kharif'|'Zaid'

            # Validate zone
            if zone_id < 1 or zone_id > 15:
                return jsonify({'error': f'Invalid zone_id {zone_id}. Must be 1-15.'}), 400

            # Auto-compute Season_Temp and Temp_Range from lookup table
            season_key = season.lower()
            climate    = ZONE_CLIMATE.get(zone_id, ZONE_CLIMATE[6])
            season_temp = climate.get(season_key, climate['kharif'])
            temp_range  = climate['temp_range']

            nitrogen   = float(data.get('nitrogen',    120))
            phosphorus = float(data.get('phosphorus',   25))
            potassium  = float(data.get('potassium',   120))
            ph         = float(data.get('ph',          6.5))
            soil_moist = float(data.get('soil_moisture', 35))
            irrigation = float(data.get('irrigation',   0))

            # Soil preprocessing not needed for new model (no sand/silt/clay);
            # kept for fallback chain compatibility
            soil_result = preprocess_soil_inputs(
                sand=None, silt=None, clay=None,
                zone_id=zone_id - 1, auto_fill=True, normalize=True
            )

            inputs_unified = {
                # New model fields (zone 0-based for existing +1 offset in predict_crop_new)
                'Agro_Zone':    zone_id - 1,
                'Season':       season,
                'mean_temp':    season_temp,
                'temp_range':   temp_range,
                'N_kg_ha':      nitrogen,
                'P_kg_ha':      phosphorus,
                'K_kg_ha':      potassium,
                'pH':           ph,
                'soil_moisture': soil_moist,
                'irrigation':   irrigation,
                # Legacy fields for v2/v3 fallback
                'rain_avg':     80.0,
                'Sand_pct':     soil_result['sand'],
                'Silt_pct':     soil_result['silt'],
                'Clay_pct':     soil_result['clay'],
                'Humidity_pct': 65.0,
            }

            result = ACTIVE_PREDICT(ACTIVE_MODEL, inputs_unified)
            result['soil_source']    = soil_result['source']
            result['soil_message']   = soil_result['message']
            result['season_temp']    = round(season_temp, 2)
            result['temp_range']     = round(temp_range, 2)
            result['zone_id']        = zone_id
            result['is_island_zone'] = (zone_id == 15)
            return jsonify(result)

        # ── LEGACY FLOW (old predict.html — zone 0-based, user enters temp) ──
        season  = data.get('season', 'Kharif')
        zone_id = data.get('zone_id')

        if zone_id is None:
            state    = data.get('state')
            district = data.get('district')
            if state and district:
                zone_info = get_zone_from_district(state, district, ZONE_DATA)
                if zone_info:
                    zone_id = zone_info['zone_id']
                else:
                    return jsonify(
                        {'error': 'Could not determine agro-climatic zone for selected location'}
                    ), 400
            else:
                zone_id = 4  # Default: Upper Gangetic Plains

        zone_id   = int(zone_id)
        mean_temp = float(data.get('mean_temp', 25))

        # ── Rainfall (kept for legacy routes; not used by new model) ─────────
        rmin = float(data.get('rmin', 0))
        rmax = float(data.get('rmax', 0))
        rain_avg = round((rmin + rmax) / 2, 2) if (rmin > 0 or rmax > 0) else 80.0

        # ── Soil with region defaults (legacy v2/v3 soil preprocessing) ──────
        soil_result = preprocess_soil_inputs(
            sand=data.get('sand'), silt=data.get('silt'), clay=data.get('clay'),
            zone_id=zone_id, auto_fill=True, normalize=True
        )

        # ── Build unified inputs dict (works for new model, v3, and v2) ──────
        inputs_unified = {
            # Core fields (all model versions)
            'mean_temp':    mean_temp,
            'rain_avg':     rain_avg,
            'Sand_pct':     soil_result['sand'],
            'Silt_pct':     soil_result['silt'],
            'Clay_pct':     soil_result['clay'],
            'N_kg_ha':      float(data.get('nitrogen',      120)),
            'P_kg_ha':      float(data.get('phosphorus',     25)),
            'K_kg_ha':      float(data.get('potassium',     120)),
            'Humidity_pct': float(data.get('humidity',       65)),
            'pH':           float(data.get('ph',            6.5)),
            'Season':       season,
            'Agro_Zone':    zone_id,
            # New model-specific fields
            'temp_min':      data.get('temp_min'),
            'temp_max':      data.get('temp_max'),
            'soil_moisture': data.get('soil_moisture'),
            'irrigation':    float(data.get('irrigation', 0)),
        }

    except (ValueError, TypeError) as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400

    result = ACTIVE_PREDICT(ACTIVE_MODEL, inputs_unified)
    result['soil_source']   = soil_result['source']
    result['soil_message']  = soil_result['message']
    result['rain_avg_used'] = rain_avg
    return jsonify(result)


@app.route('/crops')
def crops():
    """Crop information page."""
    crop_data = get_crop_info()
    return render_template('crops.html', crops=crop_data)


@app.route('/fertilizers')
def fertilizers():
    """Fertilizer guide page."""
    return render_template('fertilizers.html')


@app.route('/fertilizer-recommend', methods=['GET'])
def fertilizer_recommend():
    """Fertilizer recommendation page — runs after crop prediction."""
    return render_template('fertilizer_recommend.html')


@app.route('/api/fertilizer/recommend', methods=['POST'])
def api_fertilizer_recommend():
    """
    Fertilizer recommendation API endpoint.
    Accepts crop, zone_id, season, soil NPK/pH, irrigation, variety,
    organic input type, previous crop, and farm size.
    Returns NPK doses per ha + total, special treatment flags,
    and a stage-wise application schedule.
    """
    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json'}), 415

    data = request.get_json()

    required = ['crop', 'zone_id', 'season', 'n_soil', 'p_soil',
                'k_soil', 'ph', 'irrigation', 'variety',
                'organic', 'prev_crop', 'farm_size']
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({'error': f'Missing required fields: {missing}'}), 400

    if FERT_MODELS is None:
        return jsonify({'error': 'Fertilizer model not loaded. Check server logs.'}), 503

    try:
        result = predict_fertilizer(
            models    = FERT_MODELS,
            crop      = str(data['crop']),
            zone_id   = int(data['zone_id']),
            season    = int(data['season']),
            n_soil    = float(data['n_soil']),
            p_soil    = float(data['p_soil']),
            k_soil    = float(data['k_soil']),
            ph        = float(data['ph']),
            irrigation  = int(data['irrigation']),
            variety     = str(data['variety']),
            organic     = str(data['organic']),
            prev_crop   = str(data['prev_crop']),
            farm_size   = float(data['farm_size']),
        )
        result['success']      = True
        result['crop']         = data['crop']
        result['farm_size_ha'] = float(data['farm_size'])
        return jsonify(result)
    except (ValueError, KeyError) as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 400


@app.route('/about')
def about():
    """About page with model accuracy info."""
    accuracy = MODEL_GDD.get('accuracy', 99.48) if isinstance(MODEL_GDD, dict) else 99.48
    return render_template('about.html', accuracy=accuracy)


@app.route('/api/get_districts/<state>')
def get_districts(state):
    """API endpoint to get districts for a state."""
    if ZONE_DATA and state in ZONE_DATA['state_district_mapping']:
        return jsonify({
            'districts': ZONE_DATA['state_district_mapping'][state]['districts'],
            'zone_id': ZONE_DATA['state_district_mapping'][state]['zone_id'],
            'zone_name': ZONE_DATA['state_district_mapping'][state]['zone_name']
        })
    return jsonify({'error': 'State not found'}), 404


@app.route('/api/soil_defaults/<int:zone_id>')
def get_soil_defaults(zone_id):
    """API endpoint to get default soil values for an agro-climatic zone.
    
    Returns sand%, silt%, clay% defaults based on ICAR soil survey data.
    """
    defaults = get_soil_defaults_for_region(zone_id)
    zone_name = None
    if ZONE_DATA and 0 <= zone_id < len(ZONE_DATA['zones']):
        zone_name = ZONE_DATA['zones'][zone_id]['name']
    
    return jsonify({
        'zone_id': zone_id,
        'zone_name': zone_name,
        'soil_defaults': defaults,
        'source': 'ICAR soil survey data'
    })


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    REST API endpoint (v3 model, v2 fallback) for external tools / Raspberry Pi.

    Required JSON fields:
        mean_temp, rmin, rmax, nitrogen, phosphorus, potassium, humidity, ph
    Optional:
        sand, silt, clay  (auto-filled from zone defaults if missing)
        season             (default: 'Kharif')
        zone_id            (default: 4 — Upper Gangetic Plains)

    v3 change: send rmin + rmax; rain_avg is computed server-side.
    """
    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json'}), 415

    data = request.get_json()

    required = ['mean_temp', 'rmin', 'rmax',
                'nitrogen', 'phosphorus', 'potassium', 'humidity', 'ph']
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({'error': f'Missing fields: {missing}'}), 400

    try:
        season  = data.get('season', 'Kharif')
        zone_id = int(data.get('zone_id', 4))
        rmin    = float(data['rmin'])
        rmax    = float(data['rmax'])
        rain_avg = round((rmin + rmax) / 2, 2)

        soil_result = preprocess_soil_inputs(
            sand=data.get('sand'), silt=data.get('silt'), clay=data.get('clay'),
            zone_id=zone_id, auto_fill=True, normalize=True
        )

        inputs_unified = {
            'mean_temp':    float(data['mean_temp']),
            'rain_avg':     rain_avg,
            'Sand_pct':     soil_result['sand'],
            'Silt_pct':     soil_result['silt'],
            'Clay_pct':     soil_result['clay'],
            'N_kg_ha':      float(data['nitrogen']),
            'P_kg_ha':      float(data['phosphorus']),
            'K_kg_ha':      float(data['potassium']),
            'Humidity_pct': float(data['humidity']),
            'pH':           float(data['ph']),
            'Season':       season,
            'Agro_Zone':    zone_id,
            # New model-specific fields
            'temp_min':      data.get('temp_min'),
            'temp_max':      data.get('temp_max'),
            'soil_moisture': data.get('soil_moisture'),
            'irrigation':    float(data.get('irrigation', 0)),
        }
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    result = ACTIVE_PREDICT(ACTIVE_MODEL, inputs_unified)
    result['soil_source']  = soil_result['source']
    result['soil_message'] = soil_result['message']
    result['rain_avg_used'] = rain_avg
    return jsonify(result)


@app.route('/api/predict_legacy', methods=['POST'])
def api_predict_legacy():
    """
    Legacy REST API endpoint (v1 GDD model) — kept for backward compatibility.
    Use /api/predict for the improved v2 model.
    """
    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json'}), 415

    data = request.get_json()
    required = ['mean_temp', 'gdd', 'rmin', 'rmax',
                'nitrogen', 'phosphorus', 'potassium', 'humidity', 'ph']
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({'error': f'Missing fields: {missing}'}), 400

    try:
        season      = data.get('season', 'Kharif')
        season_code = 0 if season == 'Rabi' else 1
        zone_id     = int(data.get('zone_id', 4))
        soil_result = preprocess_soil_inputs(
            sand=data.get('sand'), silt=data.get('silt'), clay=data.get('clay'),
            zone_id=zone_id, auto_fill=True, normalize=True
        )
        features = [
            float(data['mean_temp']), float(data['gdd']),
            float(data['rmin']),      float(data['rmax']),
            soil_result['sand'],      soil_result['clay'],  soil_result['silt'],
            float(data['nitrogen']),  float(data['phosphorus']), float(data['potassium']),
            float(data['humidity']),  float(data['ph']),
            season_code, zone_id,
        ]
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    result = predict_crop_gdd(MODEL_GDD, features, season=season, zone_id=zone_id)
    result['soil_source']  = soil_result['source']
    result['soil_message'] = soil_result['message']
    result['model_note']   = 'Legacy v1 GDD model. Use /api/predict for v2.'
    return jsonify(result)


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
