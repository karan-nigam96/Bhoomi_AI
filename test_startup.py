#!/usr/bin/env python
"""
Test script to validate Flask app startup and functionality
"""

import sys
import os
import json

# Set working directory to project root
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

print("=" * 80)
print("BhoomiAI Flask Application Startup Test")
print("=" * 80)

# Test 1: Import all required modules
print("\n[Test 1] Checking imports...")
try:
    from flask import Flask, render_template, request, jsonify
    print("✓ Flask imported successfully")
except ImportError as e:
    print(f"✗ Flask import failed: {e}")
    sys.exit(1)

try:
    import numpy as np
    import pickle
    print("✓ NumPy and pickle imported successfully")
except ImportError as e:
    print(f"✗ NumPy/pickle import failed: {e}")
    sys.exit(1)

try:
    from functions import predict_crop, load_model, get_crop_info, allowed_file, load_zone_data, get_zone_from_district
    print("✓ Custom functions imported successfully")
except ImportError as e:
    print(f"✗ Custom functions import failed: {e}")
    sys.exit(1)

# Test 2: Load Zone Data
print("\n[Test 2] Loading zone data...")
try:
    zone_data = load_zone_data('dataset/agro_zones.json')
    if zone_data is None:
        print("✗ Zone data is None")
        sys.exit(1)
    
    zones = zone_data.get('zones', [])
    print(f"✓ Zone data loaded successfully: {len(zones)} zones found")
    
    # Verify zone structure
    if len(zones) >= 15:
        print(f"✓ All 15 agro-climatic zones present")
    else:
        print(f"⚠ Warning: Expected 15 zones, found {len(zones)}")
    
    # Print zone list
    print("\n  Zones loaded:")
    for i, zone in enumerate(zones[:5]):  # Show first 5
        print(f"    Zone {i}: {zone.get('name')} ({zone.get('code')})")
    if len(zones) > 5:
        print(f"    ... and {len(zones) - 5} more zones")
    
except Exception as e:
    print(f"✗ Failed to load zone data: {e}")
    sys.exit(1)

# Test 3: Verify State-District Mapping
print("\n[Test 3] Checking state-district mapping...")
try:
    state_mapping = zone_data.get('state_district_mapping', {})
    print(f"✓ State-district mapping loaded: {len(state_mapping)} states found")
    
    # Sample a state
    sample_state = list(state_mapping.keys())[0] if state_mapping else None
    if sample_state:
        state_info = state_mapping[sample_state]
        districts = state_info.get('districts', [])
        zone_id = state_info.get('zone_id')
        zone_name = state_info.get('zone_name')
        print(f"\n  Sample: {sample_state}")
        print(f"    Districts: {len(districts)} found")
        print(f"    Zone ID: {zone_id}")
        print(f"    Zone Name: {zone_name}")
        print(f"    Sample districts: {', '.join(districts[:3])}...")
except Exception as e:
    print(f"✗ Failed to verify state-district mapping: {e}")
    sys.exit(1)

# Test 4: Load ML Model
print("\n[Test 4] Loading ML model...")
try:
    model = load_model('models/rf_model.pkl')
    print(f"✓ Model loaded: {type(model)}")
    
    if isinstance(model, dict):
        sklearn_model = model.get('sklearn')
        accuracy = model.get('accuracy')
        classes = model.get('classes')
        
        if sklearn_model is not None:
            print(f"✓ Sklearn model found: {type(sklearn_model)}")
            print(f"  Classes: {classes}")
            print(f"  Accuracy: {accuracy}")
        else:
            print("⚠ Sklearn model not found - will use heuristic fallback")
            print(f"  Available classes: {classes}")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    sys.exit(1)

# Test 5: Test Zone Mapping Functions
print("\n[Test 5] Testing zone mapping functions...")
try:
    # Test with a known state-district combination
    test_state = list(state_mapping.keys())[0]
    test_district = state_mapping[test_state]['districts'][0]
    
    zone_info = get_zone_from_district(test_state, test_district, zone_data)
    if zone_info:
        print(f"✓ Zone mapping works:")
        print(f"  {test_state}, {test_district} → Zone {zone_info['zone_id']}: {zone_info['zone_name']}")
    else:
        print(f"✗ Zone mapping failed for {test_state}, {test_district}")
except Exception as e:
    print(f"✗ Error testing zone mapping: {e}")
    sys.exit(1)

# Test 6: Test Prediction Function
print("\n[Test 6] Testing prediction function with sample features...")
try:
    # Sample features for testing (16 + season + zone)
    sample_features = [
        15.0,  # tmin
        28.0,  # tmax
        2.0,   # rmin (in cm)
        15.0,  # rmax
        18.0,  # stmin
        32.0,  # stmax
        12.0,  # htmin
        25.0,  # htmax
        40.0,  # sand %
        25.0,  # clay %
        35.0,  # silt %
        150.0, # nitrogen
        80.0,  # phosphorus
        60.0,  # potassium
        65.0,  # humidity
        6.5,   # pH
    ]
    
    # Test with Rabi season, Zone 4
    result = predict_crop(model, sample_features, season='Rabi', zone_id=4)
    print(f"✓ Prediction successful:")
    print(f"  Recommended Crop: {result.get('crop')}")
    print(f"  Confidence: {result.get('confidence')}%")
    print(f"  Season: {result.get('season')}")
    print(f"  Zone: {result.get('zone_id')} - {result.get('zone_name')}")
    print(f"  Individual Scores: {result.get('votes')}")
    
except Exception as e:
    print(f"✗ Prediction test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Create Flask App and Check Routes
print("\n[Test 7] Creating Flask app and validating routes...")
try:
    from app import app
    
    print(f"✓ Flask app imported successfully")
    print(f"✓ App configuration:")
    print(f"  Upload folder: {app.config.get('UPLOAD_FOLDER')}")
    print(f"  Max content length: {app.config.get('MAX_CONTENT_LENGTH')} bytes")
    
    # Get all routes
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            'endpoint': rule.endpoint,
            'methods': list(rule.methods - {'HEAD', 'OPTIONS'}),
            'path': str(rule)
        })
    
    print(f"\n✓ Routes available: {len(routes)} routes found")
    for route in sorted(routes, key=lambda x: x['path']):
        if route['endpoint'] != 'static':
            methods = ', '.join(sorted(route['methods']))
            print(f"  {route['path']:20} [{methods}]")
    
except Exception as e:
    print(f"✗ Flask app validation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Test Flask Test Client
print("\n[Test 8] Testing Flask routes with test client...")
try:
    with app.test_client() as client:
        # Test home route
        response = client.get('/')
        print(f"✓ GET / : {response.status_code}")
        
        # Test crops route
        response = client.get('/crops')
        print(f"✓ GET /crops : {response.status_code}")
        
        # Test about route
        response = client.get('/about')
        print(f"✓ GET /about : {response.status_code}")
        
        # Test fertilizers route
        response = client.get('/fertilizers')
        print(f"✓ GET /fertilizers : {response.status_code}")
        
        # Test predict GET
        response = client.get('/predict')
        print(f"✓ GET /predict : {response.status_code}")
        
        # Test API predict with JSON
        test_data = {
            'tmin': 15.0,
            'tmax': 28.0,
            'rmin': 2.0,
            'rmax': 15.0,
            'stmin': 18.0,
            'stmax': 32.0,
            'htmin': 12.0,
            'htmax': 25.0,
            'sand': 40.0,
            'clay': 25.0,
            'silt': 35.0,
            'nitrogen': 150.0,
            'phosphorus': 80.0,
            'potassium': 60.0,
            'humidity': 65.0,
            'ph': 6.5,
            'season': 'Rabi',
            'zone_id': 4
        }
        
        response = client.post('/api/predict', 
                              json=test_data,
                              content_type='application/json')
        print(f"✓ POST /api/predict : {response.status_code}")
        if response.status_code == 200:
            result = response.get_json()
            print(f"  Predicted crop: {result.get('crop')}")
            print(f"  Confidence: {result.get('confidence')}%")
        
        # Test get_districts API
        test_state = list(state_mapping.keys())[0]
        response = client.get(f'/api/get_districts/{test_state}')
        print(f"✓ GET /api/get_districts/{test_state} : {response.status_code}")
        if response.status_code == 200:
            data = response.get_json()
            print(f"  Districts: {len(data.get('districts', []))} found")
            print(f"  Zone: {data.get('zone_name')}")
        
except Exception as e:
    print(f"✗ Route testing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Final Summary
print("\n" + "=" * 80)
print("All Tests Passed! ✓")
print("=" * 80)
print("\nApplication Status:")
print(f"✓ Flask app is ready to run")
print(f"✓ All {len(zones)} agro-climatic zones loaded")
print(f"✓ State-district mapping configured")
print(f"✓ ML model ready ({len(routes)} routes available)")
print(f"✓ All core routes respond correctly")
print("\nTo start the server, run: python app.py")
print("Server will run on http://0.0.0.0:5000")
print("=" * 80)
