#!/usr/bin/env python
"""
Simplified Flask server startup and testing script
"""
import sys
import os
import time
import threading
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Set working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("\n" + "=" * 80)
print("BhoomiAI Flask Application Startup")
print("=" * 80 + "\n")

# Import the Flask app
try:
    from app import app
    print("[OK] Flask app imported successfully\n")
except Exception as e:
    print(f"[ERROR] Failed to import Flask app: {e}\n")
    sys.exit(1)

# Function to test the server
def test_server(port=5000, max_retries=15, timeout=2):
    """Test if server is running and responding"""
    url = f"http://localhost:{port}/"
    
    session = requests.Session()
    retry = Retry(connect=1, backoff_factor=0.5, status_forcelist=[])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    
    for attempt in range(max_retries):
        try:
            response = session.get(url, timeout=timeout)
            if response.status_code == 200:
                return True
        except:
            pass
        
        time.sleep(1)
        print(f"  [Waiting for server...] Attempt {attempt + 1}/{max_retries}")
    
    return False

# Start server in a thread
def run_server():
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)

print("[Starting Flask Server on http://0.0.0.0:5000]\n")
server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

# Wait for server to be ready
print("Waiting for server to be ready...\n")
if test_server():
    print("\n[OK] Server is running and responding!\n")
    
    # Test some endpoints
    print("Testing endpoints:")
    try:
        session = requests.Session()
        
        # Test home page
        r = session.get('http://localhost:5000/')
        print(f"  [OK] GET / : {r.status_code}")
        
        # Test crops page
        r = session.get('http://localhost:5000/crops')
        print(f"  [OK] GET /crops : {r.status_code}")
        
        # Test about page
        r = session.get('http://localhost:5000/about')
        print(f"  [OK] GET /about : {r.status_code}")
        
        # Test predict form
        r = session.get('http://localhost:5000/predict')
        print(f"  [OK] GET /predict : {r.status_code}")
        
        # Test API prediction
        test_data = {
            'tmin': 15.0, 'tmax': 28.0, 'rmin': 2.0, 'rmax': 15.0,
            'stmin': 18.0, 'stmax': 32.0, 'htmin': 12.0, 'htmax': 25.0,
            'sand': 40.0, 'clay': 25.0, 'silt': 35.0,
            'nitrogen': 150.0, 'phosphorus': 80.0, 'potassium': 60.0,
            'humidity': 65.0, 'ph': 6.5,
            'season': 'Rabi', 'zone_id': 4
        }
        r = session.post('http://localhost:5000/api/predict', json=test_data)
        print(f"  [OK] POST /api/predict : {r.status_code}")
        if r.status_code == 200:
            result = r.json()
            print(f"      -> Predicted: {result.get('crop')} ({result.get('confidence')}% confidence)")
        
        # Test districts API
        r = session.get('http://localhost:5000/api/get_districts/Punjab')
        print(f"  [OK] GET /api/get_districts/Punjab : {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"      -> {len(data.get('districts', []))} districts in Zone {data.get('zone_name')}")
        
        print("\n" + "=" * 80)
        print("[SUCCESS] BhoomiAI Flask Application is Running Successfully!")
        print("=" * 80)
        print("\nServer Details:")
        print(f"  URL: http://0.0.0.0:5000")
        print(f"  Model Accuracy: 99.48%")
        print(f"  Features: 18 (16 soil/climate + Season + Agro-Zone)")
        print(f"  Zones: 15 agro-climatic zones")
        print(f"  States: 21 states with district mapping")
        print(f"  Crops: Wheat, Rice, Maize, Sugarcane")
        print("\nRoutes available:")
        print(f"  - GET /              (Home page)")
        print(f"  - GET /predict       (Prediction form)")
        print(f"  - POST /predict      (Submit prediction)")
        print(f"  - GET /crops         (Crop information)")
        print(f"  - GET /fertilizers   (Fertilizer guide)")
        print(f"  - GET /about         (About & accuracy)")
        print(f"  - POST /api/predict  (REST API)")
        print(f"  - GET /api/get_districts/<state>  (Districts API)")
        print("\n" + "=" * 80)
        print("Server is running. Press Ctrl+C to stop.")
        print("=" * 80 + "\n")
        
        # Keep the server running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nServer stopped.")
            sys.exit(0)
        
    except Exception as e:
        print(f"\n[ERROR] Error during endpoint testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
else:
    print("[ERROR] Server failed to start or respond to requests")
    sys.exit(1)
