"""Live API validation tests for BhoomiAI v3 model."""
import urllib.request
import json

BASE = 'http://localhost:5000/api/predict'

TESTS = [
    ('Wheat',     {'mean_temp': 17, 'rmin': 30,  'rmax': 55,
                   'nitrogen': 90,  'phosphorus': 40, 'potassium': 40,
                   'humidity': 50,  'ph': 6.8,
                   'sand': 50, 'silt': 30, 'clay': 20,
                   'season': 'Rabi', 'zone_id': 13}),
    ('Rice',      {'mean_temp': 30, 'rmin': 140, 'rmax': 180,
                   'nitrogen': 120, 'phosphorus': 40, 'potassium': 40,
                   'humidity': 88,  'ph': 5.2,
                   'sand': 18, 'silt': 35, 'clay': 47,
                   'season': 'Kharif', 'zone_id': 10}),
    ('Sugarcane', {'mean_temp': 31, 'rmin': 100, 'rmax': 140,
                   'nitrogen': 220, 'phosphorus': 80, 'potassium': 80,
                   'humidity': 72,  'ph': 6.8,
                   'sand': 25, 'silt': 35, 'clay': 40,
                   'season': 'Kharif', 'zone_id': 5}),
    ('Maize',     {'mean_temp': 25, 'rmin': 60,  'rmax': 90,
                   'nitrogen': 100, 'phosphorus': 45, 'potassium': 45,
                   'humidity': 65,  'ph': 6.5,
                   'sand': 45, 'silt': 30, 'clay': 25,
                   'season': 'Kharif', 'zone_id': 9}),
]

print('=' * 62)
print('  BhoomiAI v3 -- Live API Validation Tests')
print('=' * 62)

passed = 0
for label, payload in TESTS:
    try:
        data = json.dumps(payload).encode()
        req  = urllib.request.Request(
            BASE, data=data, headers={'Content-Type': 'application/json'}
        )
        resp = json.loads(urllib.request.urlopen(req, timeout=10).read())

        crop    = resp.get('crop', 'N/A')
        conf    = resp.get('confidence', 0)
        version = resp.get('model_version', '?')
        votes   = resp.get('votes', {})
        err     = resp.get('error')

        if err:
            print(f'  [FAIL] {label}')
            print(f'         error: {err}')
            continue

        ok = label.lower() in crop.lower()
        mark = 'PASS' if ok else 'FAIL'
        if ok:
            passed += 1
        print(f'  [{mark}] Expected={label:10s} Got={crop:10s} conf={conf:.1f}% model={version}')
        print(f'         votes: {votes}')

    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f'  [ERR ] {label}: HTTP {e.code}')
        for line in body.split('\n'):
            stripped = line.strip()
            if ('Error' in stripped or 'error' in stripped) and len(stripped) < 200:
                print(f'         {stripped}')
                break
    except Exception as ex:
        print(f'  [ERR ] {label}: {ex}')

print('=' * 62)
print(f'  Result: {passed}/{len(TESTS)} tests passed')
print('=' * 62)
