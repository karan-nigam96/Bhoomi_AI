"""
BhoomiAI -- Authentic Dataset Builder v3
========================================
Improvements over v2:
  - Zone-coupled climate: temp/rain ranges tightened per zone
    so the model can learn zone ↔ climate relationships
  - Soil type categorical flag: sandy / loamy / clayey
    derived from texture fractions → one-hot encoded in training
  - Macro-zone groups: arid / humid / plains
    for interaction features in training
  - 400 samples per crop (1600 total) for more diversity
  - Zone-specific crop exclusions (no sugarcane in desert zone 13)

Sources:
  - ICAR Crop Production Guides (icar.org.in)
  - NBSS&LUP National Soil Survey (nbsslup.icar.gov.in)
  - IMD Climate Normals (imd.gov.in)
  - Planning Commission 15 Agro-Climatic Zone Classification

Output:
  dataset/crop_train_v3.csv
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

# --- Zone-Specific Climate Modifiers -----------------------------------------
# Realistic temperature and rainfall bounds per zone (IMD Climate Normals)
# These OVERRIDE crop-level ranges when the zone imposes tighter bounds

ZONE_CLIMATE = {
    0:  {'temp': (8, 22),   'rain': (50, 90),   'name': 'W.Himalayan'},
    1:  {'temp': (14, 30),  'rain': (100, 200),  'name': 'E.Himalayan'},
    2:  {'temp': (22, 36),  'rain': (100, 180),  'name': 'Lower Gangetic'},
    3:  {'temp': (18, 35),  'rain': (70, 140),   'name': 'Middle Gangetic'},
    4:  {'temp': (16, 34),  'rain': (55, 120),   'name': 'Upper Gangetic'},
    5:  {'temp': (14, 36),  'rain': (40, 90),    'name': 'Trans-Gangetic'},
    6:  {'temp': (20, 34),  'rain': (80, 150),   'name': 'E.Plateau'},
    7:  {'temp': (20, 36),  'rain': (60, 120),   'name': 'Central Plateau'},
    8:  {'temp': (22, 36),  'rain': (50, 100),   'name': 'W.Plateau'},
    9:  {'temp': (22, 35),  'rain': (60, 130),   'name': 'S.Plateau'},
    10: {'temp': (24, 34),  'rain': (90, 180),   'name': 'E.Coast'},
    11: {'temp': (22, 34),  'rain': (120, 250),  'name': 'W.Coast'},
    12: {'temp': (20, 38),  'rain': (40, 90),    'name': 'Gujarat'},
    13: {'temp': (22, 42),  'rain': (15, 50),    'name': 'W.Dry (Desert)'},
    14: {'temp': (24, 32),  'rain': (120, 220),  'name': 'Islands'},
}

# --- Macro-Zone Groups (for interaction features) ----------------------------
ZONE_GROUP_ARID   = {5, 12, 13}
ZONE_GROUP_HUMID  = {1, 2, 10, 11, 14}
ZONE_GROUP_PLAINS = {0, 3, 4, 6, 7, 8, 9}

# --- Authentic ICAR Crop Parameters ------------------------------------------
CROP_PARAMS = {
    'Wheat': {
        'season': ['Rabi'],
        'zones': [3, 4, 5, 7, 8, 12, 13],
        'mean_temp': (12.0, 24.0),
        'rain_avg': (25.0, 70.0),
        'humidity': (38.0, 68.0),
        'pH': (6.0, 7.5),
        'N': (60.0, 150.0),
        'P': (25.0, 60.0),
        'K': (25.0, 60.0),
    },
    'Rice': {
        'season': ['Kharif'],
        'zones': [1, 2, 3, 6, 10, 11, 14],
        'mean_temp': (23.0, 36.0),
        'rain_avg': (100.0, 190.0),
        'humidity': (72.0, 95.0),
        'pH': (4.5, 6.5),
        'N': (80.0, 150.0),
        'P': (25.0, 55.0),
        'K': (25.0, 60.0),
    },
    'Maize': {
        'season': ['Kharif', 'Kharif', 'Kharif', 'Rabi'],
        'zones': [0, 4, 5, 6, 7, 8, 9],
        'mean_temp': (18.0, 32.0),
        'rain_avg': (45.0, 120.0),
        'humidity': (50.0, 80.0),
        'pH': (5.5, 7.5),
        'N': (80.0, 150.0),
        'P': (30.0, 60.0),
        'K': (30.0, 60.0),
    },
    'Sugarcane': {
        'season': ['Kharif'],
        # Removed zone 13 (desert) -- unrealistic for sugarcane
        'zones': [2, 3, 4, 5, 9, 10, 11, 12],
        'mean_temp': (22.0, 38.0),
        'rain_avg': (80.0, 168.0),
        'humidity': (60.0, 85.0),
        'pH': (6.0, 7.5),
        'N': (120.0, 250.0),
        'P': (55.0, 100.0),
        'K': (55.0, 100.0),
    },
}

# --- Crop-Specific Soil Texture Preferences -----------------------------------
# In reality, farmers select fields with appropriate soil for each crop.
# These biases shift the zone soil distribution toward crop-preferred texture.
#   Rice      → heavy clay/silty (paddy needs water retention)
#   Wheat     → sandy-loam (well-drained, aerated roots)
#   Maize     → sandy-loam to loam (moderate drainage)
#   Sugarcane → deep loam/clay-loam (moisture + nutrients)
#
# Format: {'sand_shift': %, 'clay_shift': %}  (silt absorbs remainder)

CROP_SOIL_BIAS = {
    'Rice':      {'sand_shift': -15, 'clay_shift': +18},  # push toward clay
    'Wheat':     {'sand_shift': +14, 'clay_shift': -10},  # push toward sandy-loam
    'Maize':     {'sand_shift': +6,  'clay_shift': -4},   # slight sandy-loam
    'Sugarcane': {'sand_shift': -6,  'clay_shift': +10},  # push toward clay-loam
}

# --- NBSS&LUP Zone-specific Soil Composition ---------------------------------
ZONE_SOIL = {
    0:  {'sand': (40, 58), 'silt': (24, 38), 'clay': (12, 25)},
    1:  {'sand': (32, 52), 'silt': (26, 42), 'clay': (15, 32)},
    2:  {'sand': (18, 36), 'silt': (28, 46), 'clay': (26, 45)},
    3:  {'sand': (22, 42), 'silt': (30, 46), 'clay': (20, 38)},
    4:  {'sand': (28, 50), 'silt': (28, 44), 'clay': (16, 34)},
    5:  {'sand': (32, 56), 'silt': (25, 42), 'clay': (13, 30)},
    6:  {'sand': (28, 48), 'silt': (20, 35), 'clay': (22, 42)},
    7:  {'sand': (16, 34), 'silt': (22, 40), 'clay': (32, 52)},
    8:  {'sand': (12, 28), 'silt': (22, 38), 'clay': (38, 58)},
    9:  {'sand': (25, 48), 'silt': (22, 40), 'clay': (24, 44)},
    10: {'sand': (22, 44), 'silt': (28, 46), 'clay': (18, 36)},
    11: {'sand': (32, 55), 'silt': (22, 38), 'clay': (16, 36)},
    12: {'sand': (28, 52), 'silt': (24, 42), 'clay': (18, 36)},
    13: {'sand': (48, 72), 'silt': (15, 30), 'clay': (10, 24)},
    14: {'sand': (40, 62), 'silt': (20, 36), 'clay': (12, 28)},
}

SAMPLES_PER_CROP = 400
RANDOM_SEED = 42
OUTPUT_PATH = os.path.join('dataset', 'crop_train_v3.csv')


def gen_soil_for_zone(zone_id: int, crop: str = None) -> tuple:
    """
    Generate soil composition for a zone with crop-specific bias.

    The base distribution comes from NBSS&LUP zone data, then a crop-specific
    shift is applied to simulate real-world field selection patterns:
      - Rice farmers select clay-heavy paddy fields
      - Wheat farmers prefer well-drained sandy-loam
      - Sugarcane needs deep loam/clay-loam

    Normalises to ensure sand + silt + clay = exactly 100%.
    """
    z = ZONE_SOIL[zone_id]
    sand = np.random.uniform(*z['sand'])
    silt = np.random.uniform(*z['silt'])
    clay = np.random.uniform(*z['clay'])

    # Apply crop-specific soil texture bias
    if crop and crop in CROP_SOIL_BIAS:
        bias = CROP_SOIL_BIAS[crop]
        sand += bias['sand_shift']
        clay += bias['clay_shift']
        # Silt absorbs the remainder to keep proportions reasonable
        silt -= (bias['sand_shift'] + bias['clay_shift']) * 0.5

    # Clamp to valid ranges (must be positive)
    sand = max(5.0, sand)
    silt = max(5.0, silt)
    clay = max(5.0, clay)

    total = sand + silt + clay
    sand = round(sand / total * 100, 2)
    silt = round(silt / total * 100, 2)
    clay = round(100.0 - sand - silt, 2)
    return sand, silt, clay


def classify_soil_type(sand: float, silt: float, clay: float) -> str:
    """
    Classify soil into broad textural category.

    Thresholds tuned for better distribution across crops:
      - clayey: clay >= 35%  (rice/sugarcane-heavy soils)
      - sandy:  sand >= 45%  (wheat/maize preferred soils)
      - loamy:  everything else (balanced texture)
    """
    if clay >= 35.0:
        return 'clayey'
    elif sand >= 45.0:
        return 'sandy'
    else:
        return 'loamy'


def get_zone_group(zone_id: int) -> str:
    """Return macro zone group: arid, humid, or plains."""
    if zone_id in ZONE_GROUP_ARID:
        return 'arid'
    elif zone_id in ZONE_GROUP_HUMID:
        return 'humid'
    else:
        return 'plains'


def gen_npk(crop: str, tier: str) -> tuple:
    """Generate NPK values at low / medium / high tiers."""
    p = CROP_PARAMS[crop]
    n_lo, n_hi = p['N']
    p_lo, p_hi = p['P']
    k_lo, k_hi = p['K']
    nr, pr, kr = n_hi - n_lo, p_hi - p_lo, k_hi - k_lo

    if tier == 'low':
        n  = np.random.uniform(n_lo,             n_lo + nr * 0.33)
        pv = np.random.uniform(p_lo,             p_lo + pr * 0.33)
        k  = np.random.uniform(k_lo,             k_lo + kr * 0.33)
    elif tier == 'high':
        n  = np.random.uniform(n_lo + nr * 0.67, n_hi)
        pv = np.random.uniform(p_lo + pr * 0.67, p_hi)
        k  = np.random.uniform(k_lo + kr * 0.67, k_hi)
    else:  # medium
        n  = np.random.uniform(n_lo + nr * 0.33, n_lo + nr * 0.67)
        pv = np.random.uniform(p_lo + pr * 0.33, p_lo + pr * 0.67)
        k  = np.random.uniform(k_lo + kr * 0.33, k_lo + kr * 0.67)

    return round(n, 1), round(pv, 1), round(k, 1)


def clamp(val, lo, hi):
    """Clamp val between lo and hi."""
    return max(lo, min(hi, val))


def gen_zone_coupled_climate(zone_id: int, crop: str) -> tuple:
    """
    Generate temperature and rainfall coupled to the zone's realistic climate.

    Strategy: intersect zone climate bounds with crop climate bounds.
    This forces zone 13 (desert) to have low rain and zone 2 (L.Gangetic)
    to have high rain, making zone a meaningful predictor.
    """
    zc = ZONE_CLIMATE[zone_id]
    cp = CROP_PARAMS[crop]

    # Intersect temp range: crop requirement ∩ zone reality
    temp_lo = max(cp['mean_temp'][0], zc['temp'][0])
    temp_hi = min(cp['mean_temp'][1], zc['temp'][1])
    if temp_lo > temp_hi:
        # Fallback: use midpoint with small jitter
        mid = (cp['mean_temp'][0] + cp['mean_temp'][1] + zc['temp'][0] + zc['temp'][1]) / 4
        temp_lo, temp_hi = mid - 2, mid + 2

    # Intersect rain range
    rain_lo = max(cp['rain_avg'][0], zc['rain'][0])
    rain_hi = min(cp['rain_avg'][1], zc['rain'][1])
    if rain_lo > rain_hi:
        mid = (cp['rain_avg'][0] + cp['rain_avg'][1] + zc['rain'][0] + zc['rain'][1]) / 4
        rain_lo, rain_hi = mid - 10, mid + 10

    mean_temp = round(np.random.uniform(temp_lo, temp_hi), 2)
    rain_avg  = round(np.random.uniform(rain_lo, rain_hi), 2)

    return mean_temp, rain_avg


def generate_samples(crop: str) -> pd.DataFrame:
    """
    Generate SAMPLES_PER_CROP realistic samples for a crop.

    Key v3 improvements:
    - Zone-coupled climate (temp/rain bounded by zone's realistic range)
    - Soil type classification (sandy/loamy/clayey)
    - Macro-zone group (arid/humid/plains)
    """
    params = CROP_PARAMS[crop]
    records = []

    # Equal NPK tiers across all samples
    n_per_tier = SAMPLES_PER_CROP // 3
    npk_tiers = (['low'] * n_per_tier +
                 ['medium'] * n_per_tier +
                 ['high'] * (SAMPLES_PER_CROP - 2 * n_per_tier))
    np.random.shuffle(npk_tiers)

    for i in range(SAMPLES_PER_CROP):
        zone_id = int(np.random.choice(params['zones']))
        season = str(np.random.choice(params['season']))

        # Zone-coupled climate (v3 key improvement)
        mean_temp, rain_avg = gen_zone_coupled_climate(zone_id, crop)

        # Humidity: coupled to zone humidity character
        hum_lo, hum_hi = params['humidity']
        # Humid zones get +5-10% humidity boost, arid zones get -5%
        zone_group = get_zone_group(zone_id)
        if zone_group == 'humid':
            hum_lo = clamp(hum_lo + 5, 30, 95)
            hum_hi = clamp(hum_hi + 5, 40, 98)
        elif zone_group == 'arid':
            hum_lo = clamp(hum_lo - 5, 25, 80)
            hum_hi = clamp(hum_hi - 5, 35, 90)

        humidity = round(np.random.uniform(hum_lo, hum_hi), 1)
        pH = round(np.random.uniform(*params['pH']), 2)

        sand, silt, clay = gen_soil_for_zone(zone_id, crop)
        n, pv, k = gen_npk(crop, npk_tiers[i])

        # Soil type classification (v3 addition)
        soil_type = classify_soil_type(sand, silt, clay)

        records.append({
            'Crop':         crop,
            'Season':       season,
            'Agro_Zone':    zone_id,
            'Zone_Group':   zone_group,
            'mean_temp':    mean_temp,
            'rain_avg':     rain_avg,
            'Sand_pct':     sand,
            'Silt_pct':     silt,
            'Clay_pct':     clay,
            'Soil_Type':    soil_type,
            'N_kg_ha':      n,
            'P_kg_ha':      pv,
            'K_kg_ha':      k,
            'Humidity_pct': humidity,
            'pH':           pH,
        })

    return pd.DataFrame(records)


def validate_dataset(df: pd.DataFrame, tolerance: float = 2.0) -> pd.DataFrame:
    """Remove rows with invalid soil composition."""
    soil_sum = df['Sand_pct'] + df['Silt_pct'] + df['Clay_pct']
    valid = (soil_sum - 100.0).abs() <= tolerance
    dropped = (~valid).sum()
    if dropped > 0:
        print(f"  [Soil Validation] Removed {dropped} invalid rows (|sum-100| > {tolerance}%)")
    return df[valid].reset_index(drop=True)


def print_feature_stats(df: pd.DataFrame) -> None:
    """Print per-crop feature ranges to verify authentic data."""
    print("\n  Per-crop feature ranges:")
    cols = ['mean_temp', 'rain_avg', 'N_kg_ha', 'P_kg_ha', 'K_kg_ha', 'Humidity_pct', 'pH']
    for crop in sorted(df['Crop'].unique()):
        sub = df[df['Crop'] == crop]
        print(f"\n  [{crop}]")
        for col in cols:
            print(f"    {col:<18}: {sub[col].min():.1f} - {sub[col].max():.1f}  "
                  f"(mean={sub[col].mean():.1f})")
        # Soil type distribution
        st = sub['Soil_Type'].value_counts().to_dict()
        print(f"    {'Soil_Type':<18}: {st}")
        # Zone group distribution
        zg = sub['Zone_Group'].value_counts().to_dict()
        print(f"    {'Zone_Group':<18}: {zg}")


def main():
    np.random.seed(RANDOM_SEED)

    print("=" * 65)
    print("  BhoomiAI -- Authentic Dataset Builder v3")
    print("=" * 65)
    print("\n  v3 Improvements over v2:")
    print("  +- Zone-coupled climate generation (temp/rain bounded by zone)")
    print("  +- Soil type categorical flag (sandy/loamy/clayey)")
    print("  +- Macro-zone groups (arid/humid/plains)")
    print("  +- 400 samples per crop (1600 total)")
    print("  +- Zone-specific crop exclusions")

    dfs = []
    print()
    for crop in CROP_PARAMS:
        df_crop = generate_samples(crop)
        print(f"  [{crop:<10}] {len(df_crop)} samples | "
              f"Zones: {sorted(df_crop['Agro_Zone'].unique())} | "
              f"Seasons: {df_crop['Season'].unique().tolist()} | "
              f"Soil types: {df_crop['Soil_Type'].value_counts().to_dict()}")
        dfs.append(df_crop)

    df = pd.concat(dfs, ignore_index=True)
    df = validate_dataset(df)

    # Shuffle
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Realism checks
    rice_low_rain = df[(df['Crop'] == 'Rice') & (df['rain_avg'] < 80)]
    wheat_high_rain = df[(df['Crop'] == 'Wheat') & (df['rain_avg'] > 100)]
    sugar_desert = df[(df['Crop'] == 'Sugarcane') & (df['Agro_Zone'] == 13)]
    print(f"\n  [Realism Check]")
    print(f"  Rice rows with rain_avg < 80 cm  : {len(rice_low_rain)} (should be 0)")
    print(f"  Wheat rows with rain_avg > 100cm : {len(wheat_high_rain)} (should be 0)")
    print(f"  Sugarcane in desert zone 13      : {len(sugar_desert)} (should be 0)")

    os.makedirs('dataset', exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    # Summary
    print(f"\n  [Summary]")
    print(f"  Total samples      : {len(df)}")
    print(f"  Class distribution : {df['Crop'].value_counts().to_dict()}")
    print(f"  Seasons            : {df['Season'].value_counts().to_dict()}")
    print(f"  Zones covered      : {sorted(df['Agro_Zone'].unique())}")
    print(f"  Zone groups        : {df['Zone_Group'].value_counts().to_dict()}")
    print(f"  Soil types         : {df['Soil_Type'].value_counts().to_dict()}")
    soil_sum = df['Sand_pct'] + df['Silt_pct'] + df['Clay_pct']
    print(f"  Soil sum min/max   : {soil_sum.min():.2f}% / {soil_sum.max():.2f}%")

    print_feature_stats(df)

    print(f"\n  [OK] Dataset saved -> {OUTPUT_PATH}")
    return df


if __name__ == '__main__':
    main()
