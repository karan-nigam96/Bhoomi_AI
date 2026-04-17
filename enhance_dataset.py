"""
enhance_dataset.py — Add Season and Agro-climatic Zone features to dataset
Based on ICAR crop suitability guidelines and Planning Commission agro-climatic zones
"""

import pandas as pd
import numpy as np
import json

# Load existing dataset
df = pd.read_csv('dataset/crop_train.csv')

# Load zone information
with open('dataset/agro_zones.json', 'r', encoding='utf-8') as f:
    zone_data = json.load(f)

print(f"Original dataset: {len(df)} records")
print(f"Crops: {df['Crop'].value_counts().to_dict()}")

# Crop suitability by season based on ICAR guidelines
crop_season_suitability = {
    'Wheat': {
        'season': 'Rabi',  # Primary Rabi crop
        'season_code': 0,
        'suitable_zones': [0, 2, 3, 4, 5, 7, 8, 12, 13],  # Northern, plains, plateau regions
        'zone_weights': {
            4: 0.23, 5: 0.23,  # Upper & Trans-Gangetic (highest suitability)
            3: 0.18, 2: 0.14,  # Middle & Lower Gangetic
            7: 0.09, 8: 0.05,  # Central & Western Plateau
            12: 0.05, 13: 0.02, 0: 0.01  # Gujarat, Western Dry, Himalayan (limited)
        }
    },
    'Rice': {
        'season': 'Kharif',  # Primary Kharif crop
        'season_code': 1,
        'suitable_zones': [1, 2, 3, 4, 5, 6, 9, 10, 11],  # High rainfall zones
        'zone_weights': {
            2: 0.22, 3: 0.22,  # Gangetic plains (highest suitability)
            10: 0.18, 6: 0.15,  # East Coast, Eastern Plateau
            9: 0.10, 11: 0.08,  # Southern Plateau, West Coast
            1: 0.03, 4: 0.01, 5: 0.01  # Eastern Himalayan, Upper/Trans Gangetic (limited)
        }
    },
    'Maize': {
        'season': 'Kharif',  # Primarily Kharif, also Rabi in some zones
        'season_code': 1,
        'suitable_zones': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12],  # Very versatile crop
        'zone_weights': {
            5: 0.18, 4: 0.15,  # Trans & Upper Gangetic
            3: 0.12, 7: 0.12,  # Middle Gangetic, Central Plateau
            8: 0.10, 9: 0.10,  # Western & Southern Plateau
            6: 0.08, 2: 0.07,  # Eastern Plateau, Lower Gangetic
            12: 0.05, 1: 0.02, 0: 0.01  # Gujarat, Himalayan regions
        }
    },
    'Sugarcane': {
        'season': 'Kharif',  # Annual crop, primarily planted Kharif season
        'season_code': 1,
        'suitable_zones': [2, 3, 4, 5, 8, 9, 10, 12],  # Tropical/subtropical zones
        'zone_weights': {
            4: 0.25, 5: 0.20,  # Upper & Trans-Gangetic (highest)
            3: 0.18, 10: 0.15,  # Middle Gangetic, East Coast
            9: 0.10, 2: 0.07,  # Southern Plateau, Lower Gangetic
            8: 0.03, 12: 0.02  # Western Plateau, Gujarat (limited)
        }
    }
}

# Create enhanced dataset
enhanced_records = []

for idx, row in df.iterrows():
    crop = row['Crop']
    crop_info = crop_season_suitability[crop]
    
    # Assign season based on crop
    season = crop_info['season']
    season_code = crop_info['season_code']
    
    # Select agro-climatic zone based on crop suitability
    suitable_zones = crop_info['suitable_zones']
    zone_weights_dict = crop_info['zone_weights']
    
    # Weighted random selection of zone
    zones = list(zone_weights_dict.keys())
    weights = list(zone_weights_dict.values())
    selected_zone = np.random.choice(zones, p=weights)
    
    # Add new columns to record
    new_record = row.to_dict()
    new_record['Season'] = season
    new_record['Season_code'] = season_code
    new_record['Agro_Zone'] = selected_zone
    new_record['Zone_name'] = zone_data['zones'][selected_zone]['name']
    
    enhanced_records.append(new_record)

# Create DataFrame
enhanced_df = pd.DataFrame(enhanced_records)

# Reorder columns: Crop first, then all original features, then Season and Zone
cols = ['Crop'] + [col for col in df.columns if col != 'Crop'] + ['Season', 'Season_code', 'Agro_Zone', 'Zone_name']
enhanced_df = enhanced_df[cols]

print(f"\n[SUCCESS] Enhanced dataset: {len(enhanced_df)} records")
print(f"\nSeason distribution:")
print(enhanced_df['Season'].value_counts())
print(f"\nZone distribution:")
print(enhanced_df['Agro_Zone'].value_counts().sort_index())
print(f"\nCrop-Season combinations:")
print(enhanced_df.groupby(['Crop', 'Season']).size())

# Save enhanced dataset
enhanced_df.to_csv('dataset/crop_train_enhanced.csv', index=False)
print(f"\n[SAVED] Saved to: dataset/crop_train_enhanced.csv")

# Create feature list for model training (without Zone_name for ML)
feature_df = enhanced_df.drop(columns=['Zone_name'])
feature_df.to_csv('dataset/crop_train_ml.csv', index=False)
print(f"[SAVED] Saved ML-ready dataset to: dataset/crop_train_ml.csv")

# Print summary statistics
print("\n" + "="*60)
print("DATASET SUMMARY")
print("="*60)
print(f"Total Records: {len(enhanced_df)}")
print(f"Features: {len(feature_df.columns) - 1} (excluding Crop label)")
print(f"  - Original soil/climate features: 16")
print(f"  - New features: Season_code, Agro_Zone")
print(f"Classes: {feature_df['Crop'].nunique()} crops")
print(f"\nRecords per crop:")
for crop in sorted(feature_df['Crop'].unique()):
    count = len(feature_df[feature_df['Crop'] == crop])
    print(f"  {crop}: {count}")
print("="*60)
