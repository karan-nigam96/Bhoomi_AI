from fertilizer_functions import load_fertilizer_models, predict_fertilizer
import warnings
warnings.filterwarnings('ignore')

m = load_fertilizer_models()
r = predict_fertilizer(m, crop='Wheat', zone_id=6, season=1,
    n_soil=195, p_soil=28, k_soil=185, ph=7.2, irrigation=1,
    variety='HYV', organic='None', prev_crop='Cereal', farm_size=2.0)

print('NPK/ha:', r['npk_per_ha'])
print('NPK total:', r['npk_total'])
print('Flags:', r['special_treatments'])
print('Stages:', len(r['stage_schedule']))
for s in r['stage_schedule']:
    print(f"  {s['stage']} ({s['timing']}): N={s['N_kg_ha']}  P={s['P_kg_ha']}  K={s['K_kg_ha']}")
