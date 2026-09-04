#!/usr/bin/env python3
"""phase3b_lineage.py — Phase 3b: 双谱系机制对比 (startup vs run)
（研究方案 §5.4 / 手册: 双谱系 = 启动型首分钟突变? 运行型功率未衰减?）

两路证据:
 1) Stage2 家族判别(LightGBM P(startup|fault), 故障池 owner1-6 训练)的特征重要性 top-20
    → 哪些通道/统计量区分 startup 与 run
 2) 通道级早期画像: test 故障(startup/run) + 正常短事务 在 τ=2min 前缀的通道统计量对比
    → 启动型是否首分钟 v/a/p 突变; 运行型是否功率持续未衰减

输出 (docs/):
  phase3b_lineage.json — 特征重要性 top + 三组通道画像均值
"""
import pandas as pd, numpy as np, json, os, warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data'); OUT = os.path.join(BASE, 'docs')
SCHEMA = json.load(open(os.path.join(DATA, 'prefix_features_v1.json')))
FEAT_COLS = SCHEMA['feature_cols']

feat = pd.read_parquet(os.path.join(DATA, 'prefix_feats_v1.parquet'),
                       columns=FEAT_COLS + ['transaction_id', 'prefix_type', 'prefix_val',
                                            'owner', 'family', 'label'])
CH = {'chargingv': '电压V', 'charginga': '电流A', 'out_power': '功率kW',
      'charging_gun_temperature1': '枪温1℃', 'charging_gun_temperature2': '枪温2℃', 'current_soc': 'SOC%'}

# ---- 1) Stage2 家族判别特征重要性 (τ=3, 跨站 owner1-6 故障池) ----
sub = feat[(feat['prefix_type'] == 'time') & (feat['prefix_val'] == 3)]
tr = sub[sub['owner'].isin([f'Sheet{i}' for i in range(1, 7)])]
tr_f = tr[tr['family'].isin(['startup', 'run'])]
y2 = (tr_f['family'] == 'startup').astype(int).values
m2 = lgb.LGBMClassifier(objective='binary', learning_rate=0.05, num_leaves=31,
                        min_child_samples=30, n_estimators=400, random_state=0,
                        n_jobs=4, verbosity=-1, scale_pos_weight=float((y2 == 0).sum()) / max(1, int(y2.sum())))
m2.fit(tr_f[FEAT_COLS].values, y2)
imp = pd.Series(m2.feature_importances_, index=FEAT_COLS).sort_values(ascending=False)
print('=== Stage2 家族判别(startup vs run)特征重要性 top-20 (τ=3, 跨站) ===')
for i, (f, v) in enumerate(imp.head(20).items()):
    base = f.rsplit('_', 1)[0] if f.rsplit('_', 1)[0] in CH else f.split('_')[0]
    ch = CH.get(f.rsplit('_', 1)[0], '?') if f.rsplit('_', 1)[0] in CH else CH.get(f.split('_')[0], '?')
    stat = f.rsplit('_', 1)[1] if f.rsplit('_', 1)[0] in CH else '?'
    print(f'  {i+1:>2}. {f:<32} {v:>4}  ({ch} · {stat})')
imp_top = {k: int(v) for k, v in imp.head(20).items()}

# ---- 2) 通道级早期画像 (τ=2min 前缀, test owner7-8) ----
sub2 = feat[(feat['prefix_type'] == 'time') & (feat['prefix_val'] == 2)]
te = sub2[sub2['owner'].isin(['Sheet7', 'Sheet8'])].copy()
te['grp'] = np.where(te['label'] == 0, 'normal',
             np.where(te['family'] == 'startup', 'startup', 'run'))
# 画像列: 每通道的 first/last/slope + 功率主动比 + SOC delta + 枪温升速
PORTRAIT = ['chargingv_first', 'chargingv_last', 'chargingv_slope',
            'charginga_first', 'charginga_last', 'charginga_slope',
            'out_power_first', 'out_power_last', 'out_power_slope',
            'current_soc_first', 'current_soc_last', 'soc_delta',
            'power_active_ratio', 'power_peak_pos', 'gunT1_rise']
prof = te.groupby('grp')[PORTRAIT].mean().round(3)
print('\n=== 通道级早期画像 (τ=2min 前缀, test) ===')
print(f'n: {te.groupby("grp").size().to_dict()}')
print(prof.T.to_string())

json.dump({'stage2_imp_top20': imp_top,
           'portrait_cols': PORTRAIT,
           'portrait_mean': {g: {c: float(v) for c, v in prof.loc[g].items()} for g in prof.index},
           'group_n': {g: int(n) for g, n in te.groupby('grp').size().items()}},
          open(os.path.join(OUT, 'phase3b_lineage.json'), 'w'), indent=2, ensure_ascii=False)
print(f'\n结果已存: docs/phase3b_lineage.json')
