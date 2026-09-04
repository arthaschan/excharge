#!/usr/bin/env python3
"""ensemble_eval.py — Phase 2 方案A: Token-Attn 7-seed 概率集成 vs LightGBM 对照
（gate_report_phase2_probe §3 选项A / 手册 R6: 深度成绩 = ≥5 seed 概率集成 + 统计检验）

输入（docs/）:
  prefix_tokenattn_tau{τ}_s{0..6}_prob.npy —— 各 seed 的 test 概率（长度 = n_test, 与 seq_tensors 对齐）
对照:
  同协议重训 LightGBM 结构A Stage1（跨站 owner1-6→7-8, seed0 与 structure_ab 同超参）
输出（docs/）:
  ensemble7_tau{τ}_results.json —— 集成 PR-AUC/AUC/逐 owner + bootstrap 差分布 p 值

用法: TAU=3 [B=2000] python ensemble_eval.py
"""
import numpy as np, pickle, json, os, sys
from sklearn.metrics import roc_auc_score, average_precision_score

TAU = int(os.environ.get('TAU', 3))
B = int(os.environ.get('B', 2000))          # bootstrap 次数
SEEDS = list(range(7))                       # 0..6
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data'); OUT = os.path.join(BASE, 'docs')

# 1) 载入 test 真值 + owner
D = pickle.load(open(f'{DATA}/seq_tensors_tau{TAU}.pkl', 'rb'))
yte = np.asarray(D['y_te']); owner_te = np.asarray(D['owner_te'])
n = len(yte)
print(f'[τ={TAU}] test n={n} 故障 {int(yte.sum())} ({yte.mean()*100:.2f}%) | Sheet7 {int((owner_te=="Sheet7").sum())} Sheet8 {int((owner_te=="Sheet8").sum())}')

# 2) 载入 7 个 seed prob 并平均（概率集成）
probs = []
for s in SEEDS:
    p = np.load(f'{OUT}/prefix_tokenattn_tau{TAU}_s{s}_prob.npy')
    assert len(p) == n, f'seed{s} prob len {len(p)} != {n}'
    probs.append(p)
    print(f'  seed{s}: 单模型 PR-AUC={average_precision_score(yte, p):.4f}')
P_ens = np.mean(probs, axis=0)
ens_pr = average_precision_score(yte, P_ens)
ens_auc = roc_auc_score(yte, P_ens)
print(f'  7-seed 集成: PR-AUC={ens_pr:.4f} AUC={ens_auc:.4f}')

# 3) 同协议 LightGBM（跨站结构A Stage1, seed0, 与 structure_ab cross 同超参）
import pandas as pd
import lightgbm as lgb
SCHEMA = json.load(open(f'{DATA}/prefix_features_v1.json'))
FEAT_COLS = SCHEMA['feature_cols']
feat = pd.read_parquet(f'{DATA}/prefix_feats_v1.parquet',
                       columns=FEAT_COLS + ['owner', 'label', 'prefix_type', 'prefix_val', 'transaction_id'])
sub = feat[(feat['prefix_type'] == 'time') & (feat['prefix_val'] == TAU)].reset_index(drop=True)
tr = sub[sub['owner'].isin([f'Sheet{i}' for i in range(1, 7)])]
te = sub[sub['owner'].isin(['Sheet7', 'Sheet8'])]
pos_w = float((tr['label'] == 0).sum()) / max(1, int((tr['label'] == 1).sum()))
m = lgb.LGBMClassifier(objective='binary', learning_rate=0.05, num_leaves=31,
                       min_child_samples=30, n_estimators=400, random_state=0,
                       n_jobs=4, verbosity=-1, scale_pos_weight=pos_w)
m.fit(tr[FEAT_COLS].values, tr['label'].values)
P_lgb = m.predict_proba(te[FEAT_COLS].values)[:, 1]
lgb_pr = average_precision_score(te['label'].values, P_lgb)
print(f'  LightGBM 同协议(跨站A/seed0): PR-AUC={lgb_pr:.4f} (structure_ab 3-seed mean={json.load(open(f"{OUT}/structure_ab_results.json"))["by_tau"][str(TAU)]["cross"]["A"]["ap_term"]["mean"]})')
# 对齐顺序: seq_tensors 的 test 顺序 = owner7-8 在 cohort 中的顺序。需验证与 te 顺序一致。
tids_seq = np.asarray(D['tids_te'])
tids_te = te['transaction_id'].values
assert len(tids_seq) == len(tids_te), (len(tids_seq), len(tids_te))
if not np.array_equal(tids_seq, tids_te):
    # 顺序可能不同 → 按 tid 重排 LightGBM prob 到 seq 顺序
    idx = {tid: i for i, tid in enumerate(tids_te)}
    order = np.array([idx[t] for t in tids_seq])
    P_lgb = P_lgb[order]
    print('  ⚠️ test 顺序不一致 → 已按 transaction_id 重排 LightGBM prob')
assert len(P_lgb) == n
y_lgb = te['label'].values
if not np.array_equal(tids_seq, tids_te):
    y_lgb = y_lgb[order]

# 4) bootstrap 差分布（paired, 逐样本重采样）
rng = np.random.default_rng(42)
diffs = np.zeros(B); ens_prs = np.zeros(B)
for b in range(B):
    idx = rng.integers(0, n, n)
    ens_prs[b] = average_precision_score(yte[idx], P_ens[idx])
    diffs[b] = ens_prs[b] - average_precision_score(y_lgb[idx], P_lgb[idx])
p_gt = float((diffs > 0).mean())                     # P(深度集成 > LightGBM)
p_lt = float((diffs < 0).mean())
ci = np.percentile(ens_prs, [2.5, 97.5])
print(f'\n=== bootstrap (B={B}) ===')
print(f'  深度集成 PR-AUC 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}] | 均值 {ens_prs.mean():.4f}')
print(f'  Δ(深度-树): mean={diffs.mean():+.4f} | P(深度>树)={p_gt:.4f} | P(深度<树)={p_lt:.4f}')
if p_gt > 0.95:
    verdict = '深度集成显著超越树基线 (p<0.05)'
elif p_gt > 0.05:
    verdict = '深度与树无显著差异 (p>0.05)'
else:
    verdict = '深度显著低于树基线'
print(f'  判定: {verdict}')

# 5) 逐 owner
res = {'tau': TAU, 'n_test': n, 'n_seeds': len(SEEDS), 'bootstrap_B': B,
       'deep_ensemble': {'PR-AUC': round(float(ens_pr), 4), 'AUC': round(float(ens_auc), 4),
                         'CI95': [round(ci[0], 4), round(ci[1], 4)]},
       'lightgbm_same_protocol': {'PR-AUC': round(float(lgb_pr), 4)},
       'delta_mean': round(float(diffs.mean()), 4),
       'p_deep_gt_tree': round(float(p_gt), 4), 'verdict': verdict,
       'by_owner': {}}
for ow in ['Sheet7', 'Sheet8']:
    mk = owner_te == ow
    if mk.sum() >= 10 and len(np.unique(yte[mk])) == 2:
        res['by_owner'][ow] = {'n': int(mk.sum()), 'fault': int(yte[mk].sum()),
                               'deep_pr': round(float(average_precision_score(yte[mk], P_ens[mk])), 4)}
print('\n逐 owner 深度集成 PR-AUC:', {k: v['deep_pr'] for k, v in res['by_owner'].items()})

json.dump(res, open(f'{OUT}/ensemble7_tau{TAU}_results.json', 'w'), indent=2, ensure_ascii=False)
print(f'\n结果已保存: docs/ensemble7_tau{TAU}_results.json')
