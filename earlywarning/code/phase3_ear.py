#!/usr/bin/env python3
"""phase3_ear.py — Phase 3a: EAR(最早可预警前缀)分析, 主报 LightGBM 结构A Stage1
（研究方案 §5.4 / gate_report_phase2 终版裁决 C: 树为主报）

EAR 定义: 对每个 test(owner7-8)故障事务, 在"该事务可用的 τ 前缀"上逐 τ 打分,
首个 P(终止) ≥ 该 τ 判定阈值 的 τ = EAR。不可用 τ(=前缀行 <2)不计入,
从而区分「模型未预警」vs「数据不允许预警」(插枪后起充晚/采样稀疏)。

输出 (docs/):
  phase3_ear_results.json — 汇总: 逐 τ 可用性 + 事务级 EAR 分布(startup/run)
  phase3_ear_by_txn.csv   — 每事务明细: tid, family, dur, 可用τ集, 各τ概率, EAR
"""
import pandas as pd, numpy as np, json, os, warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data'); OUT = os.path.join(BASE, 'docs')
SCHEMA = json.load(open(os.path.join(DATA, 'prefix_features_v1.json')))
FEAT_COLS = SCHEMA['feature_cols']
TAUS = [1, 2, 3, 5, 10, 20]

feat = pd.read_parquet(os.path.join(DATA, 'prefix_feats_v1.parquet'),
                       columns=FEAT_COLS + ['transaction_id', 'prefix_type', 'prefix_val',
                                            'n_prefix_rows', 'owner', 'family', 'label'])
sub = feat[feat['prefix_type'] == 'time'].copy()
print(f'time 行: {len(sub)} | τ 分布: {sub.groupby("prefix_val").size().to_dict()}')

# ---- 1) 逐 τ 训练 LightGBM(跨站 owner1-6 → 7-8, 同 structure_ab/ensemble_eval 协议) ----
#     val 校准: owner1-6 内 stratify 20% val(seed42, 同 build_seq_tensors), 找 precision≥0.90 的最高 recall 阈值
from sklearn.metrics import precision_recall_curve, precision_score
models, ths = {}, {}
for tau in TAUS:
    s = sub[sub['prefix_val'] == tau]
    tr_all = s[s['owner'].isin([f'Sheet{i}' for i in range(1, 7)])]
    te = s[s['owner'].isin(['Sheet7', 'Sheet8'])]
    if len(te) == 0 or len(np.unique(tr_all['label'])) < 2:
        print(f'  τ={tau}: 跳过 (train={len(tr_all)} test={len(te)})'); continue
    # stratify 切 val(seed42)
    from sklearn.model_selection import train_test_split
    tr, va = train_test_split(tr_all, test_size=0.2, random_state=42, stratify=tr_all['label'])
    pos_w = float((tr['label'] == 0).sum()) / max(1, int((tr['label'] == 1).sum()))
    m = lgb.LGBMClassifier(objective='binary', learning_rate=0.05, num_leaves=31,
                           min_child_samples=30, n_estimators=400, random_state=0,
                           n_jobs=4, verbosity=-1, scale_pos_weight=pos_w)
    m.fit(tr[FEAT_COLS].values, tr['label'].values)
    # val 校准阈值: precision≥0.90 的最高 recall 对应阈值
    Pva = m.predict_proba(va[FEAT_COLS].values)[:, 1]
    prec, rec, th = precision_recall_curve(va['label'].values, Pva)
    cand = [(t, r) for p, r, t in zip(prec, rec, np.concatenate([th, [1.0]])) if p >= 0.90]
    th_tau = max(cand, key=lambda x: x[1])[0] if cand else 0.5
    Pte = m.predict_proba(te[FEAT_COLS].values)[:, 1]
    models[tau] = (m, te.copy(), Pte)
    ths[tau] = th_tau
    n_f = int(te['label'].sum())
    # val precision/recall at th_tau
    vp = (Pva >= th_tau).astype(int)
    vprec = precision_score(va['label'].values, vp, zero_division=0)
    vrec = float((vp[va['label'].values == 1]).mean()) if (va['label'].values == 1).any() else 0
    print(f'  τ={tau:>2}: train={len(tr)} val={len(va)} test={len(te)} (故障 {n_f}) | '
          f'thr={th_tau:.2f} valPrec={vprec:.2f} valRec={vrec:.2f}')

# ---- 2) 事务级 EAR 组装 ----
# 对每个故障事务, 收集其可用 τ → P
recs = []
for tau in TAUS:
    if tau not in models: continue
    _, te, Pte = models[tau]
    mk = te['label'] == 1
    for tid, p, fam in zip(te['transaction_id'][mk], Pte[mk], te['family'][mk]):
        recs.append({'transaction_id': tid, 'family': fam, 'tau': tau, 'P': float(p)})
ear_df = pd.DataFrame(recs)
print(f'\n故障行(事务×可用τ): {len(ear_df)}')

# 每个事务: 可用 τ 排序列表 + 各 τ 概率
rows = []
for tid, g in ear_df.groupby('transaction_id'):
    g = g.sort_values('tau')
    taus_avail = g['tau'].tolist()
    Ps = dict(zip(g['tau'], g['P']))
    fam = g['family'].iloc[0]
    # EAR = 可用 τ 中首个 P≥thr(该 τ 高精度阈值); 若全部 <thr → 未预警(None)
    ear = None; ear_p = None
    for t in taus_avail:
        if Ps[t] >= ths.get(t, 0.5):
            ear, ear_p = t, Ps[t]; break
    rows.append({'transaction_id': tid, 'family': fam,
                 'n_tau_avail': len(taus_avail), 'taus_avail': taus_avail,
                 'ear': ear, 'ear_P': ear_p,
                 **{f'P_tau{t}': Ps.get(t, np.nan) for t in TAUS}})
res = pd.DataFrame(rows)
# 注入 dur (来自固化数据集)
d = pd.read_parquet(os.path.join(DATA, 'prefix_dataset_full.parquet'), columns=['transaction_id', 'dur_min'])
res = res.merge(d, on='transaction_id', how='left')
print(f'故障事务: {len(res)} | startup={int((res.family=="startup").sum())} run={int((res.family=="run").sum())}')

# ---- 3) 汇总统计 ----
# 预警提前量 lead = dur - EAR (运维响应窗口)。run 型 dur≥30 → EAR=2 时 lead≥28min
res['lead_min'] = res['dur_min'] - res['ear']

def summarize(fam):
    g = res[res['family'] == fam]
    n = len(g)
    alerted = g['ear'].notna()
    n_alert = int(alerted.sum())
    ears = g.loc[alerted, 'ear']
    leads = g.loc[alerted, 'lead_min']
    return {'n': n, 'n_alerted': n_alert, 'alert_rate': round(n_alert / n, 4),
            'ear_median': float(np.median(ears)) if len(ears) else None,
            'ear_mean': round(float(ears.mean()), 2) if len(ears) else None,
            'ear_dist': {str(int(t)): int((ears == t).sum()) for t in sorted(ears.unique())} if len(ears) else {},
            'lead_median': round(float(np.median(leads)), 1) if len(leads) else None,
            'lead_mean': round(float(leads.mean()), 1) if len(leads) else None,
            'lead_p25': round(float(np.percentile(leads, 25)), 1) if len(leads) else None,
            'lead_p75': round(float(np.percentile(leads, 75)), 1) if len(leads) else None,
            'n_unalerted': n - n_alert}

summary = {f: summarize(f) for f in ['startup', 'run']}
print('\n=== EAR 汇总(高精度校准: val precision≥0.90) ===')
for f, s in summary.items():
    print(f'  {f:<8} n={s["n"]} 预警 {s["n_alerted"]} ({s["alert_rate"]*100:.1f}%) '
          f'EAR中位={s["ear_median"]}min | 提前量中位={s["lead_median"]}min '
          f'(p25={s["lead_p25"]} p75={s["lead_p75"]}) | 分布={s["ear_dist"]} | 未预警 {s["n_unalerted"]}')

json.dump({'summary': summary,
           'note': ('EAR=可用τ前缀中首个P≥该τ高精度阈值的τ(val precision≥0.90 校准); '
                    'lead_min=dur-EAR 为预警提前量; 不可用τ(前缀行<2,插枪晚/采样疏)不计入'),
           'thr_by_tau': {str(t): round(ths.get(t, 0.5), 3) for t in TAUS if t in models},
           'n_fault_txn': int(len(res)), 'n_rows': int(len(ear_df))},
          open(os.path.join(OUT, 'phase3_ear_results.json'), 'w'), indent=2, ensure_ascii=False)
res.to_csv(os.path.join(OUT, 'phase3_ear_by_txn.csv'), index=False)
print(f'\n结果已存: docs/phase3_ear_results.json + phase3_ear_by_txn.csv')
