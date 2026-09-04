#!/usr/bin/env python3
"""P7 / 候选 E — 监督 vs 无监督 可解释异常检测系统同口径对比，CPU

背景（承接 HANDOFF 第六节 E / IFAC 2025）：
  IFAC 2025 用无监督 Isolation Forest + DIFFI（深度隔离森林特征重要性）做 EV 充电站异常检测，
  我们主线用监督（LightGBM / Token-Attn）。这里做「同口径」系统对比，作差异化 + 诚实性贡献。

同口径 = 同一数据（52 维前缀特征）、同一切分（owner1-6 训 / owner7-8 测）、同一指标：
  1. 前缀级检测质量：每 τ 的 PR-AUC（test 上，故障 vs normal 排序）。
  2. 事务级 EAR：预警率 @ val precision≥0.90 校准、EAR 中位、lead 中位。
  3. 可解释性：监督=LightGBM gain；无监督=DIFFI 式「深度加权隔离特征重要性」；
     对比两者特征排序（Spearman）与 top 特征重叠。

公平性说明：
  - LightGBM 用标签训练（监督）；Isolation Forest 在训练集上**不用标签**训练（无监督，fit 全部 train 数据）。
  - 两者的 val 阈值校准都用了 val 标签（这是阈值选择，非训练，与监督模型的校准协议一致、且是评估无监督 AD 的标准做法）。
  - IF 用原始 anomaly score（-score_samples，越大越异常），contamination 不影响打分、只影响 predict 阈值。

DIFFI 式重要性（轻量重实现，公式见下）：
  imp[f] = Σ_{trees} Σ_{内部分裂节点 n: feature(n)=f}  ( n_samples(n)/n_total ) / depth(n)
  depth(n) 从 1 起算。这是「隔离深度加权」思路的 IF 版（对应树的 gain），
  非 DIFFI-RX 逐字复刻，命名上称「DIFFI 式」。

产出：journal/docs/p7_supervised_vs_unsupervised.json
"""
import pandas as pd, numpy as np, json, os, warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score
from scipy.stats import spearmanr

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE)
EARLY = os.path.join(ROOT, 'earlywarning')
DATA = os.path.join(EARLY, 'data')
OUT = os.path.join(BASE, 'docs')
os.makedirs(OUT, exist_ok=True)
SEED = 42
TAUS = [1, 2, 3, 5, 10, 20]
LINEAGES = ['startup', 'run']

SCHEMA = json.load(open(os.path.join(DATA, 'prefix_features_v1.json')))
FEAT_COLS = SCHEMA['feature_cols']

feat = pd.read_parquet(os.path.join(DATA, 'prefix_feats_v1.parquet'),
                       columns=FEAT_COLS + ['transaction_id', 'prefix_type', 'prefix_val',
                                            'n_prefix_rows', 'owner', 'family', 'label'])
sub = feat[feat['prefix_type'] == 'time'].copy()
meta = pd.read_parquet(os.path.join(DATA, 'prefix_dataset_full.parquet'), columns=['transaction_id', 'dur_min'])
sub = sub.merge(meta, on='transaction_id', how='left')
train_owners = [f'Sheet{i}' for i in range(1, 7)]
test_owners = ['Sheet7', 'Sheet8']
print(f'[load] time rows {len(sub)}', flush=True)


def calibrate_threshold(yval, score, target=0.90):
    prec, rec, th = precision_recall_curve(yval, score)
    cand = [(t, r) for p, r, t in zip(prec, rec, np.concatenate([th, [1.0]])) if p >= target]
    if cand:
        return float(max(cand, key=lambda x: x[1])[0])
    return float(np.inf)  # 无法达到 0.90 → 不触发


def if_diffi_importance(model, X):
    """DIFFI 式深度加权隔离特征重要性。X 为训练集特征（用于 n_samples 归一化）。"""
    n_feat = X.shape[1]
    n_total = len(X)
    imp = np.zeros(n_feat, dtype=float)
    for tree in model.estimators_:
        t = tree.tree_
        try:
            depths = t.compute_node_depths()
        except Exception:
            depths = np.zeros(t.node_count, dtype=int)
        for node_id in range(t.node_count):
            if t.children_left[node_id] == t.children_right[node_id]:
                continue  # 叶子
            f = t.feature[node_id]
            ns = t.n_node_samples[node_id]
            imp[f] += (ns / n_total) / (depths[node_id] + 1.0)
    return imp


def assemble_ear(score_rows, thr_by_tau):
    """score_rows: DataFrame[transaction_id, family, tau, score]。返回 (ear_df, per_family_summary)。"""
    rows = []
    for tid, g in score_rows.groupby('transaction_id'):
        g = g.sort_values('tau')
        fam = g['family'].iloc[0]
        ear = None
        for t in g['tau']:
            if t in thr_by_tau and g.loc[g['tau'] == t, 'score'].iloc[0] >= thr_by_tau[t]:
                ear = t
                break
        rows.append({'transaction_id': tid, 'family': fam, 'ear': ear})
    res = pd.DataFrame(rows).merge(meta, on='transaction_id', how='left')
    res['lead_min'] = res['dur_min'] - res['ear']
    summary = {}
    for L in LINEAGES:
        g = res[res['family'] == L]
        n = len(g)
        alerted = g['ear'].notna()
        n_alert = int(alerted.sum())
        ears = g.loc[alerted, 'ear']
        leads = g.loc[alerted, 'lead_min']
        summary[L] = {'n': n, 'n_alerted': n_alert, 'alert_rate': round(n_alert / n, 4) if n else None,
                      'ear_median': float(np.median(ears)) if len(ears) else None,
                      'lead_median': round(float(np.median(leads)), 1) if len(leads) else None}
    return res, summary


# ---------- 主循环 ----------
per_tau = {}
lgb_ear_rows, if_ear_rows, if_norm_ear_rows = [], [], []
for tau in TAUS:
    s = sub[sub['prefix_val'] == tau]
    tr_s = s[s['owner'].isin(train_owners)]
    te_s = s[s['owner'].isin(test_owners)]
    if len(te_s) == 0 or len(np.unique(tr_s['label'])) < 2:
        continue
    trs, vas = train_test_split(tr_s, test_size=0.2, random_state=42, stratify=tr_s['label'])
    Xtr, ytr = trs[FEAT_COLS].values, trs['label'].values
    Xva, yva = vas[FEAT_COLS].values, vas['label'].values
    Xte, yte = te_s[FEAT_COLS].values, te_s['label'].values

    # ---- 监督 LightGBM（全球：全部故障 vs normal）----
    pos_w = float((ytr == 0).sum()) / max(1, int((ytr == 1).sum()))
    lgb_m = lgb.LGBMClassifier(objective='binary', learning_rate=0.05, num_leaves=31,
                               min_child_samples=30, n_estimators=400, random_state=0,
                               n_jobs=4, verbosity=-1, scale_pos_weight=pos_w)
    lgb_m.fit(Xtr, ytr)
    lgb_score_va = lgb_m.predict_proba(Xva)[:, 1]
    lgb_score_te = lgb_m.predict_proba(Xte)[:, 1]
    lgb_thr = calibrate_threshold(yva, lgb_score_va)
    lgb_pr = float(average_precision_score(yte, lgb_score_te))
    lgb_gain = lgb_m.booster_.feature_importance(importance_type='gain')

    # ---- 无监督 Isolation Forest（fit 全部 train，不用标签）----
    if_m = IsolationForest(n_estimators=200, max_samples='auto', random_state=0, n_jobs=4, verbose=0)
    if_m.fit(tr_s[FEAT_COLS].values)
    if_score_va = -if_m.score_samples(Xva)   # 越大越异常
    if_score_te = -if_m.score_samples(Xte)
    if_thr = calibrate_threshold(yva, if_score_va)
    if_pr = float(average_precision_score(yte, if_score_te))
    if_imp = if_diffi_importance(if_m, tr_s[FEAT_COLS].values)

    # ---- 稳健性变体：IF 仅 fit normal（半监督对照，堵「训练集被故障污染」的质疑）----
    tr_norm = tr_s[tr_s['label'] == 0][FEAT_COLS].values
    if_norm_m = IsolationForest(n_estimators=200, max_samples='auto', random_state=0, n_jobs=4, verbose=0)
    if_norm_m.fit(tr_norm)
    if_norm_score_va = -if_norm_m.score_samples(Xva)
    if_norm_score_te = -if_norm_m.score_samples(Xte)
    if_norm_thr = calibrate_threshold(yva, if_norm_score_va)
    if_norm_pr = float(average_precision_score(yte, if_norm_score_te))

    # ---- 可解释性对比 ----
    lgb_rank = np.argsort(-lgb_gain)
    if_rank = np.argsort(-if_imp)
    rho, pval = spearmanr(lgb_gain, if_imp)
    topk = 10
    lgb_top = [FEAT_COLS[i] for i in lgb_rank[:topk]]
    if_top = [FEAT_COLS[i] for i in if_rank[:topk]]
    overlap = len(set(lgb_top) & set(if_top))

    per_tau[tau] = {
        'n_train': int(len(tr_s)), 'n_test': int(len(te_s)), 'n_test_fault': int(yte.sum()),
        'lgb': {'pr_auc': round(lgb_pr, 4), 'thr': round(lgb_thr, 4),
                'val_prec': round(precision_score(yva, (lgb_score_va >= lgb_thr).astype(int), zero_division=0), 4) if np.isfinite(lgb_thr) else None,
                'val_rec': round(recall_score(yva, (lgb_score_va >= lgb_thr).astype(int), zero_division=0), 4) if np.isfinite(lgb_thr) else None,
                'top_gain': lgb_top[:5]},
        'iforest': {'pr_auc': round(if_pr, 4), 'thr': round(if_thr, 4),
                    'val_prec': round(precision_score(yva, (if_score_va >= if_thr).astype(int), zero_division=0), 4) if np.isfinite(if_thr) else None,
                    'val_rec': round(recall_score(yva, (if_score_va >= if_thr).astype(int), zero_division=0), 4) if np.isfinite(if_thr) else None,
                    'top_diffi': if_top[:5]},
        'iforest_normal_only': {'pr_auc': round(if_norm_pr, 4), 'thr': round(if_norm_thr, 4),
                                'val_prec': round(precision_score(yva, (if_norm_score_va >= if_norm_thr).astype(int), zero_division=0), 4) if np.isfinite(if_norm_thr) else None,
                                'val_rec': round(recall_score(yva, (if_norm_score_va >= if_norm_thr).astype(int), zero_division=0), 4) if np.isfinite(if_norm_thr) else None},
        'interp': {'spearman_rho': round(float(rho), 4), 'spearman_p': round(float(pval), 4),
                   'top10_overlap': overlap},
    }
    # EAR 行（test 故障）
    mk = te_s['label'] == 1
    for tid, fam, lsc, isc, isnc in zip(te_s['transaction_id'][mk], te_s['family'][mk],
                                        lgb_score_te[mk], if_score_te[mk], if_norm_score_te[mk]):
        lgb_ear_rows.append({'transaction_id': tid, 'family': fam, 'tau': tau, 'score': float(lsc)})
        if_ear_rows.append({'transaction_id': tid, 'family': fam, 'tau': tau, 'score': float(isc)})
        if_norm_ear_rows.append({'transaction_id': tid, 'family': fam, 'tau': tau, 'score': float(isnc)})
    print(f'  τ={tau}: test故障 {int(yte.sum())} | PR-AUC LGB {lgb_pr:.3f} vs IF {if_pr:.3f} (norm-only {if_norm_pr:.3f}) | '
          f'ρ={rho:.2f} top10重叠 {overlap}/10', flush=True)

lgb_thr_map = {t: per_tau[t]['lgb']['thr'] for t in per_tau}
if_thr_map = {t: per_tau[t]['iforest']['thr'] for t in per_tau}
if_norm_thr_map = {t: per_tau[t]['iforest_normal_only']['thr'] for t in per_tau}
_, lgb_ear_summary = assemble_ear(pd.DataFrame(lgb_ear_rows), lgb_thr_map)
_, if_ear_summary = assemble_ear(pd.DataFrame(if_ear_rows), if_thr_map)
_, if_norm_ear_summary = assemble_ear(pd.DataFrame(if_norm_ear_rows), if_norm_thr_map)

results = {'per_tau': per_tau,
           'ear': {'lightgbm': lgb_ear_summary,
                   'isolation_forest': if_ear_summary,
                   'isolation_forest_normal_only': if_norm_ear_summary},
           'note': ('同口径对比：同一 52 维前缀特征/同一 owner1-6→7-8 切分/同一指标。'
                    'LightGBM=监督(用标签训练)；IsolationForest=无监督(fit 全部 train,不用标签)；'
                    'isolation_forest_normal_only=半监督对照(仅 fit normal,堵训练集污染质疑)。'
                    '两者阈值都在 val 上用标签校准到 precision≥0.90(标准做法)。'
                    'DIFFI 式重要性=深度加权隔离特征重要性 imp[f]=Σ_trees Σ_nodes (n_node/n_total)/depth。'
                    'IF 异常分=-score_samples(越大越异常)。')}
with open(os.path.join(OUT, 'p7_supervised_vs_unsupervised.json'), 'w', encoding='utf-8') as fp:
    json.dump(results, fp, ensure_ascii=False, indent=2, default=str)

print('\n=== E 结果：EAR（val precision≥0.90 校准）===')
for m in ['lightgbm', 'isolation_forest', 'isolation_forest_normal_only']:
    for L in LINEAGES:
        d = results['ear'][m][L]
        print(f'  {m:<26} {L:<8} 预警 {d["n_alerted"]}/{d["n"]} ({d["alert_rate"]}) '
              f'EAR中位 {d["ear_median"]}min lead中位 {d["lead_median"]}min', flush=True)
print('结果已存 journal/docs/p7_supervised_vs_unsupervised.json', flush=True)
