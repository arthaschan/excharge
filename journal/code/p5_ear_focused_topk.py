#!/usr/bin/env python3
"""P5 / 候选 A — 聚焦 top-k 最早分叉特征的 EAR（对比全 52 特征），CPU

背景（承接 HANDOFF 第六节 A）：
  思路1 的单特征规则失败，因为单特征太弱。这里改试「聚焦 LightGBM」：
  每谱系取「最早分叉」的 top-k 特征（k=5/10，按短 τ∈{1,2,3} 单特征 PR-AUC 峰值排序），
  用这 top-k 特征训 LightGBM 做 EAR，对比同谱系「全 52 特征」LightGBM EAR。

判据（A 的要求）：短 τ 的 recall@precision≥0.90 是否更高（以及事务级预警率/EAR/lead）。

协议（严格沿用 phase3_ear.py）：
  - 训练 owner1-6 → 测试 owner7-8（跨站）。
  - 每 τ 单独训练：谱系故障(label=1) vs normal(label=0)，owner1-6 内 stratify 20% val（seed42）。
  - val 上 precision≥0.90 校准阈值（取最高 recall），test 上评估 recall/precision。
  - 事务级 EAR = 可用 τ 中首个 P≥阈值 的 τ；lead_min = dur - EAR。
  - 特征选择在训练集(owner1-6)上完成，不碰测试集（无泄漏）。

产出：journal/docs/p5_ear_focused_topk.json
"""
import pandas as pd, numpy as np, json, os, warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE)
EARLY = os.path.join(ROOT, 'earlywarning')
DATA = os.path.join(EARLY, 'data')
OUT = os.path.join(BASE, 'docs')
os.makedirs(OUT, exist_ok=True)
SEED = 42
TAUS = [1, 2, 3, 5, 10, 20]
SHORT_TAUS = [1, 2, 3]
KS = [5, 10]
LINEAGES = ['startup', 'run']

SCHEMA = json.load(open(os.path.join(DATA, 'prefix_features_v1.json')))
FEAT_COLS = SCHEMA['feature_cols']

# ---------- 数据 ----------
feat = pd.read_parquet(os.path.join(DATA, 'prefix_feats_v1.parquet'),
                       columns=FEAT_COLS + ['transaction_id', 'prefix_type', 'prefix_val',
                                            'n_prefix_rows', 'owner', 'family', 'label'])
sub = feat[feat['prefix_type'] == 'time'].copy()
meta = pd.read_parquet(os.path.join(DATA, 'prefix_dataset_full.parquet'), columns=['transaction_id', 'dur_min'])
sub = sub.merge(meta, on='transaction_id', how='left')
train_owners = [f'Sheet{i}' for i in range(1, 7)]
test_owners = ['Sheet7', 'Sheet8']
tr_all = sub[sub['owner'].isin(train_owners)]
te = sub[sub['owner'].isin(test_owners)]
print(f'[load] time rows {len(sub)} | train(owner1-6) {len(tr_all)} | test(owner7-8) {len(te)}', flush=True)


def calibrate_threshold(yval, Pva, target=0.90):
    """val 上 precision≥target 的最高 recall 阈值（phase3_ear.py 协议）。返回 (thr, val_prec, val_rec)。"""
    prec, rec, th = precision_recall_curve(yval, Pva)
    cand = [(t, r) for p, r, t in zip(prec, rec, np.concatenate([th, [1.0]])) if p >= target]
    if cand:
        thr = max(cand, key=lambda x: x[1])[0]
    else:
        thr = 1.0  # 无法在 val 达到 0.90 → 不触发（诚实）
    pred = (Pva >= thr).astype(int)
    vp = precision_score(yval, pred, zero_division=0)
    vr = recall_score(yval, pred, zero_division=0)
    return float(thr), float(vp), float(vr)


# ---------- 特征选择：每谱系短 τ 单特征 PR-AUC 峰值排序（仅用训练集 owner1-6） ----------
print('[1/4] 特征选择（训练集上，短 τ∈{1,2,3} 单特征 PR-AUC 峰值）...', flush=True)
single_auc_peak, topk_feats = {}, {}
for L in LINEAGES:
    best, best_tau = {}, {}
    for tau in SHORT_TAUS:
        sv = tr_all[(tr_all['prefix_val'] == tau) & (tr_all['family'] == L)]
        sn = tr_all[(tr_all['prefix_val'] == tau) & (tr_all['label'] == 0)]
        if len(sv) < 20 or len(sn) < 20:
            continue
        y = np.concatenate([np.ones(len(sv)), np.zeros(len(sn))])
        for c in FEAT_COLS:
            x = np.concatenate([sv[c].values, sn[c].values]).astype(float)
            x = np.where(np.isnan(x), np.nanmedian(x), x)
            try:
                auc = float(average_precision_score(y, x))
            except Exception:
                auc = 0.5
            if c not in best or auc > best[c]:
                best[c], best_tau[c] = auc, tau
    ranking = sorted(best.items(), key=lambda kv: -kv[1])
    single_auc_peak[L] = {'top': {c: round(a, 4) for c, a in ranking[:10]},
                          'peak_tau': {c: best_tau[c] for c, _ in ranking[:10]}}
    topk_feats[L] = {k: [c for c, _ in ranking[:k]] for k in KS}
    print(f'  {L}: top10 = ' + ', '.join(f'{c}({a:.3f}@τ{best_tau[c]})' for c, a in ranking[:10]), flush=True)


# ---------- 训练 + 评估：每谱系 × 每 τ × {full, top5, top10} ----------
print('[2/4] 训练谱系模型（full 52 / top5 / top10）...', flush=True)
# lineage_models[L][featset][tau] = (thr, test_fault_df_with_P, test_prec, test_rec, n_test_fault)
def build_lineage_set(split_df, L, tau):
    """返回 (X, y, fault_mask_idx) for lineage L vs normal at τ."""
    m_fault = (split_df['prefix_val'] == tau) & (split_df['family'] == L)
    m_norm = (split_df['prefix_val'] == tau) & (split_df['label'] == 0)
    X = np.concatenate([split_df.loc[m_fault, FEAT_COLS].values, split_df.loc[m_norm, FEAT_COLS].values]).astype(float)
    y = np.concatenate([np.ones(int(m_fault.sum())), np.zeros(int(m_norm.sum()))])
    X = np.where(np.isnan(X), 0.0, X)
    return X, y, int(m_fault.sum())

results = {'lineage': {}, 'single_feat_auc_peak': single_auc_peak, 'topk_feats': {L: {str(k): v for k, v in d.items()} for L, d in topk_feats.items()}}

for L in LINEAGES:
    results['lineage'][L] = {}
    featsets = {'full': FEAT_COLS, 'top5': topk_feats[L][5], 'top10': topk_feats[L][10]}
    for fsname, cols in featsets.items():
        # 每 τ 训练 + 校准 + test 预测
        per_tau = {}
        ear_rows = []  # (tid, tau, P) for test lineage faults
        for tau in TAUS:
            Xtr, ytr, ntr_fault = build_lineage_set(tr_all, L, tau)
            if ntr_fault < 15 or int((ytr == 0).sum()) < 50:
                continue
            Xte, yte, nte_fault = build_lineage_set(te, L, tau)
            if nte_fault == 0:
                continue
            # stratify 切 val
            try:
                itr, iva = train_test_split(np.arange(len(ytr)), test_size=0.2, random_state=SEED, stratify=ytr)
            except ValueError:
                itr, iva = train_test_split(np.arange(len(ytr)), test_size=0.2, random_state=SEED)
            pos_w = float((ytr[itr] == 0).sum()) / max(1, int((ytr[itr] == 1).sum()))
            m = lgb.LGBMClassifier(objective='binary', learning_rate=0.05, num_leaves=31,
                                   min_child_samples=30, n_estimators=400, random_state=0,
                                   n_jobs=4, verbosity=-1, scale_pos_weight=pos_w)
            m.fit(Xtr[itr][:, [FEAT_COLS.index(c) for c in cols]], ytr[itr])
            Pva = m.predict_proba(Xtr[iva][:, [FEAT_COLS.index(c) for c in cols]])[:, 1]
            thr, vp, vr = calibrate_threshold(ytr[iva], Pva)
            Pte = m.predict_proba(Xte[:, [FEAT_COLS.index(c) for c in cols]])[:, 1]
            pred = (Pte >= thr).astype(int)
            tprec = precision_score(yte, pred, zero_division=0)
            trec = recall_score(yte, pred, zero_division=0)
            per_tau[tau] = {'thr': round(thr, 4), 'val_prec': round(vp, 4), 'val_rec': round(vr, 4),
                            'test_prec': round(tprec, 4), 'test_rec': round(trec, 4),
                            'n_test_fault': nte_fault}
            # 收集 test 谱系故障行的 P（用于 EAR）
            fault_mask = (te['prefix_val'] == tau) & (te['family'] == L)
            for tid, p, fam in zip(te.loc[fault_mask, 'transaction_id'], Pte[:nte_fault], te.loc[fault_mask, 'family']):
                ear_rows.append((tid, tau, float(p)))
        # 事务级 EAR
        edf = pd.DataFrame(ear_rows, columns=['transaction_id', 'tau', 'P'])
        rows = []
        for tid, g in edf.groupby('transaction_id'):
            g = g.sort_values('tau')
            thr_map = {t: per_tau[t]['thr'] for t in g['tau'] if t in per_tau}
            ear = None
            for t in g['tau']:
                if t in thr_map and g.loc[g['tau'] == t, 'P'].iloc[0] >= thr_map[t]:
                    ear = t
                    break
            rows.append((tid, ear))
        earres = pd.DataFrame(rows, columns=['transaction_id', 'ear'])
        earres = earres.merge(meta, on='transaction_id', how='left')
        earres['lead_min'] = earres['dur_min'] - earres['ear']
        n = len(earres)
        alerted = earres['ear'].notna()
        n_alert = int(alerted.sum())
        ears = earres.loc[alerted, 'ear']
        leads = earres.loc[alerted, 'lead_min']
        results['lineage'][L][fsname] = {
            'n_fault_txn': n,
            'n_alerted': n_alert,
            'alert_rate': round(n_alert / n, 4) if n else None,
            'ear_median': float(np.median(ears)) if len(ears) else None,
            'ear_dist': {str(int(t)): int((ears == t).sum()) for t in sorted(ears.unique())} if len(ears) else {},
            'lead_median': round(float(np.median(leads)), 1) if len(leads) else None,
            'n_unalerted': n - n_alert,
            'per_tau': per_tau,
        }
        print(f'  [{L}/{fsname}] 预警 {n_alert}/{n} ({results["lineage"][L][fsname]["alert_rate"]}) '
              f'EAR中位 {results["lineage"][L][fsname]["ear_median"]}min lead中位 {results["lineage"][L][fsname]["lead_median"]}min', flush=True)


# ---------- 全球全模型 EAR 复现（sanity anchor，对标 phase3 0.6593/0.8148） ----------
print('[3/4] 复现全球全模型 EAR（sanity，对标 phase3）...', flush=True)
gmodels, gths = {}, {}
for tau in TAUS:
    s = sub[sub['prefix_val'] == tau]
    tr_s = s[s['owner'].isin(train_owners)]
    te_s = s[s['owner'].isin(test_owners)]
    if len(te_s) == 0 or len(np.unique(tr_s['label'])) < 2:
        continue
    trs, vas = train_test_split(tr_s, test_size=0.2, random_state=42, stratify=tr_s['label'])
    pos_w = float((trs['label'] == 0).sum()) / max(1, int((trs['label'] == 1).sum()))
    m = lgb.LGBMClassifier(objective='binary', learning_rate=0.05, num_leaves=31,
                           min_child_samples=30, n_estimators=400, random_state=0,
                           n_jobs=4, verbosity=-1, scale_pos_weight=pos_w)
    m.fit(trs[FEAT_COLS].values, trs['label'].values)
    Pva = m.predict_proba(vas[FEAT_COLS].values)[:, 1]
    thr, vp, vr = calibrate_threshold(vas['label'].values, Pva)
    Pte = m.predict_proba(te_s[FEAT_COLS].values)[:, 1]
    gmodels[tau] = (te_s.copy(), Pte)
    gths[tau] = thr
grows = []
for tau in TAUS:
    if tau not in gmodels:
        continue
    te_s, Pte = gmodels[tau]
    mk = te_s['label'] == 1
    for tid, p, fam in zip(te_s['transaction_id'][mk], Pte[mk], te_s['family'][mk]):
        grows.append({'transaction_id': tid, 'family': fam, 'tau': tau, 'P': float(p)})
gdf = pd.DataFrame(grows)
global_summary = {}
for fam in ['startup', 'run']:
    gf = gdf[gdf['family'] == fam]
    tids = gf['transaction_id'].unique()
    n = len(tids)
    alerted = []
    for tid in tids:
        g = gf[gf['transaction_id'] == tid].sort_values('tau')
        ear = None
        for t in g['tau']:
            if g.loc[g['tau'] == t, 'P'].iloc[0] >= gths.get(t, 0.5):
                ear = t
                break
        if ear is not None:
            alerted.append(tid)
    global_summary[fam] = {'n': n, 'n_alerted': len(alerted), 'alert_rate': round(len(alerted) / n, 4)}
    print(f'  [global/{fam}] {len(alerted)}/{n} ({global_summary[fam]["alert_rate"]})  参考: startup 0.6593 / run 0.8148', flush=True)
results['global_full'] = global_summary

with open(os.path.join(OUT, 'p5_ear_focused_topk.json'), 'w', encoding='utf-8') as fp:
    json.dump(results, fp, ensure_ascii=False, indent=2, default=str)

print('[4/4] 结果已存 journal/docs/p5_ear_focused_topk.json', flush=True)
print('\n=== A 判据：短 τ recall@precision≥0.90（test）===')
for L in LINEAGES:
    print(f'  --- {L} ---')
    for fsname in ['full', 'top5', 'top10']:
        d = results['lineage'][L][fsname]
        per_tau = d['per_tau']
        short = {t: per_tau[t] for t in SHORT_TAUS if t in per_tau}
        line = ', '.join(f'τ{t}: rec={per_tau[t]["test_rec"]:.3f} prec={per_tau[t]["test_prec"]:.3f}' for t in short)
        print(f'    {fsname:<6} EAR预警 {d["alert_rate"]} ({d["n_alerted"]}/{d["n_fault_txn"]}) | {line}', flush=True)
