#!/usr/bin/env python3
"""P6 / 候选 F — 反向角度：可解释 EAR 诊断（为何有些故障无法提前预警），CPU

背景（承接 HANDOFF 第六节 F）：
  不再追求「反哺提升」，而是把可解释性作为 EAR 的「诊断/可信度」工具：
  用检测模型自己的归因（每 τ LightGBM gain 重要性）解释「为什么某些故障没能被提前预警」。

做法（严格沿用 phase3_ear.py 全球全模型协议，先复现 0.6593/0.8148）：
  1. 每 τ 训全球 LightGBM（全部故障 vs normal），取 gain 重要性 top-k（k=8）作该 τ 的归因特征。
  2. 逐故障事务组装 EAR（首个 P≥该 τ 阈值 的 τ），并落盘每 τ 概率明细。
  3. 对「未预警」故障分类：
     - data-disallowed：最早可用 τ > 3（无短前缀数据，插枪晚/采样稀疏 → 本质上无法早预警）。
     - model-missed：有 τ≤3 可用但仍未触发。
  4. 对 model-missed（及对照：已预警故障在其 EAR τ）：
     在最早可用 τ / EAR τ 上，计算故障在 top-k 归因特征上相对 normal 分布（同 τ、测试站）的偏离度：
     特征值落在 normal 的 [5%,95%] 分位内 = 「无偏离」；统计 top-k 里有多少个特征偏离。
     无偏离 → 短前缀本质上不可分（可解释性诊断：不是模型没学到，是信号还没出现）。

产出：
  journal/docs/p6_ear_interpretability_diagnosis.json
  journal/docs/p6_ear_unalerted_by_txn.csv（未预警故障明细 + 诊断标签）
  journal/docs/p6_ear_by_txn.csv（全部故障事务 EAR 明细）
"""
import pandas as pd, numpy as np, json, os, warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, precision_score, recall_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE)
EARLY = os.path.join(ROOT, 'earlywarning')
DATA = os.path.join(EARLY, 'data')
OUT = os.path.join(BASE, 'docs')
os.makedirs(OUT, exist_ok=True)
SEED = 42
TAUS = [1, 2, 3, 5, 10, 20]
TOPK = 8
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
tr_all = sub[sub['owner'].isin(train_owners)]
te = sub[sub['owner'].isin(test_owners)]
print(f'[load] time rows {len(sub)} | train {len(tr_all)} | test {len(te)}', flush=True)


def calibrate_threshold(yval, Pva, target=0.90):
    prec, rec, th = precision_recall_curve(yval, Pva)
    cand = [(t, r) for p, r, t in zip(prec, rec, np.concatenate([th, [1.0]])) if p >= target]
    if cand:
        return float(max(cand, key=lambda x: x[1])[0])
    return 1.0


# ---------- 1. 每 τ 全球全模型 + 归因(gain top-k) ----------
print('[1/4] 训练全球全模型 + 提取 gain 归因...', flush=True)
models, thr_by_tau = {}, {}
gain_topk_by_tau = {}
# 测试站 normal 分布统计（每 τ 每特征 mean/std + 分位数），用于偏离度
normal_stats = {}
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
    thr = calibrate_threshold(vas['label'].values, Pva)
    Pte = m.predict_proba(te_s[FEAT_COLS].values)[:, 1]
    models[tau] = (te_s.copy(), Pte)
    thr_by_tau[tau] = thr
    # gain 重要性 top-k
    gains = m.booster_.feature_importance(importance_type='gain')
    order = np.argsort(-gains)
    gain_topk_by_tau[tau] = [FEAT_COLS[i] for i in order[:TOPK]]
    # 测试站 normal 分布（分位数用）
    norm = te_s[te_s['label'] == 0][FEAT_COLS]
    normal_stats[tau] = {c: {'q05': float(np.percentile(norm[c], 5)), 'q95': float(np.percentile(norm[c], 95)),
                             'mean': float(norm[c].mean()), 'std': float(norm[c].std()),
                             'n': int(len(norm))} for c in FEAT_COLS}
    print(f'  τ={tau}: test故障 {int(te_s["label"].sum())} thr={thr:.3f} '
          f'topGain=' + ','.join(gain_topk_by_tau[tau][:5]), flush=True)


# ---------- 2. 事务级 EAR 组装 ----------
print('[2/4] 事务级 EAR 组装...', flush=True)
recs = []
for tau in TAUS:
    if tau not in models:
        continue
    te_s, Pte = models[tau]
    mk = te_s['label'] == 1
    for tid, p, fam in zip(te_s['transaction_id'][mk], Pte[mk], te_s['family'][mk]):
        recs.append({'transaction_id': tid, 'family': fam, 'tau': tau, 'P': float(p)})
edf = pd.DataFrame(recs)
rows = []
for tid, g in edf.groupby('transaction_id'):
    g = g.sort_values('tau')
    fam = g['family'].iloc[0]
    taus_avail = g['tau'].tolist()
    ear = None
    for t in taus_avail:
        if g.loc[g['tau'] == t, 'P'].iloc[0] >= thr_by_tau.get(t, 1.0):
            ear = t
            break
    rows.append({'transaction_id': tid, 'family': fam, 'taus_avail': taus_avail,
                 'earliest_tau': taus_avail[0], 'ear': ear,
                 **{f'P_tau{t}': float(g.loc[g['tau'] == t, 'P'].iloc[0]) if t in taus_avail else np.nan for t in TAUS}})
res = pd.DataFrame(rows)
res = res.merge(meta, on='transaction_id', how='left')
res['lead_min'] = res['dur_min'] - res['ear']
print(f'故障事务 {len(res)} | startup {(res.family=="startup").sum()} run {(res.family=="run").sum()}', flush=True)

# ---------- 3. 偏离度诊断（未预警 + 已预警对照） ----------
print('[3/4] 偏离度诊断（top-k gain 特征 vs normal 分布）...', flush=True)


def deviation_at(feat_row, tau):
    """在 τ 上，故障在 top-k gain 特征里相对 normal 分位的偏离情况。返回 (n_outside, detail, k)。"""
    if tau not in gain_topk_by_tau:
        return None, {}, 0
    topk = gain_topk_by_tau[tau]
    n_out = 0
    detail = {}
    for c in topk:
        v = feat_row.get(c, np.nan)
        st = normal_stats[tau][c]
        if not np.isfinite(v) or st['n'] < 5 or st['q05'] == st['q95']:
            detail[c] = {'val': None, 'outside': None, 'q05': st['q05'], 'q95': st['q95']}
            continue
        outside = bool(v < st['q05'] or v > st['q95'])
        if outside:
            n_out += 1
        detail[c] = {'val': round(float(v), 3), 'outside': outside,
                     'q05': st['q05'], 'q95': st['q95']}
    return n_out, detail, len(topk)


def classify(n_out, k):
    if k == 0:
        return 'unknown'
    if n_out == 0:
        return 'indistinguishable'   # 全部归因特征都落在 normal 中段 → 短前缀本质上不可分
    if n_out <= max(1, k // 2):
        return 'partial'             # 少数特征偏离，但不足以致命
    return 'strong'                  # 多数特征已偏离，理应可被模型抓到


# 测试站故障行的特征查找表 (tid, tau) -> dict(FEAT_COLS)
te_fault = te[te['label'] == 1]
feat_by_tx_tau = {(tid, tau): dict(zip(FEAT_COLS, row)) for (tid, tau), row in
                  zip(zip(te_fault['transaction_id'], te_fault['prefix_val']),
                      te_fault[FEAT_COLS].itertuples(index=False, name=None))}

diag = []
for _, r in res.iterrows():
    # 诊断 τ：未预警 → 最早可用 τ；已预警 → 其 EAR τ
    tau = r['ear'] if pd.notna(r['ear']) else r['earliest_tau']
    feat_row = feat_by_tx_tau.get((r['transaction_id'], int(tau)))
    if feat_row is None:
        n_out, detail, k = None, {}, 0
    else:
        n_out, detail, k = deviation_at(feat_row, int(tau))
    cls = classify(n_out, k) if n_out is not None else 'unknown'
    # data-disallowed 判定：最早可用 τ > 3
    disallowed = bool(r['earliest_tau'] > 3)
    if pd.isna(r['ear']):
        tag = 'data-disallowed' if disallowed else ('model-missed/' + cls)
    else:
        tag = 'alerted/' + cls
    diag.append({'transaction_id': r['transaction_id'], 'family': r['family'],
                 'earliest_tau': r['earliest_tau'], 'ear': r['ear'], 'lead_min': r['lead_min'],
                 'diag_tau': int(tau), 'n_outside': n_out, 'k': k, 'tag': tag,
                 'deviation': detail})
diag_df = pd.DataFrame(diag)

# ---------- 4. 汇总 ----------
print('[4/4] 汇总...', flush=True)
summary = {}
for L in LINEAGES:
    g = diag_df[diag_df['family'] == L]
    n = len(g)
    unalerted = g[g['ear'].isna()]
    n_un = len(unalerted)
    # 未预警的细分
    disallowed = unalerted[unalerted['tag'] == 'data-disallowed']
    missed = unalerted[unalerted['tag'] != 'data-disallowed']
    n_dis = len(disallowed)
    n_miss = len(missed)
    miss_break = missed['tag'].value_counts().to_dict() if n_miss else {}
    # 已预警的偏离度分布（对照）
    alerted = g[g['ear'].notna()]
    alerted_n_out = alerted['n_outside']
    summary[L] = {
        'n': n, 'n_alerted': int(alerted.shape[0]), 'n_unalerted': n_un,
        'unalerted': {
            'data_disallowed': n_dis,
            'model_missed': n_miss,
            'model_missed_breakdown': miss_break,
        },
        'alerted_n_outside': {
            'median': float(alerted_n_out.median()) if len(alerted_n_out) else None,
            'frac_0': float((alerted_n_out == 0).mean()) if len(alerted_n_out) else None,
        },
    }
    print(f'  {L}: 预警 {summary[L]["n_alerted"]}/{n} | 未预警 {n_un} = 数据不允许 {n_dis} + 模型漏报 {n_miss}', flush=True)
    print(f'      漏报细分 {miss_break}', flush=True)
    print(f'      已预警故障在其EAR τ 的 top-k 偏离特征数: 中位 {summary[L]["alerted_n_outside"]["median"]}, '
          f'0偏离占比 {summary[L]["alerted_n_outside"]["frac_0"]:.2f}', flush=True)

# ---------- 5. 模型置信度边际分析（未预警故障的 Pmax vs 阈值） ----------
# 未预警故障：取所有可用 τ 的 P 最大值 Pmax，与其所在 τ 的阈值比，判断「hard(模型确信normal)」vs「soft(边缘,卡在精度阈值下)」
def margin_bucket(r, thr_by_tau):
    ts = r['taus_avail']
    Ps = [(t, r[f'P_tau{t}']) for t in ts if pd.notna(r[f'P_tau{t}'])]
    if not Ps:
        return None, None, 'no-P'
    tmax, pmax = max(Ps, key=lambda x: x[1])
    thr = thr_by_tau.get(tmax, 1.0)
    if pmax < 0.5:
        return float(pmax), float(thr), 'hard'      # 模型确信是 normal，短前缀信号未出现
    if pmax < thr:
        return float(pmax), float(thr), 'soft'      # 有故障信号但低于高精度阈值（精度-召回权衡）
    return float(pmax), float(thr), 'fired'         # 不应发生（已预警）

margin = {}
for L in LINEAGES:
    g = res[(res['family'] == L) & (res['ear'].isna())]
    buckets = {'hard': 0, 'soft': 0, 'no-P': 0}
    pmax_list = []
    for _, r in g.iterrows():
        pmax, thr, b = margin_bucket(r, thr_by_tau)
        buckets[b] += 1
        if pmax is not None:
            pmax_list.append(pmax)
    margin[L] = {'n_unalerted': int(len(g)),
                 'buckets': buckets,
                 'pmax_median': float(np.median(pmax_list)) if pmax_list else None,
                 'pmax_p75': float(np.percentile(pmax_list, 75)) if pmax_list else None}
    print(f'  {L} 未预警 {len(g)}: hard(确信normal) {buckets["hard"]} / soft(边缘) {buckets["soft"]} / no-P {buckets["no-P"]} | '
          f'Pmax 中位 {margin[L]["pmax_median"]}', flush=True)

results = {'summary': summary,
           'margin_analysis': margin,
           'note': ('可解释 EAR 诊断：每 τ 用全球 LightGBM gain 重要性 top-8 作归因特征；'
                    '偏离=故障特征值落在测试站 normal 同 τ 的 [5%,95%] 分位之外；'
                    'data-disallowed=最早可用 τ>3(无短前缀数据)；'
                    'model-missed=有短前缀但仍未触发，按 top-k 归因特征的偏离数细分：'
                    'indistinguishable(0偏离,短前缀本质上不可分)/partial/strong；'
                    'margin_analysis: hard=未预警故障 Pmax<0.5(模型确信normal,短前缀信号未出现)'
                    'soft=0.5<=Pmax<阈值(有信号但卡在精度阈值下)'),
           'gain_topk_by_tau': {str(t): v for t, v in gain_topk_by_tau.items()},
           'thr_by_tau': {str(t): round(v, 3) for t, v in thr_by_tau.items()}}
with open(os.path.join(OUT, 'p6_ear_interpretability_diagnosis.json'), 'w', encoding='utf-8') as fp:
    json.dump(results, fp, ensure_ascii=False, indent=2, default=str)
res.to_csv(os.path.join(OUT, 'p6_ear_by_txn.csv'), index=False)
# 全量诊断明细（371 事务，含 tag + 偏离详情）
diag_df.to_csv(os.path.join(OUT, 'p6_ear_diag_by_txn.csv'), index=False)
# 仅未预警明细（与文件名一致）
diag_df[diag_df['ear'].isna()].drop(columns=['deviation']).to_csv(os.path.join(OUT, 'p6_ear_unalerted_by_txn.csv'), index=False)
print('结果已存 journal/docs/p6_ear_interpretability_diagnosis.json / p6_ear_by_txn.csv / '
      'p6_ear_diag_by_txn.csv / p6_ear_unalerted_by_txn.csv', flush=True)
