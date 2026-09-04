#!/usr/bin/env python3
"""P4/思路1 探路 — 可解释性反哺 EAR（更早预警），而非 PR-AUC

背景：老师方向"可解释性反哺提升效果"此前四条路都在优化 PR-AUC（已饱和 0.918），全负。
思路1 换个目标：把归因定位的"最早分叉特征"做成 EAR 专用预警规则，看能否比全特征 LightGBM
更早、更高预警率地触发（提升预警率/lead-time，而非 PR-AUC）。

Phase 3 归因已定位最早信号：
  - startup = 首分钟大电流突增（charginga_first 114.7 vs normal 60.9）
  - run     = 起步高压缓充（chargingv_first 357.9 vs normal 340.2）

探路（纯 CPU）：
  1. 逐 τ 算单特征 PR-AUC，验证"最早分叉特征"确实在 τ=1/2 就有判别力；
  2. 构造归因引导单特征规则（startup: charginga_first；run: chargingv_first），
     阈值在 owner1-6 内部 val 校准到 precision≥0.90（与 Phase 3 EAR 同协议）；
  3. 在 owner7-8 测试集上对比：规则 EAR vs 全特征 LightGBM EAR 的预警率 / EAR / lead-time。

判据：规则在相同 precision≥0.90 下预警率更高、或 EAR 更早（lead-time 更长）→ 思路1 成立。
输出：journal/docs/p4_ear_attribution_guided.json
"""
import pandas as pd, numpy as np, json, os, time, warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE)
EARLY = os.path.join(ROOT, 'earlywarning')
FEAT = os.path.join(EARLY, 'data', 'prefix_feats_v1.parquet')
OUT = os.path.join(BASE, 'docs')
os.makedirs(OUT, exist_ok=True)
SEED = 42
TIME_TAUS = [1, 2, 3, 5, 10, 20]

# ---------- 1. 载入 + 切分 ----------
df = pd.read_parquet(FEAT)
# 合并 dur_min（用于 lead-time 计算）
meta = pd.read_parquet(os.path.join(EARLY, 'data', 'prefix_dataset_full.parquet'),
                       columns=['transaction_id', 'dur_min'])
df = df.merge(meta, on='transaction_id', how='left')
df = df[df['prefix_type'] == 'time'].copy()
meta_cols = ['transaction_id', 'prefix_type', 'prefix_val', 'n_prefix_rows', 'owner', 'family', 'label', 'dur_min']
feat_cols = [c for c in df.columns if c not in meta_cols]
train_owners = [f'Sheet{i}' for i in range(1, 7)]
test_owners = ['Sheet7', 'Sheet8']
tr_all = df[df['owner'].isin(train_owners)]
te = df[df['owner'].isin(test_owners)]
# owner1-6 内部 80/20 分层切 val（与 Phase 3 同协议）
tr, va = train_test_split(tr_all, test_size=0.2, stratify=tr_all['owner'].astype(str) + '_' + tr_all['label'].astype(str), random_state=SEED)
print(f'[1/4] train {len(tr):,} val {len(va):,} test {len(te):,} | feat {len(feat_cols)}', flush=True)

# ---------- 2. 逐 τ 单特征 PR-AUC（验证最早分叉特征）----------
print('[2/4] 逐 τ 单特征 PR-AUC（定位最早分叉特征）...', flush=True)
# 逐 τ 分族单特征 PR-AUC（定位各谱系最早分叉特征）
single_feat_auc = {}
for tau in TIME_TAUS:
    sub_va = va[va['prefix_val'] == tau]
    row = {}
    for lineage in ['startup', 'run']:
        subv = sub_va[sub_va['family'] == lineage]; subn = sub_va[sub_va['family'] == 'normal']
        if len(subv) < 20 or len(subn) < 20:
            row[lineage] = {}
            continue
        L = {}
        for c in feat_cols:
            x = np.concatenate([subv[c].values, subn[c].values])
            y = np.concatenate([np.ones(len(subv)), np.zeros(len(subn))])
            x = np.where(np.isnan(x), np.nanmedian(x), x)
            try:
                L[c] = float(average_precision_score(y, x))
            except Exception:
                L[c] = np.nan
        row[lineage] = L
    single_feat_auc[tau] = row

# 打印各谱系最早 τ 下的 top 单特征
print('  分族单特征 PR-AUC（正=特征值越大越像该族故障）:', flush=True)
for lineage in ['startup', 'run']:
    for tau in [1, 2, 3]:
        if tau in single_feat_auc and single_feat_auc[tau].get(lineage):
            top = sorted(single_feat_auc[tau][lineage].items(), key=lambda kv: -kv[1])[:4]
            print(f'    {lineage} τ={tau}: ' + ', '.join(f'{k}={v:.3f}' for k, v in top), flush=True)

# ---------- 3. 归因引导规则 + 全模型 EAR 对比 ----------
print('[3/4] 构造 EAR 规则并对比 ...', flush=True)
RULES = {'startup': ['chargingv_min', 'chargingv_mean', 'chargingv_first'],
         'run': ['out_power_last', 'out_power_max', 'out_power_mean']}

def calibrate_threshold(vals, y, target_prec=0.90):
    """找使 'vals>θ' 的 precision≥target_prec 的最低阈值 θ（最大化 recall）。"""
    order = np.argsort(-vals)
    svals, sy = vals[order], y[order]
    cum_f = np.cumsum(sy)
    cum_prec = cum_f / np.arange(1, len(sy) + 1)
    valid = np.where(cum_prec >= target_prec)[0]
    if len(valid) == 0:
        return None, None
    i = int(valid[-1])   # 最低阈值且 precision≥0.90
    return float(svals[i]), {'n_alert': i + 1, 'n_fault': int(cum_f[i]), 'prec': float(cum_prec[i])}

def evaluate_ear(fault_tx, tau_groups, scorer):
    """对每个故障事务，返回第一个满足 scorer 的 τ（EAR）与 lead_min。scorer(tau)->bool 或 prob>=thr。
    fault_tx: 故障事务的 df（含各 τ 行）；tau_groups: 该事务各 τ 的行 dict。"""
    alerted, ear, lead = [], {}, {}
    for tid, g in fault_tx.groupby('transaction_id'):
        dur = g['dur_min'].iloc[0] if 'dur_min' in g.columns else np.nan
        for tau in TIME_TAUS:
            sub = g[g['prefix_val'] == tau]
            if len(sub) == 0:
                continue
            if scorer(sub, tau):
                alerted.append(tid); ear[tid] = tau; lead[tid] = dur - tau
                break
    return alerted, ear, lead

def full_lgb_scorer(tau_models, tau_thr):
    def _s(sub, tau):
        if tau not in tau_models:
            return False
        m = tau_models[tau]
        x = sub[feat_cols].values
        x = np.where(np.isnan(x), 0, x)
        p = m.predict_proba(x)[:, 1]
        return bool(p[0] >= tau_thr[tau])
    return _s

def rule_scorer(feat, tau_thr):
    def _s(sub, tau):
        if tau not in tau_thr or feat not in sub.columns:
            return False
        th = tau_thr[tau]
        if th is None:                      # 单特征无法在 val 达到 0.90 精度 → 规则不触发
            return False
        v = sub[feat].values[0]
        return bool(np.isfinite(v) and v >= th)
    return _s

results = {'rules': {}, 'lightgbm': {}, 'single_feat_auc': single_feat_auc}

# 逐 τ 训 LightGBM（全特征）+ 校准阈值；同时校准规则阈值
lgb_models, lgb_thr = {}, {}
rule_thr = {r: {} for r in RULES}
for tau in TIME_TAUS:
    sub_tr = tr[tr['prefix_val'] == tau]; sub_va = va[va['prefix_val'] == tau]
    if len(sub_tr) < 500 or sub_tr['label'].sum() < 50 or sub_va['label'].sum() < 30:
        continue
    pos = int(sub_tr['label'].sum()); neg = int((sub_tr['label'] == 0).sum())
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=SEED,
                           n_jobs=4, verbosity=-1, scale_pos_weight=neg / max(pos, 1))
    Xtr = np.where(np.isnan(sub_tr[feat_cols].values), 0, sub_tr[feat_cols].values)
    m.fit(Xtr, sub_tr['label'].values)
    pva = m.predict_proba(np.where(np.isnan(sub_va[feat_cols].values), 0, sub_va[feat_cols].values))[:, 1]
    lgb_models[tau] = m
    # 校准：val 上 precision≥0.90 的最低阈值（最大化 recall）
    order = np.argsort(-pva); sp, sy = pva[order], sub_va['label'].values[order]
    cum_prec = np.cumsum(sy) / np.arange(1, len(sy) + 1)
    valid = np.where(cum_prec >= 0.90)[0]
    lgb_thr[tau] = float(sp[valid[-1]]) if len(valid) else 1.0
    # 规则阈值校准
    for r, feats in RULES.items():
        # 用该 τ 的 startup/run 样本校准（规则是分族的）
        subv = sub_va[sub_va['family'] == r] if r != 'normal' else sub_va
        subn = sub_va[sub_va['family'] == 'normal']
        if len(subv) < 5 or len(subn) < 5:
            continue
        # 正=该族故障，负=normal；规则 'feature>θ' 判故障
        vals = np.concatenate([subv[feats[0]].values, subn[feats[0]].values])
        y = np.concatenate([np.ones(len(subv)), np.zeros(len(subn))])
        vals = np.where(np.isnan(vals), np.nanmedian(vals), vals)
        th, info = calibrate_threshold(vals, y, 0.90)
        rule_thr[r][tau] = th

# ---------- 4. 在 test 上评估 EAR ----------
print('[4/4] 评估 EAR ...', flush=True)
fault_tx = te[te['label'] == 1]

# 全模型 LightGBM EAR
al, ear, lead = evaluate_ear(fault_tx, None, full_lgb_scorer(lgb_models, lgb_thr))
results['lightgbm'] = {'n_fault': int(len(fault_tx['transaction_id'].unique())),
                       'n_alerted': len(al), 'alert_rate': float(len(al) / max(1, len(fault_tx['transaction_id'].unique()))),
                       'ear_median': float(np.median(list(ear.values()))) if ear else None,
                       'lead_median': float(np.median(list(lead.values()))) if lead else None}

# 归因引导规则 EAR（分族）
for r, feats in RULES.items():
    ftx = fault_tx[fault_tx['family'] == r]
    al, ear, lead = evaluate_ear(ftx, None, rule_scorer(feats[0], rule_thr[r]))
    results['rules'][r] = {'feat': feats[0], 'n_fault': int(len(ftx['transaction_id'].unique())),
                           'n_alerted': len(al), 'alert_rate': float(len(al) / max(1, len(ftx['transaction_id'].unique()))),
                           'ear_median': float(np.median(list(ear.values()))) if ear else None,
                           'lead_median': float(np.median(list(lead.values()))) if lead else None,
                           'thresholds': {str(k): (None if v is None else float(v)) for k, v in rule_thr[r].items()}}

with open(f'{OUT}/p4_ear_attribution_guided.json', 'w', encoding='utf-8') as fp:
    json.dump(results, fp, ensure_ascii=False, indent=2, default=str)

print('\n=== 思路1 探路判据 ===', flush=True)
print(f"  全模型 LightGBM EAR: 预警率 {results['lightgbm']['alert_rate']:.3f} "
      f"({results['lightgbm']['n_alerted']}/{results['lightgbm']['n_fault']}), "
      f"EAR中位 {results['lightgbm']['ear_median']}min, lead中位 {results['lightgbm']['lead_median']}min", flush=True)
for r, v in results['rules'].items():
    print(f"  归因规则[{r}: {v['feat']}]: 预警率 {v['alert_rate']:.3f} "
          f"({v['n_alerted']}/{v['n_fault']}), EAR中位 {v['ear_median']}min, lead中位 {v['lead_median']}min", flush=True)
print('  结果已保存 journal/docs/p4_ear_attribution_guided.json', flush=True)
