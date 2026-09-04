#!/usr/bin/env python3
"""structure_ab.py — Phase 1 多任务结构 A vs B LightGBM 判定实验
（研究方案 §4 "多任务结构(二选一)" / gate_report_phase0 §8 指引）

背景：前缀终止预警需同时输出"会不会终止(y)"与"若终止是哪个族(startup/run)"。
候选结构：
  A. 两阶段级联：Stage1 判 P(终止)；Stage2(仅在故障上训练) 判 P(startup|故障)
     → 部署分数：startup预警=p1*ps, run预警=p1*(1-ps), 三类分布=[1-p1, p1*ps, p1*(1-ps)]
  B. 三分类联合：normal/startup/run 单模型 (class_weight='balanced')
     → 部署分数：终止=1-P(normal)；条件族概率=P(startup)/(P(startup)+P(run))

判定口径（同分布为主 · 跨站为辅，3-seed LightGBM）：
  主判据1 AP_term      —— 终止检测 PR-AUC（部署核心能力，两结构应相近）
  主判据2 AP_run_fault —— 真故障集内 run 类识别 PR-AUC（run 是少数族，最能区分 A/B 结构差异）
  辅判据   AP_startup_fault / macro3(三类二值PR-AUC均值) / AP_startup_e2e(端到端启动型预警)

输出：docs/structure_ab_results.json + 控制台判定摘要
"""
import pandas as pd
import numpy as np
import json, os, time
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import lightgbm as lgb

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_V1 = os.path.join(BASE, 'data', 'prefix_feats_v1.parquet')
SCHEMA = json.load(open(os.path.join(BASE, 'data', 'prefix_features_v1.json')))
OUTJSON = os.path.join(BASE, 'docs', 'structure_ab_results.json')

SEEDS = [0, 1, 2]
TIME_TAUS = [1, 2, 3, 5, 10, 20]
META_COLS = {'transaction_id', 'begin_time', 'owner', 'label', 'class_judge', 'types',
             'dur_min', 'n_rows', 'family', 'offsets', 'vals_flat',
             'prefix_type', 'prefix_val', 'n_prefix_rows'}
FEAT_COLS = [c for c in SCHEMA['feature_cols'] if c not in META_COLS]
assert len(FEAT_COLS) == SCHEMA['n_feature_dims'] == 52, FEAT_COLS


def lgb_bin(X, y, seed, spw=None):
    params = dict(objective='binary', learning_rate=0.05, num_leaves=31,
                  min_child_samples=30, n_estimators=400, random_state=seed,
                  n_jobs=4, verbosity=-1)
    if spw is not None:
        params['scale_pos_weight'] = spw
    m = lgb.LGBMClassifier(**params)
    m.fit(X, y)
    return m


def lgb_multi(X, y, seed):
    m = lgb.LGBMClassifier(objective='multiclass', num_class=3, learning_rate=0.05,
                           num_leaves=31, min_child_samples=30, n_estimators=400,
                           class_weight='balanced', random_state=seed, n_jobs=4, verbosity=-1)
    m.fit(X, y)
    return m


def ap(y_true, score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return average_precision_score(y_true, score)


def split_by_owner_label(df, test_size=0.25, seed=0):
    strat = df['owner'].astype(str) + '_' + df['label'].astype(str)
    vc = strat.value_counts()
    rare = set(vc[vc < 4].index)
    strat = strat.apply(lambda s: ('other_' + s.split('_')[-1]) if s in rare else s)
    tr, te = train_test_split(df, test_size=test_size, stratify=strat, random_state=seed)
    return tr, te


results = {'meta': {'date': '2026-09-03', 'seeds': SEEDS,
                    'feature_version': 'prefix_feats_v1',
                    'models': 'LightGBM (binary/multiclass), class_weight balanced for B',
                    'protocol': '同分布(随机分层)主 + 跨站(owner1-6→7-8)辅',
                    'feat_dims': len(FEAT_COLS)},
           'by_tau': {}}

t0 = time.time()
print('[0] 载入 prefix_feats_v1 ...', flush=True)
feat = pd.read_parquet(FEAT_V1)
print(f'  特征表 {len(feat):,} 行 | {len(FEAT_COLS)} 维', flush=True)

# 标签编码 B 用：normal=0, startup=1, run=2
fam3 = {'normal': 0, 'startup': 1, 'run': 2}
feat['y3'] = feat['family'].map(fam3)

for tau in TIME_TAUS:
    sub = feat[(feat['prefix_type'] == 'time') & (feat['prefix_val'] == tau)].copy()
    if len(sub) < 800:
        results['by_tau'][str(tau)] = {'note': '样本过少', 'n': int(len(sub))}
        continue
    X_all = sub[FEAT_COLS].values
    y_all = sub['label'].values
    y3_all = sub['y3'].values
    pos = int(y_all.sum()); neg = int((y_all == 0).sum())
    spw1 = neg / max(pos, 1)

    per = {'n': int(len(sub)), 'pos_rate': float(y_all.mean()),
           'random': {'A': {}, 'B': {}}, 'cross': {'A': {}, 'B': {}}}

    # ============ 同分布（主判据） ============
    for sd in SEEDS:
        tr, te = split_by_owner_label(sub, seed=sd)
        tr = tr.reset_index(drop=True); te = te.reset_index(drop=True)
        Xtr, ytr, y3tr = tr[FEAT_COLS].values, tr['label'].values, tr['y3'].values
        Xte, yte, y3te = te[FEAT_COLS].values, te['label'].values, te['y3'].values

        # ---- A 两阶段 ----
        m1 = lgb_bin(Xtr, ytr, sd, spw=spw1)
        p1 = m1.predict_proba(Xte)[:, 1]
        tr_f = tr[tr['family'].isin(['startup', 'run'])].reset_index(drop=True)
        m2 = lgb_bin(tr_f[FEAT_COLS].values, (tr_f['family'] == 'startup').astype(int).values, sd)
        ps = m2.predict_proba(Xte)[:, 1]                       # P(startup|fault) 语义（模型在故障上训练）
        a_norm, a_st, a_run = 1 - p1, p1 * ps, p1 * (1 - ps)   # 三类分布

        # ---- B 三分类 ----
        mb = lgb_multi(Xtr, y3tr, sd)
        pb = mb.predict_proba(Xte)                             # [P0, P1, P2]
        b_norm, b_st, b_run = pb[:, 0], pb[:, 1], pb[:, 2]

        # 指标（每 seed 累计）
        mask_fault = y3te > 0          # 故障集 = startup(1) ∪ run(2)
        per['random']['A'].setdefault('ap_term', []).append(ap(yte, p1))
        per['random']['B'].setdefault('ap_term', []).append(ap(yte, 1 - b_norm))
        if mask_fault.sum() >= 10:
            yf = y3te[mask_fault] == 1  # 在故障集内：startup 为正
            per['random']['A'].setdefault('ap_startup_fault', []).append(ap(yf, ps[mask_fault]))
            per['random']['A'].setdefault('ap_run_fault', []).append(ap(~yf, (1 - ps)[mask_fault]))
            b_cond = (b_st + b_run)[mask_fault] + 1e-12
            per['random']['B'].setdefault('ap_startup_fault', []).append(ap(yf, b_st[mask_fault] / b_cond))
            per['random']['B'].setdefault('ap_run_fault', []).append(ap(~yf, b_run[mask_fault] / b_cond))
        # 端到端 startup 预警 + macro3
        per['random']['A'].setdefault('ap_startup_e2e', []).append(ap(yte == 1, a_st))
        per['random']['B'].setdefault('ap_startup_e2e', []).append(ap(yte == 1, b_st))
        per['random']['A'].setdefault('macro3', []).append(np.nanmean([
            ap(yte == 1, a_st), ap(yte == 1, a_run), ap(yte == 0, a_norm)]))
        per['random']['B'].setdefault('macro3', []).append(np.nanmean([
            ap(yte == 1, b_st), ap(yte == 1, b_run), ap(yte == 0, b_norm)]))

    # ============ 跨站（辅判据，owner1-6 → 7-8） ============
    tr = sub[sub['owner'].isin([f'Sheet{i}' for i in range(1, 7)])]
    te = sub[sub['owner'].isin(['Sheet7', 'Sheet8'])]
    if len(tr) >= 500 and len(te) >= 100:
        Xtr, ytr, y3tr = tr[FEAT_COLS].values, tr['label'].values, tr['y3'].values
        Xte, yte, y3te = te[FEAT_COLS].values, te['label'].values, te['y3'].values
        for sd in SEEDS:
            m1 = lgb_bin(Xtr, ytr, sd, spw=spw1)
            p1 = m1.predict_proba(Xte)[:, 1]
            tr_f = tr[tr['family'].isin(['startup', 'run'])]
            m2 = lgb_bin(tr_f[FEAT_COLS].values, (tr_f['family'] == 'startup').astype(int).values, sd)
            ps = m2.predict_proba(Xte)[:, 1]
            a_norm, a_st, a_run = 1 - p1, p1 * ps, p1 * (1 - ps)
            mb = lgb_multi(Xtr, y3tr, sd)
            pb = mb.predict_proba(Xte)
            b_norm, b_st, b_run = pb[:, 0], pb[:, 1], pb[:, 2]
            mask_fault = y3te > 0
            per['cross']['A'].setdefault('ap_term', []).append(ap(yte, p1))
            per['cross']['B'].setdefault('ap_term', []).append(ap(yte, 1 - b_norm))
            if mask_fault.sum() >= 10:
                yf = y3te[mask_fault] == 1
                per['cross']['A'].setdefault('ap_startup_fault', []).append(ap(yf, ps[mask_fault]))
                per['cross']['A'].setdefault('ap_run_fault', []).append(ap(~yf, (1 - ps)[mask_fault]))
                b_cond = (b_st + b_run)[mask_fault] + 1e-12
                per['cross']['B'].setdefault('ap_startup_fault', []).append(ap(yf, b_st[mask_fault] / b_cond))
                per['cross']['B'].setdefault('ap_run_fault', []).append(ap(~yf, b_run[mask_fault] / b_cond))
            per['cross']['A'].setdefault('ap_startup_e2e', []).append(ap(yte == 1, a_st))
            per['cross']['B'].setdefault('ap_startup_e2e', []).append(ap(yte == 1, b_st))

    # 汇总均值（跳过 nan 项）
    def agg(d):
        out = {}
        for k, v in d.items():
            vv = [x for x in v if x == x]   # drop nan
            out[k] = {'mean': round(float(np.mean(vv)), 4), 'std': round(float(np.std(vv)), 4)} if vv else None
        return out
    per['random']['A'] = agg(per['random']['A'])
    per['random']['B'] = agg(per['random']['B'])
    per['cross']['A'] = agg(per['cross']['A'])
    per['cross']['B'] = agg(per['cross']['B'])
    results['by_tau'][str(tau)] = per

    # 控制台摘要
    rA = per['random']['A']; rB = per['random']['B']
    cA = per['cross']['A']; cB = per['cross']['B']
    f1 = lambda d, k: d.get(k, {}).get('mean') if d.get(k) else None
    f2 = lambda v: f'{v:.4f}' if v is not None else 'N/A'
    print(f"\nτ={tau:2d}min n={len(sub):,} 故障率={pos/len(sub)*100:.2f}% "
          f"| startup={int((sub['family']=='startup').sum()):,} run={int((sub['family']=='run').sum()):,}", flush=True)
    print(f"  同分布 AP_term:         A={f2(f1(rA,'ap_term'))}  B={f2(f1(rB,'ap_term'))}", flush=True)
    print(f"  同分布 AP_run_fault:    A={f2(f1(rA,'ap_run_fault'))}  B={f2(f1(rB,'ap_run_fault'))}", flush=True)
    print(f"  同分布 AP_startup_fault:A={f2(f1(rA,'ap_startup_fault'))}  B={f2(f1(rB,'ap_startup_fault'))}", flush=True)
    print(f"  同分布 AP_startup_e2e:  A={f2(f1(rA,'ap_startup_e2e'))}  B={f2(f1(rB,'ap_startup_e2e'))}", flush=True)
    if cA:
        print(f"  跨站   AP_term:      A={f2(f1(cA,'ap_term'))}  B={f2(f1(cB,'ap_term'))}", flush=True)
        print(f"  跨站   AP_run_fault:  A={f2(f1(cA,'ap_run_fault'))}  B={f2(f1(cB,'ap_run_fault'))}", flush=True)

os.makedirs(os.path.dirname(OUTJSON), exist_ok=True)
with open(OUTJSON, 'w', encoding='utf-8') as fp:
    json.dump(results, fp, ensure_ascii=False, indent=2, default=str)
print(f'\n结果已保存: {OUTJSON}', flush=True)
print(f'总耗时 {time.time()-t0:.0f}s', flush=True)

# ============ 判定摘要 ============
print('\n=========== 结构 A vs B 判定摘要（同分布 3-seed 均值） ===========', flush=True)
diff_term, diff_run = [], []
for tau, per in results['by_tau'].items():
    rA, rB = per.get('random', {}).get('A', {}), per.get('random', {}).get('B', {})
    tA, tB = rA.get('ap_term', {}).get('mean'), rB.get('ap_term', {}).get('mean')
    fA, fB = rA.get('ap_run_fault', {}).get('mean'), rB.get('ap_run_fault', {}).get('mean')
    if tA and tB:
        diff_term.append(tA - tB)
        print(f'  τ={tau:>2}min: AP_term A-B={tA - tB:+.4f} | AP_run_fault A-B={(fA - fB) if (fA and fB) else float("nan"):+.4f}', flush=True)
        if fA and fB:
            diff_run.append(fA - fB)
print(f'\n  AP_term  A-B 均值差: {np.mean(diff_term):+.4f} (正=A优)', flush=True)
print(f'  AP_run_fault A-B 均值差: {np.mean(diff_run):+.4f} (正=A优, 少数族run识别)', flush=True)
