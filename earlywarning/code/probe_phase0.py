#!/usr/bin/env python3
"""probe_phase0.py — Phase 0 决策门探路实验（研究方案 §5.1）

对照研究方案决策门 4 子实验：
  0a 信息位置：时间切窗各 τ 的 LightGBM 前缀特征 PR-AUC（同分布随机分层）
  0b 启动型可分性：启动型故障 vs 正常短事务(<30min) GBDT PR-AUC（同分布）
  0c 跨站可达性：owner1-6 训练 → owner7-8 测试，逐 owner PR-AUC
  0d 时间 vs 进度切窗：两口径 PR-AUC 趋势一致性

通过标准（方案 §5.1）：
  0a: 至少一个 τ 的 LightGBM PR-AUC ≥ 0.65
  0b: PR-AUC ≥ 0.60（≠0.5 即双谱系成立）
  0c: 平均 ≥0.55 且非单站(Sheet7)独撑
  0d: 两口径趋势一致

输出：docs/gate_phase0_results.json（数字全落盘，供 gate_report_phase0.md 引用）
纪律：3-seed 平均；PR-AUC 主指标；scale_pos_weight 处理不平衡；同分布切分按 owner×label 分层。
"""
import pandas as pd
import numpy as np
import json, os, time
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import lightgbm as lgb

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT = os.path.join(BASE, 'data', 'prefix_feats.parquet')
META = os.path.join(BASE, 'data', 'probe_dataset.parquet')
OUTJSON = os.path.join(BASE, 'docs', 'gate_phase0_results.json')

SEEDS = [0, 1, 2]
TIME_TAUS = [1, 2, 3, 5, 10, 20]
PROG_PCTS = [10, 25, 50]
DROP_COLS = ['transaction_id', 'prefix_type', 'prefix_val', 'n_prefix_rows', 'owner', 'family', 'label']

def lgb_pr_auc(Xtr, ytr, Xte, yte, seed, neg_frac=None):
    """返回 PR-AUC（average_precision）"""
    params = dict(objective='binary', learning_rate=0.05, num_leaves=31,
                  min_child_samples=30, n_estimators=400, random_state=seed,
                  n_jobs=-1, verbosity=-1)
    if neg_frac is not None:
        params['scale_pos_weight'] = neg_frac
    m = lgb.LGBMClassifier(**params)
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    return average_precision_score(yte, p), p

def split_by_owner_label(df, test_size=0.25, seed=0):
    """同分布随机分层切分：owner×label 组合分层，保证切分后谱系均衡。
    稀有组合(样本<4)归并为 other_{label}，避免 StratifiedSplit 因最小类样本过少报错。"""
    strat = df['owner'].astype(str) + '_' + df['label'].astype(str)
    vc = strat.value_counts()
    rare = set(vc[vc < 4].index)
    strat = strat.apply(lambda s: ('other_' + s.split('_')[-1]) if s in rare else s)
    tr, te = train_test_split(df, test_size=test_size, stratify=strat, random_state=seed)
    return tr, te

results = {'meta': {'date': '2026-09-03', 'seeds': SEEDS, 'env': 'Mac MPS',
                    'feature_version': 'prefix_feats_v1', 'protocol': '见研究方案 §5.1'},
          '0a_information_location': {}, '0b_startup_sep': {}, '0c_cross_station': {},
          '0d_cut_sensitivity': {}}

t0 = time.time()
print('[0/4] 载入特征与元数据 ...', flush=True)
feat = pd.read_parquet(FEAT)
meta = pd.read_parquet(META)
dur_map = meta.set_index('transaction_id')['dur_min']
feat['dur_min'] = feat['transaction_id'].map(dur_map)
print(f'  特征表 {len(feat):,} 行', flush=True)

feat_cols = [c for c in feat.columns if c not in DROP_COLS and c != 'dur_min']
print(f'  特征 {len(feat_cols)} 维', flush=True)

# ============ 0a 信息位置（时间切窗 · 同分布） ============
print('\n[1/4] 0a 信息位置：时间切窗 LightGBM PR-AUC（同分布分层）...', flush=True)
for tau in TIME_TAUS:
    sub = feat[(feat['prefix_type'] == 'time') & (feat['prefix_val'] == tau)].copy()
    if len(sub) < 500:
        results['0a_information_location'][str(tau)] = {'n': int(len(sub)), 'pr_auc_mean': None, 'note': '样本过少'}
        continue
    X = sub[feat_cols].values
    y = sub['label'].values
    pos = int(y.sum()); neg = int((y == 0).sum())
    neg_frac = neg / max(pos, 1)
    aucs = []
    for sd in SEEDS:
        tr, te = split_by_owner_label(sub, seed=sd)
        Xtr, ytr, Xte, yte = tr[feat_cols].values, tr['label'].values, te[feat_cols].values, te['label'].values
        a, _ = lgb_pr_auc(Xtr, ytr, Xte, yte, sd, neg_frac=neg_frac)
        aucs.append(a)
    results['0a_information_location'][str(tau)] = {
        'n': int(len(sub)), 'pos': int(pos), 'neg': int(neg),
        'pr_auc_mean': float(np.mean(aucs)), 'pr_auc_std': float(np.std(aucs)),
        'pr_auc_per_seed': [round(a, 4) for a in aucs]}
    print(f'  τ={tau:2d}min: n={len(sub):6,} 故障率={pos/len(sub)*100:5.2f}% '
          f'PR-AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}', flush=True)

# ============ 0b 启动型可分性（同分布） ============
print('\n[2/4] 0b 启动型可分性：启动型 vs 正常短事务(<30min)...', flush=True)
for tau in [1, 2, 3, 5, 10]:
    sub = feat[(feat['prefix_type'] == 'time') & (feat['prefix_val'] == tau)].copy()
    # 正样本=启动型故障；负样本=正常事务且 dur<30min（"正常短事务"，与启动型同属短会话族群）
    sub['is_startup'] = (sub['family'] == 'startup').astype(int)
    sub['is_norm_short'] = (sub['family'] == 'normal') & (sub['dur_min'] < 30)
    sp = sub[sub['is_startup'] == 1]
    ns = sub[sub['is_norm_short'] == 1]
    # 负样本可能远多于正样本 → 抽样平衡到 2:1 以内（R6 精神，可控方差）
    if len(ns) > 2 * len(sp):
        ns = ns.sample(n=2 * len(sp), random_state=0)
    sub2 = pd.concat([sp, ns])
    if len(sp) < 200 or len(ns) < 200:
        results['0b_startup_sep'][str(tau)] = {'n_pos': int(len(sp)), 'n_neg': int(len(ns)),
                                                'pr_auc_mean': None, 'note': '样本过少'}
        print(f'  τ={tau:2d}min: 样本过少 (pos={len(sp)}, neg={len(ns)})', flush=True)
        continue
    X = sub2[feat_cols].values; y = sub2['is_startup'].values
    pos = int(y.sum()); neg = int((y == 0).sum())
    neg_frac = neg / max(pos, 1)
    aucs = []
    for sd in SEEDS:
        tr, te = train_test_split(sub2, test_size=0.25, stratify=y, random_state=sd)
        Xtr, ytr = tr[feat_cols].values, tr['is_startup'].values
        Xte, yte = te[feat_cols].values, te['is_startup'].values
        a, _ = lgb_pr_auc(Xtr, ytr, Xte, yte, sd, neg_frac=neg_frac)
        aucs.append(a)
    results['0b_startup_sep'][str(tau)] = {'n_pos': int(pos), 'n_neg': int(neg),
        'pr_auc_mean': float(np.mean(aucs)), 'pr_auc_std': float(np.std(aucs)),
        'pr_auc_per_seed': [round(a, 4) for a in aucs]}
    print(f'  τ={tau:2d}min: 启动型正样本={pos:,} 正常短负样本={neg:,} '
          f'PR-AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}', flush=True)

# ============ 0c 跨站可达性（owner1-6 → 7-8） ============
print('\n[3/4] 0c 跨站冷启动：owner1-6 训练 → owner7-8 测试 ...', flush=True)
for tau in [3, 5, 10]:
    sub = feat[(feat['prefix_type'] == 'time') & (feat['prefix_val'] == tau)].copy()
    tr = sub[sub['owner'].isin([f'Sheet{i}' for i in range(1, 7)])]
    te = sub[sub['owner'].isin(['Sheet7', 'Sheet8'])]
    if len(tr) < 300 or len(te) < 100:
        results['0c_cross_station'][str(tau)] = {'note': '样本过少', 'n_train': int(len(tr)), 'n_test': int(len(te))}
        continue
    Xtr, ytr = tr[feat_cols].values, tr['label'].values
    Xte, yte = te[feat_cols].values, te['label'].values
    pos_tr = int(ytr.sum()); neg_tr = int((ytr == 0).sum())
    neg_frac = neg_tr / max(pos_tr, 1)
    aucs_all, aucs_s7, aucs_s8 = [], [], []
    for sd in SEEDS:
        a, p = lgb_pr_auc(Xtr, ytr, Xte, yte, sd, neg_frac=neg_frac)
        aucs_all.append(a)
        m7 = te['owner'] == 'Sheet7'; m8 = te['owner'] == 'Sheet8'
        aucs_s7.append(average_precision_score(yte[m7], p[m7]) if m7.sum() else np.nan)
        aucs_s8.append(average_precision_score(yte[m8], p[m8]) if m8.sum() else np.nan)
    results['0c_cross_station'][str(tau)] = {
        'n_train': int(len(tr)), 'n_test': int(len(te)),
        'test_pos_rate': float(yte.mean()), 'test_pos': int(yte.sum()),
        'pr_auc_all': float(np.mean(aucs_all)), 'pr_auc_s7': float(np.nanmean(aucs_s7)),
        'pr_auc_s8': float(np.nanmean(aucs_s8)),
        'per_seed_all': [round(a, 4) for a in aucs_all]}
    print(f'  τ={tau:2d}min: train={len(tr):,}(故障率{pos_tr/len(tr)*100:.1f}%) '
          f'test={len(te):,}(故障率{yte.mean()*100:.1f}%) '
          f'PR-AUC_all={np.mean(aucs_all):.4f} | Sheet7={np.nanmean(aucs_s7):.4f} '
          f'| Sheet8={np.nanmean(aucs_s8):.4f}', flush=True)

# ============ 0d 时间 vs 进度切窗敏感性 ============
print('\n[4/4] 0d 时间 vs 进度切窗 PR-AUC 对照 ...', flush=True)
# 进度切窗：同分布分层 LightGBM
for pp in PROG_PCTS:
    sub = feat[(feat['prefix_type'] == 'progress') & (feat['prefix_val'] == pp)].copy()
    X = sub[feat_cols].values; y = sub['label'].values
    pos = int(y.sum()); neg = int((y == 0).sum())
    neg_frac = neg / max(pos, 1)
    aucs = []
    for sd in SEEDS:
        tr, te = split_by_owner_label(sub, seed=sd)
        a, _ = lgb_pr_auc(tr[feat_cols].values, tr['label'].values,
                          te[feat_cols].values, te['label'].values, sd, neg_frac=neg_frac)
        aucs.append(a)
    results['0d_cut_sensitivity'][f'progress@{pp}%'] = {'n': int(len(sub)),
        'pr_auc_mean': float(np.mean(aucs)), 'pr_auc_std': float(np.std(aucs))}
    print(f'  progress@{pp:3d}%: n={len(sub):,} 故障率={pos/len(sub)*100:.2f}% '
          f'PR-AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}', flush=True)
# 时间切窗 0a 数字并入 0d
for tau, v in results['0a_information_location'].items():
    if v.get('pr_auc_mean'):
        results['0d_cut_sensitivity'][f'time@{tau}min'] = {'n': v['n'],
            'pr_auc_mean': v['pr_auc_mean'], 'pr_auc_std': v['pr_auc_std']}

os.makedirs(os.path.dirname(OUTJSON), exist_ok=True)
with open(OUTJSON, 'w', encoding='utf-8') as fp:
    json.dump(results, fp, ensure_ascii=False, indent=2, default=str)
print(f'\n结果已保存: {OUTJSON}', flush=True)

# ============ 决策门汇总 ============
print('\n=========== Phase 0 决策门判定 ===========', flush=True)
print('0a 信息位置（标准：任一 τ LightGBM PR-AUC ≥ 0.65）:')
for tau, v in results['0a_information_location'].items():
    m = v.get('pr_auc_mean')
    print(f'  τ={tau:>2}min: {"%.4f"%m if m else "N/A"} {"✅" if (m and m>=0.65) else "❌"}', flush=True)
print('0b 启动型可分性（标准：≥0.60）:')
for tau, v in results['0b_startup_sep'].items():
    m = v.get('pr_auc_mean')
    print(f'  τ={tau:>2}min: {"%.4f"%m if m else "N/A"} {"✅" if (m and m>=0.60) else "❌"}', flush=True)
print('0c 跨站可达性（标准：平均≥0.55 且非 Sheet7 独撑）:')
for tau, v in results['0c_cross_station'].items():
    print(f'  τ={tau:>2}min: all={v.get("pr_auc_all", "N/A")} s7={v.get("pr_auc_s7", "N/A")} '
          f's8={v.get("pr_auc_s8", "N/A")}', flush=True)
print('0d 时间/进度切窗趋势:', flush=True)
for k, v in results['0d_cut_sensitivity'].items():
    print(f'  {k}: {"%.4f"%v["pr_auc_mean"] if v.get("pr_auc_mean") else "N/A"}', flush=True)
print(f'\n总耗时 {time.time()-t0:.0f}s', flush=True)
