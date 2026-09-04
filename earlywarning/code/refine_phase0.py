#!/usr/bin/env python3
"""refine_phase0.py — Phase 0 诚实性精查（决策门通过后的负对照 + 风险细化）

目的（手册 E10/诚实纪律）：
1. 0a 负对照：label 随机打乱后 LightGBM PR-AUC → 应回到正样本率基线（~0.08）
   —— 防"时长代理/口径泄漏"造成的虚高
2. 0c 细化：逐 owner 的 n / 故障率 / PR-AUC / 宏平均 —— Sheet8 弱是样本少还是真不可分
3. 特征重要性 top：确认模型在用物理信号而非时长代理
4. 前缀内行数 n_prefix_rows 与标签的关系：确认 n_prefix 不泄漏时长（τ 固定下应接近恒定）
"""
import pandas as pd
import numpy as np
import json, os
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import lightgbm as lgb

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT = os.path.join(BASE, 'data', 'prefix_feats.parquet')
META = os.path.join(BASE, 'data', 'probe_dataset.parquet')
OUTJSON = os.path.join(BASE, 'docs', 'gate_phase0_results.json')

SEED = 0
DROP_COLS = ['transaction_id', 'prefix_type', 'prefix_val', 'n_prefix_rows', 'owner', 'family', 'label', 'dur_min']

def lgb_model(Xtr, ytr, seed=SEED):
    pos = int(ytr.sum()); neg = int((ytr == 0).sum())
    m = lgb.LGBMClassifier(objective='binary', learning_rate=0.05, num_leaves=31,
                           min_child_samples=30, n_estimators=400, random_state=seed,
                           n_jobs=-1, verbosity=-1,
                           scale_pos_weight=neg / max(pos, 1))
    m.fit(Xtr, ytr)
    return m

print('[1/5] 载入 ...', flush=True)
feat = pd.read_parquet(FEAT)
meta = pd.read_parquet(META)
dur_map = meta.set_index('transaction_id')['dur_min']
feat['dur_min'] = feat['transaction_id'].map(dur_map)
feat_cols = [c for c in feat.columns if c not in DROP_COLS]
print(f'  特征 {len(feat_cols)} 维', flush=True)

with open(OUTJSON, encoding='utf-8') as fp:
    res = json.load(fp)

# ============ 1. 0a 负对照（label shuffle） ============
print('\n[2/5] 0a 负对照：label 打乱 → 期望 PR-AUC≈正样本率 ...', flush=True)
neg_ctl = {}
for tau in [1, 5, 10]:
    sub = feat[(feat['prefix_type'] == 'time') & (feat['prefix_val'] == tau)].copy()
    strat = sub['owner'].astype(str) + '_' + sub['label'].astype(str)
    vc = strat.value_counts(); rare = set(vc[vc < 4].index)
    strat2 = strat.apply(lambda s: ('other_' + s.split('_')[-1]) if s in rare else s)
    tr, te = train_test_split(sub, test_size=0.25, stratify=strat2, random_state=SEED)
    # 在 train 上打乱 label（保持比例），test 保持真实 label 计算 baseline PR-AUC
    ytr_shuf = tr['label'].values.copy()
    rng = np.random.default_rng(SEED)
    ytr_shuf = rng.permutation(ytr_shuf)
    m = lgb_model(tr[feat_cols].values, ytr_shuf, seed=SEED)
    p = m.predict_proba(te[feat_cols].values)[:, 1]
    base = te['label'].mean()
    a = average_precision_score(te['label'].values, p)
    neg_ctl[str(tau)] = {'pos_rate_test': float(base), 'pr_auc_shuffled_label': float(a)}
    print(f'  τ={tau:2d}min: 正样本率基线={base:.4f} | 打乱label后 PR-AUC={a:.4f} '
          f'({"≈基线✅无泄漏" if abs(a-base)<0.03 else "⚠️异常"})', flush=True)
res['neg_control_label_shuffle'] = neg_ctl

# ============ 2. 0c 细化：逐 owner ============
print('\n[3/5] 0c 细化：逐 owner 跨站（τ=3/5/10） ...', flush=True)
for tau in [3, 5, 10]:
    sub = feat[(feat['prefix_type'] == 'time') & (feat['prefix_val'] == tau)].copy()
    tr = sub[sub['owner'].isin([f'Sheet{i}' for i in range(1, 7)])]
    te = sub[sub['owner'].isin(['Sheet7', 'Sheet8'])]
    m = lgb_model(tr[feat_cols].values, tr['label'].values, seed=SEED)
    p_all = m.predict_proba(te[feat_cols].values)[:, 1]
    detail = {}
    macro = []
    for ow in ['Sheet7', 'Sheet8']:
        mo = te['owner'] == ow
        y = te.loc[mo, 'label'].values; p = p_all[mo]
        a = average_precision_score(y, p)
        macro.append(a)
        detail[ow] = {'n': int(mo.sum()), 'pos': int(y.sum()),
                      'pos_rate': float(y.mean()), 'pr_auc': float(a)}
        print(f'  τ={tau:2d}min {ow}: n={int(mo.sum()):5d} 故障={int(y.sum()):4d} '
              f'({y.mean()*100:5.2f}%) PR-AUC={a:.4f}', flush=True)
    detail['macro_mean'] = float(np.mean(macro))
    # test 整体正样本率（PR-AUC 随机基线）
    detail['test_pos_rate_pooled'] = float(te['label'].mean())
    res['0c_cross_station'][str(tau)]['per_owner_detail'] = detail
    print(f'  → 宏平均={np.mean(macro):.4f} | pooled正样本率={te["label"].mean():.4f}', flush=True)

# ============ 3. 特征重要性 top（τ=5min 同分布） ============
print('\n[4/5] 特征重要性 top12（τ=5min，同分布模型）...', flush=True)
sub = feat[(feat['prefix_type'] == 'time') & (feat['prefix_val'] == 5)].copy()
strat = sub['owner'].astype(str) + '_' + sub['label'].astype(str)
vc = strat.value_counts(); rare = set(vc[vc < 4].index)
strat2 = strat.apply(lambda s: ('other_' + s.split('_')[-1]) if s in rare else s)
tr, te = train_test_split(sub, test_size=0.25, stratify=strat2, random_state=SEED)
m = lgb_model(tr[feat_cols].values, tr['label'].values, seed=SEED)
imp = sorted(zip(feat_cols, m.feature_importances_), key=lambda x: -x[1])[:12]
for name, v in imp:
    print(f'  {name:30s} {v:6d}', flush=True)
res['feature_importance_tau5'] = [{'feature': n, 'gain': int(v)} for n, v in imp]

# ============ 4. n_prefix_rows 与 label / 时长的关系 ============
print('\n[5/5] n_prefix_rows 泄漏检查：τ 固定下应与 dur 无关（时长代理排查）...', flush=True)
res.setdefault('n_prefix_leak_check', {})
for tau in [5, 10]:
    sub = feat[(feat['prefix_type'] == 'time') & (feat['prefix_val'] == tau)].copy()
    corr = sub['n_prefix_rows'].corr(sub['dur_min'])
    # 正常 vs 故障 的 n_prefix_rows 均值差（若故障的长得短，n 会系统性小 → 模型可能靠 n 偷时长）
    g = sub.groupby('label')['n_prefix_rows'].mean()
    print(f'  τ={tau:2d}min: n_prefix 与 dur 相关={corr:.4f} | 正常行数均={g.get(0,0):.2f} '
          f'故障行数均={g.get(1,0):.2f} 差={g.get(1,0)-g.get(0,0):+.2f}', flush=True)
    res['n_prefix_leak_check'][str(tau)] = {'corr_with_dur': float(corr),
        'mean_n_normal': float(g.get(0, 0)), 'mean_n_fault': float(g.get(1, 0))}

with open(OUTJSON, 'w', encoding='utf-8') as fp:
    json.dump(res, fp, ensure_ascii=False, indent=2, default=str)
print(f'\n精查结果已并入 {OUTJSON}', flush=True)
