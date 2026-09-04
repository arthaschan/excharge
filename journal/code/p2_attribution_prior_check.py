#!/usr/bin/env python3
"""P2 前提检查（E8 纪律）—— 归因先验跨站一致性

方案C（归因正则/辅助模块）要用"训练域归因先验"去约束/监督模型。
E8 前提：这个先验在测试域(owner7-8)是否一致？不一致则正则有害，方案C直接 no-go。

检查方法（模型无关，稳健）：
  1. 训练域(owner1-6, 13,505条/642故障) LightGBM → gain importance 排名（62维）
  2. 测试域(owner7-8, 2,776条/129故障) LightGBM → gain importance 排名（bootstrap 多次平均，缓解129故障噪声）
  3. 对比：Spearman 秩相关 + Top-10/Top-20 Jaccard 重叠 + 两域各自 Top 特征列表

判据（先定，避免事后解释）：
  - Spearman ρ ≥ 0.5 且 Top-10 重叠 ≥ 0.4 → 先验基本迁移，方案C可做；
  - 否则 → 归因先验跨站不成立，C1(归因一致性正则) no-go，仅考虑 C2(辅助头，弱先验)。

输出：journal/docs/p2_attribution_prior_check.json
"""
import pickle, numpy as np, json, time, os, warnings
import lightgbm as lgb
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE)
DATA = os.path.join(ROOT, 'data', 'real')
OUT = os.path.join(BASE, 'docs')
os.makedirs(OUT, exist_ok=True)

D = pickle.load(open(f'{DATA}/fusion_data.pkl', 'rb'))
Ftr = D['train']['X_feat'].astype(np.float32); ytr = D['train']['y']
Fva = D['val']['X_feat'].astype(np.float32);   yva = D['val']['y']
Fte = D['test']['X_feat'].astype(np.float32);  yte = D['test']['y']
cols = D['feat_cols']
print(f'cols={len(cols)} | train {Ftr.shape} fault={ytr.sum()} | test {Fte.shape} fault={yte.sum()}', flush=True)

def train_importance(F, y, seed=42, early_stop=False):
    m = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31,
                           subsample=0.8, colsample_bytree=0.8, random_state=seed,
                           n_jobs=4, verbosity=-1)
    if early_stop:
        m.fit(F, y, eval_set=[(Fva, yva)], eval_metric='auc',
              callbacks=[lgb.early_stopping(100, verbose=False)])
    else:
        m.fit(F, y)
    imp = m.booster_.feature_importance(importance_type='gain')
    return imp

# ---- 1. 训练域归因先验 ----
t0 = time.time()
imp_train = train_importance(Ftr, ytr, early_stop=True)
rank_train = np.argsort(-imp_train)   # 降序：重要性从高到低的特征索引

# ---- 2. 测试域归因（bootstrap 平均，缓解 129 故障噪声）----
# 在测试域上做 bootstrap 采样（含正负样本），多次训 LightGBM 取 gain importance 平均
rng = np.random.default_rng(0)
N_BOOT = 20
imp_boots = []
for b in range(N_BOOT):
    idx = rng.integers(0, len(Fte), len(Fte))
    Fb, yb = Fte[idx], yte[idx]
    if yb.sum() < 5 or (yb == 0).sum() < 5:   # 保证 bootstrap 样本里两类都有
        continue
    imp_boots.append(train_importance(Fb, yb, seed=b))
imp_test = np.mean(imp_boots, axis=0)
rank_test = np.argsort(-imp_test)

# ---- 3. 对比 ----
rho, pval = spearmanr(imp_train, imp_test)
def jaccard_topk(r1, r2, k):
    return len(set(r1[:k]) & set(r2[:k])) / k

top10 = jaccard_topk(rank_train, rank_test, 10)
top20 = jaccard_topk(rank_train, rank_test, 20)

res = {
    'meta': {'date': '2026-09-04', 'n_boot': len(imp_boots),
             'method': 'LightGBM gain importance; 训练域单次 vs 测试域 bootstrap 均值',
             'train': {'n': int(len(Ftr)), 'fault': int(ytr.sum())},
             'test': {'n': int(len(Fte)), 'fault': int(yte.sum())}},
    'spearman_rho': float(rho), 'spearman_p': float(pval),
    'jaccard_top10': float(top10), 'jaccard_top20': float(top20),
    'top10_train': [cols[i] for i in rank_train[:10]],
    'top10_test':  [cols[i] for i in rank_test[:10]],
    'top20_train': [cols[i] for i in rank_train[:20]],
    'top20_test':  [cols[i] for i in rank_test[:20]],
}
with open(f'{OUT}/p2_attribution_prior_check.json', 'w', encoding='utf-8') as fp:
    json.dump(res, fp, ensure_ascii=False, indent=2)

print(f'\n=== P2 前提检查（归因先验跨站一致性）===', flush=True)
print(f'  Spearman ρ = {rho:.4f} (p={pval:.4f})', flush=True)
print(f'  Top-10 Jaccard = {top10:.2f} | Top-20 Jaccard = {top20:.2f}', flush=True)
print(f'  训练域 Top-10: {res["top10_train"]}', flush=True)
print(f'  测试域 Top-10: {res["top10_test"]}', flush=True)
print(f'\n  判据: ρ≥0.5 且 Top-10≥0.4 → 先验迁移, C1 可做; 否则 C1 no-go', flush=True)
verdict = (rho >= 0.5 and top10 >= 0.4)
print(f'  => 结论: {"先验基本迁移，方案C(C1)可做 ✅" if verdict else "先验跨站不成立，C1 no-go，仅考虑 C2(弱先验) ⚠️"}', flush=True)
print(f'  耗时 {time.time()-t0:.0f}s', flush=True)
