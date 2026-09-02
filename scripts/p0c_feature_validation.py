#!/usr/bin/env python3
"""P0c: 特征验证实验 —— 新增 4 类时序特征后, GBDT 是否还能涨?

回答核心问题: 62 维手工特征是否已把时序信息"榨干"?
  - 若 62+新特征 的 PR-AUC 明显 > 0.868 (LightGBM 62维) → 时序里还有货 → 值得上预训练/更深挖掘
  - 若 ≈ 0.868 → 信号已被统计量榨干 → 后续侧重"增强正则 + 残差化"而非挖新信息

新增特征组 (全部 numpy 自实现, 无重依赖):
  A. 二阶差分: 6 通道的二阶差分 std/max (12 维)
  B. 跨通道滞后相关: 8 个关键通道对的 0-滞后相关 + 最优滞后最大相关 (20 维)
  C. 频域谱特征: v/a/p 的主频/谱质心/谱平坦度/高低频能量比 (12 维)
  D. 变点代理: v/a/p 的重标极差(Hurst代理) + 6通道符号变化 + v/a/p 滑窗均值位移 (12 维)

流程: 载入 all_data.parquet → 按 fusion_data 的 train/val/test tx 划分对齐
     → 计算新特征 → 拼接 62 维 → LightGBM/XGBoost 增量对比
输出: docs/p0c_feature_validation.json + data/real/p0c_new_features.pkl (缓存)
"""
import os, pickle, warnings, json, time
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = _ROOT + '/data/real/'
OUT = _ROOT + '/docs/'

CHANS = ['chargingv', 'charginga', 'out_power',
         'charging_gun_temperature1', 'charging_gun_temperature2', 'current_soc']
PAIRS = [('chargingv', 'charginga'), ('chargingv', 'out_power'), ('charginga', 'out_power'),
         ('charging_gun_temperature1', 'charging_gun_temperature2'), ('current_soc', 'out_power'),
         ('current_soc', 'chargingv'), ('charging_gun_temperature1', 'current_soc'),
         ('charging_gun_temperature2', 'current_soc')]
MAXLAG = 6


def _nan(x):
    return np.asarray(x, dtype=np.float64)


def feat_group_A(sub):
    """二阶差分: 各通道 diff(diff(x)) 的 std / max abs。"""
    f = {}
    for c in CHANS:
        x = _nan(sub[c].values)
        x = x[~np.isnan(x)]
        if len(x) >= 3:
            d2 = np.diff(x, n=2)
            f[f'{c}_d2_std'] = float(np.std(d2))
            f[f'{c}_d2_max'] = float(np.max(np.abs(d2)))
        else:
            f[f'{c}_d2_std'] = 0.0
            f[f'{c}_d2_max'] = 0.0
    return f


def _corr_lags(x, y, maxlag=MAXLAG):
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    m = min(len(x), len(y))
    if m < 4:
        return 0.0, 0.0, 0
    x = x[:m]; y = y[:m]
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    corr0 = float(np.dot(x, y) / m)
    best, bl = 0.0, 0
    for lag in range(1, maxlag + 1):
        c = float(np.dot(x[:-lag], y[lag:]) / (m - lag))
        if abs(c) > abs(best):
            best, bl = c, lag
        c = float(np.dot(x[lag:], y[:-lag]) / (m - lag))
        if abs(c) > abs(best):
            best, bl = c, -lag
    return corr0, best, bl


def feat_group_B(sub):
    """跨通道滞后相关。"""
    f = {}
    for c1, c2 in PAIRS:
        x = _nan(sub[c1].values); y = _nan(sub[c2].values)
        c0, cmax, lag = _corr_lags(x, y)
        key = f'{c1}__{c2}'
        f[f'{key}_corr0'] = float(c0)
        f[f'{key}_maxlag_corr'] = float(cmax)
        f[f'{key}_best_lag'] = float(lag)
    return f


def _spec(x):
    x = x[~np.isnan(x)]
    if len(x) < 8:
        return 0.0, 0.0, 0.0, 0.0
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x)) ** 2
    if spec.sum() <= 0:
        return 0.0, 0.0, 0.0, 0.0
    freqs = np.fft.rfftfreq(len(x))
    spec = spec[1:]; freqs = freqs[1:]  # 去 DC
    if len(spec) == 0 or spec.sum() <= 0:
        return 0.0, 0.0, 0.0, 0.0
    dom = float(freqs[np.argmax(spec)])
    centroid = float((freqs * spec).sum() / spec.sum())
    gm = float(np.exp(np.mean(np.log(spec + 1e-12))))
    flatness = gm / (float(spec.mean()) + 1e-12)
    q = len(spec) // 4
    low = spec[:q].sum(); high = spec[-q:].sum()
    ratio = float(low / (high + 1e-12))
    return dom, centroid, flatness, ratio


def feat_group_C(sub):
    """频域谱特征 (v/a/p)。"""
    f = {}
    for c in ['chargingv', 'charginga', 'out_power']:
        dom, cen, flat, ratio = _spec(_nan(sub[c].values))
        f[f'{c}_domfreq'] = dom
        f[f'{c}_centroid'] = cen
        f[f'{c}_flatness'] = flat
        f[f'{c}_lowhigh'] = ratio
    return f


def _rs(x):
    """重标极差 (R/S), Hurst 指数代理。"""
    x = x[~np.isnan(x)]
    if len(x) < 8:
        return 0.5
    x = x - x.mean()
    cs = np.cumsum(x)
    r = cs.max() - cs.min()
    s = x.std() + 1e-12
    return float(r / s / np.sqrt(len(x)))  # ~ (n^H)/(n^0.5), 归一化使其量级稳定


def feat_group_D(sub):
    """变点代理: 重标极差 + 符号变化数 + 滑窗均值位移。"""
    f = {}
    for c in ['chargingv', 'charginga', 'out_power']:
        f[f'{c}_rs'] = _rs(_nan(sub[c].values))
    for c in CHANS:
        x = _nan(sub[c].values)
        x = x[~np.isnan(x)]
        nsign = int(np.sum(np.diff(np.sign(np.diff(x))) != 0)) if len(x) >= 3 else 0
        f[f'{c}_nsign'] = float(nsign)
    for c in ['chargingv', 'charginga', 'out_power']:
        x = _nan(sub[c].values)
        x = x[~np.isnan(x)]
        n = len(x); w = max(4, n // 8)
        maxshift = 0.0
        if n >= 2 * w:
            for i in range(w, n - w + 1):
                s = abs(x[i:i + w].mean() - x[i - w:i].mean())
                if s > maxshift:
                    maxshift = s
        f[f'{c}_maxshift'] = float(maxshift)
    return f


def compute_for_tx(sub):
    f = {}
    f.update(feat_group_A(sub))
    f.update(feat_group_B(sub))
    f.update(feat_group_C(sub))
    f.update(feat_group_D(sub))
    return f


def main():
    t0 = time.time()
    D = pickle.load(open(f'{DATA}/fusion_data.pkl', 'rb'))
    print(f'Loaded fusion_data.pkl ({time.time()-t0:.0f}s)', flush=True)

    df = pd.read_parquet(f'{DATA}/all_data.parquet')
    print(f'Loaded all_data.parquet {len(df):,} rows ({time.time()-t0:.0f}s)', flush=True)
    groups = {tx: sub for tx, sub in df.groupby('transaction_id', sort=False)}

    # 按 fusion_data 的 tx 顺序对齐
    def build_matrix(tx_list):
        feats = []
        missing = 0
        for tx in tx_list:
            if tx not in groups:
                missing += 1
                continue
            feats.append(compute_for_tx(groups[tx]))
        if missing:
            print(f'  (warning: {missing} tx missing from parquet)', flush=True)
        return feats

    new_cols = None
    split_feats = {}
    for name in ['train', 'val', 'test']:
        print(f'Computing new features for {name}...', flush=True)
        feats = build_matrix(D[name]['tx'])
        if new_cols is None:
            new_cols = list(feats[0].keys())
        M = np.zeros((len(feats), len(new_cols)), dtype=np.float64)
        for i, fd in enumerate(feats):
            for j, c in enumerate(new_cols):
                M[i, j] = fd.get(c, 0.0)
        split_feats[name] = M

    print(f'New features: {len(new_cols)} dims ({time.time()-t0:.0f}s total)', flush=True)

    # NaN 用训练集中位数填充
    med = np.nanmedian(split_feats['train'], axis=0)
    for name in ['train', 'val', 'test']:
        M = split_feats[name]
        M = np.where(np.isnan(M), med, M)
        split_feats[name] = M

    cache = {'cols': new_cols, 'train': split_feats['train'],
             'val': split_feats['val'], 'test': split_feats['test']}
    pickle.dump(cache, open(f'{DATA}/p0c_new_features.pkl', 'wb'))
    print(f'Cached new features to p0c_new_features.pkl', flush=True)

    # ---------- GBDT 对比 ----------
    from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
    import lightgbm as lgb

    Xtr62 = D['train']['X_feat'].astype(np.float64)
    Xva62 = D['val']['X_feat'].astype(np.float64)
    Xte62 = D['test']['X_feat'].astype(np.float64)
    ytr = D['train']['y']; yva = D['val']['y']; yte = D['test']['y']

    newtr = split_feats['train']; newva = split_feats['val']; newte = split_feats['test']

    def run_lgb(Xtr, Xva, Xte, tag):
        m = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31,
                               subsample=0.8, colsample_bytree=0.8, random_state=42,
                               n_jobs=4, verbosity=-1)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric='auc',
              callbacks=[lgb.early_stopping(100, verbose=False)])
        p = m.predict_proba(Xte)[:, 1]
        return {'PR-AUC': float(average_precision_score(yte, p)),
                'AUC': float(roc_auc_score(yte, p)),
                'best_iter': int(m.best_iteration_), 'n_feat': Xtr.shape[1]}

    results = {}
    results['62feat_baseline'] = run_lgb(Xtr62, Xva62, Xte62, '62')
    # 组边界用列名前缀切分 (而非硬编码索引)
    A_cols = [i for i, c in enumerate(new_cols) if c.endswith('_d2_std') or c.endswith('_d2_max')]
    B_cols = [i for i, c in enumerate(new_cols) if '__' in c]
    C_cols = [i for i, c in enumerate(new_cols) if c.endswith(('_domfreq', '_centroid', '_flatness', '_lowhigh'))]
    D_cols = [i for i, c in enumerate(new_cols) if c.endswith(('_rs', '_nsign', '_maxshift'))]

    def cols(*grps):
        idx = []
        for g in grps:
            idx += g
        return sorted(idx)

    groups_map = {'A_二阶差分': A_cols, 'B_滞后相关': B_cols, 'C_频域谱': C_cols, 'D_变点': D_cols}

    # 增量: 62 + A, +AB, +ABC, +ABCD
    inc = [('A', cols(A_cols)), ('AB', cols(A_cols, B_cols)),
           ('ABC', cols(A_cols, B_cols, C_cols)), ('ABCD', cols(A_cols, B_cols, C_cols, D_cols))]
    for tag, idx in inc:
        results[f'62+{tag}'] = run_lgb(np.hstack([Xtr62, newtr[:, idx]]), np.hstack([Xva62, newva[:, idx]]),
                                       np.hstack([Xte62, newte[:, idx]]), f'62+{tag}')

    # 单组独立增量 (62 + 单独一组)
    for gname, idx in groups_map.items():
        results[f'62+{gname}'] = run_lgb(np.hstack([Xtr62, newtr[:, idx]]), np.hstack([Xva62, newva[:, idx]]),
                                         np.hstack([Xte62, newte[:, idx]]), f'62+{gname}')

    out = {'new_feat_cols': new_cols, 'group_sizes': {k: len(v) for k, v in groups_map.items()},
           'results': results,
           'baseline_62feat': results['62feat_baseline']}
    json.dump(out, open(f'{OUT}/p0c_feature_validation.json', 'w'), indent=2, ensure_ascii=False)
    print('\n=== P0c 特征验证 (LightGBM, owner7-8 test) ===', flush=True)
    for k, v in results.items():
        print(f'  {k:20s} PR-AUC={v["PR-AUC"]:.4f}  AUC={v["AUC"]:.4f}  n_feat={v["n_feat"]}', flush=True)
    base = results['62feat_baseline']['PR-AUC']
    best = max(results.values(), key=lambda v: v['PR-AUC'])
    print(f'\n  基线 62维 PR-AUC={base:.4f}; 最优 PR-AUC={best["PR-AUC"]:.4f} ({best["n_feat"]}维)', flush=True)
    print(f'  结论: {"时序里还有货(>0.87, 值得预训练/深挖)" if best["PR-AUC"] > base + 0.005 else "信号已被统计量榨干(≈0.868)"}', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
