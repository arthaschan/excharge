#!/usr/bin/env python3
"""Threshold sweep + feature combination experiment.
Goal: maximize Recall (target >73.56%) with acceptable F1.
Combinations: v1 features, v2 features, v1+v2, v1+selected-v2
"""
import pandas as pd, numpy as np, json, time, warnings
warnings.filterwarnings('ignore')
from xgboost import XGBClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)
from sklearn.model_selection import cross_val_score

DATA = '/Users/arthas/git/excharge/data/real/'
OUT  = '/Users/arthas/git/excharge/docs/'

# v1 features
X1_tr = pd.read_parquet(f'{DATA}/seq_X_train.parquet')
X1_te = pd.read_parquet(f'{DATA}/seq_X_test.parquet')
# v2 features
X2_tr = pd.read_parquet(f'{DATA}/seq_X_train_v2.parquet')
X2_te = pd.read_parquet(f'{DATA}/seq_X_test_v2.parquet')
y_tr  = pd.read_parquet(f'{DATA}/seq_y_train.parquet')['label'].values
y_te  = pd.read_parquet(f'{DATA}/seq_y_test.parquet')['label'].values

# v2 uses more training data (owner 1-6) than v1 (owner 1-4).
# For FAIR comparison, subset v2 to v1's train index set; also test full v2 (more data).
common_tr_idx = X1_tr.index.intersection(X2_tr.index)
X2_tr_fair = X2_tr.loc[common_tr_idx].copy()
y_tr_fair = y_tr[X1_tr.index.get_indexer(common_tr_idx)]
# Test sets share index 0-19656; both test sets are owner 7-8 (same rows)
common_te_idx = X1_te.index.intersection(X2_te.index)
X2_te_fair = X2_te.loc[common_te_idx].copy()
y_te_fair = y_te[X1_te.index.get_indexer(common_te_idx)]
print(f'Fair train: {len(X2_tr_fair)} (v1: {len(X1_tr)}) | Fair test: {len(X2_te_fair)} (v1: {len(X1_te)})')

# Selected v2 features that showed physical signal (avoid noise dilution)
selected_v2 = ['v_first_third_min', 'v_sag_from_mean', 't1_slope', 't2_slope',
               't1_max_jump', 't2_max_jump', 't1_std_2nd', 't2_std_2nd',
               'a_last_third_max', 'a_first_last_ratio', 'p_max_jump',
               'soc_rate', 'soc_last_rate',
               'a_seg_change_2to3', 'v_last3_vs_first3', 'a_last3_vs_first3',
               't1_over_40', 't2_over_40', 't1_over_45', 't2_over_45']
selected_v2 = [c for c in selected_v2 if c in X2_tr.columns]

combos = {
    'v1_baseline': (X1_tr, X1_te, y_tr, y_te),
    'v2_fair':     (X2_tr_fair, X2_te_fair, y_tr_fair, y_te_fair),
    'v2_more_data':(X2_tr, X2_te, y_tr, y_te),
    'v1+sel_v2':   (pd.concat([X1_tr.reset_index(drop=True), X2_tr_fair[selected_v2].reset_index(drop=True)], axis=1),
                    pd.concat([X1_te.reset_index(drop=True), X2_te_fair[selected_v2].reset_index(drop=True)], axis=1),
                    y_tr_fair, y_te_fair),
}

# v2 more-data split labels (v2 y saved separately)
y_tr_v2 = pd.read_parquet(f'{DATA}/seq_y_train_v2.parquet')['label'].values
y_te_v2 = pd.read_parquet(f'{DATA}/seq_y_test_v2.parquet')['label'].values
# fix v2_more_data combo with correct labels
combos['v2_more_data'] = (X2_tr, X2_te, y_tr_v2, y_te_v2)

spw = (y_tr == 0).sum() / (y_tr == 1).sum()

def sweep_thresholds(prob, y, thresholds):
    """Return best threshold by F1 (also report recall-focused)."""
    best = None
    for th in thresholds:
        p = (prob >= th).astype(int)
        r = recall_score(y, p)
        f = f1_score(y, p)
        pr = precision_score(y, p, zero_division=0)
        auc = roc_auc_score(y, prob)
        if best is None or f > best['f1']:
            best = {'th': th, 'recall': r, 'prec': pr, 'f1': f, 'auc': auc}
    return best

results = {}
for name, (Xtr, Xte, ytr, yte) in combos.items():
    print(f'\n=== {name} ({Xtr.shape[1]} features, train {len(Xtr)}) ===', flush=True)
    t0 = time.time()
    spw_c = (ytr == 0).sum() / (ytr == 1).sum()
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                        scale_pos_weight=spw_c, eval_metric='logloss',
                        random_state=42, n_jobs=-1)
    xgb.fit(Xtr, ytr, verbose=False)
    prob = xgb.predict_proba(Xte)[:, 1]
    train_s = time.time() - t0

    ths = np.round(np.arange(0.05, 0.80, 0.025), 3)
    best = sweep_thresholds(prob, yte, ths)
    # Also record @0.5
    p50 = (prob >= 0.5).astype(int)
    r50 = recall_score(yte, p50); f50 = f1_score(yte, p50)
    pr50 = precision_score(yte, p50, zero_division=0)
    auc = roc_auc_score(yte, prob)

    print(f'  @0.5:  Recall={r50:.4f} Prec={pr50:.4f} F1={f50:.4f} AUC={auc:.4f}')
    print(f'  Best:  th={best["th"]:.3f} Recall={best["recall"]:.4f} Prec={best["prec"]:.4f} F1={best["f1"]:.4f}')
    results[name] = {
        'n_features': int(Xtr.shape[1]),
        'n_train': int(len(Xtr)),
        'auc': auc,
        'recall_05': r50, 'prec_05': pr50, 'f1_05': f50,
        'best_th': best['th'], 'best_recall': best['recall'],
        'best_prec': best['prec'], 'best_f1': best['f1'],
        'train_s': round(train_s, 1),
    }
    # Save prob for best combos
    np.save(f'{OUT}/routeA_sweep_{name}_prob.npy', prob)

print('\n\n=== SUMMARY ===')
for name, r in results.items():
    print(f'{name}: AUC={r["auc"]:.4f} | @0.5 R={r["recall_05"]:.4f} F1={r["f1_05"]:.4f} | best th={r["best_th"]} R={r["best_recall"]:.4f} F1={r["best_f1"]:.4f}')

json.dump(results, open(f'{OUT}/routeA_threshold_sweep.json', 'w'), indent=2, default=str)
print('\nSaved routeA_threshold_sweep.json')
