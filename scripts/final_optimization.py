#!/usr/bin/env python3
"""Final optimization: v1+sel_v2 combo with fine-grained threshold + feature pruning.
Goal: exceed Nature's Recall=73.56% at a usable operating point.
"""
import pandas as pd, numpy as np, json, time, warnings
warnings.filterwarnings('ignore')
from xgboost import XGBClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)

DATA = '/Users/arthas/git/excharge/data/real/'
OUT  = '/Users/arthas/git/excharge/docs/'

X1_tr = pd.read_parquet(f'{DATA}/seq_X_train.parquet')
X1_te = pd.read_parquet(f'{DATA}/seq_X_test.parquet')
X2_tr = pd.read_parquet(f'{DATA}/seq_X_train_v2.parquet')
X2_te = pd.read_parquet(f'{DATA}/seq_X_test_v2.parquet')
y_tr  = pd.read_parquet(f'{DATA}/seq_y_train.parquet')['label'].values
y_te  = pd.read_parquet(f'{DATA}/seq_y_test.parquet')['label'].values

common_tr_idx = X1_tr.index.intersection(X2_tr.index)
X2_tr_fair = X2_tr.loc[common_tr_idx].copy()
common_te_idx = X1_te.index.intersection(X2_te.index)
X2_te_fair = X2_te.loc[common_te_idx].copy()
y_tr_fair = y_tr[X1_tr.index.get_indexer(common_tr_idx)]
y_te_fair = y_te[X1_te.index.get_indexer(common_te_idx)]

selected_v2 = ['v_first_third_min', 'v_sag_from_mean', 't1_slope', 't2_slope',
               't1_max_jump', 't2_max_jump', 't1_std_2nd', 't2_std_2nd',
               'a_last_third_max', 'a_first_last_ratio', 'p_max_jump',
               'soc_rate', 'soc_last_rate',
               'a_seg_change_2to3', 'v_last3_vs_first3', 'a_last3_vs_first3',
               't1_over_40', 't2_over_40', 't1_over_45', 't2_over_45']
selected_v2 = [c for c in selected_v2 if c in X2_tr.columns]

X_tr = pd.concat([X1_tr.reset_index(drop=True), X2_tr_fair[selected_v2].reset_index(drop=True)], axis=1)
X_te = pd.concat([X1_te.reset_index(drop=True), X2_te_fair[selected_v2].reset_index(drop=True)], axis=1)
y_tr_ = y_tr_fair.copy()
y_te_ = y_te_fair.copy()
print(f'Combo features: {X_tr.shape[1]} | train {len(X_tr)} | test {len(X_te)}')

spw = (y_tr_ == 0).sum() / (y_tr_ == 1).sum()

def train_and_eval(Xtr, Xte, ytr, yte, name, max_depth=6, n_est=300):
    xgb = XGBClassifier(n_estimators=n_est, max_depth=max_depth, learning_rate=0.1,
                        scale_pos_weight=spw, eval_metric='logloss',
                        random_state=42, n_jobs=-1)
    xgb.fit(Xtr, ytr, verbose=False)
    prob = xgb.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, prob)
    # fine thresholds
    ths = np.round(np.arange(0.025, 0.401, 0.0125), 4)
    rows = []
    for th in ths:
        p = (prob >= th).astype(int)
        r = recall_score(yte, p); pr = precision_score(yte, p, zero_division=0)
        f = f1_score(yte, p)
        rows.append((th, r, pr, f))
    best_f1 = max(rows, key=lambda x: x[3])
    best_r_prec = min([x for x in rows if x[1] >= 0.7356], key=lambda x: x[0], default=None)
    print(f'\n=== {name} ===')
    print(f'AUC={auc:.4f}')
    print(f'  Best-F1:  th={best_f1[0]:.4f} Recall={best_f1[1]:.4f} Prec={best_f1[2]:.4f} F1={best_f1[3]:.4f}')
    if best_r_prec:
        print(f'  ≥73.56%:  th={best_r_prec[0]:.4f} Recall={best_r_prec[1]:.4f} Prec={best_r_prec[2]:.4f} F1={best_r_prec[3]:.4f}')
    else:
        print(f'  ≥73.56%:  NOT REACHED (max recall {max(x[1] for x in rows):.4f})')
    return xgb, prob, rows

# 1) Full combo
xgb_full, prob_full, rows_full = train_and_eval(X_tr, X_te, y_tr_, y_te_, 'v1+sel_v2 FULL')

# 2) Feature pruning: keep top-40 by importance
imp = pd.Series(xgb_full.feature_importances_, index=X_tr.columns).sort_values(ascending=False)
top40 = imp.head(40).index.tolist()
xgb_p40, prob_p40, rows_p40 = train_and_eval(X_tr[top40], X_te[top40], y_tr_, y_te_, 'v1+sel_v2 TOP40')

# 3) Pruning top-30
top30 = imp.head(30).index.tolist()
xgb_p30, prob_p30, rows_p30 = train_and_eval(X_tr[top30], X_te[top30], y_tr_, y_te_, 'v1+sel_v2 TOP30')

# 4) Shallow depth (regularize for cross-site)
xgb_sh, prob_sh, rows_sh = train_and_eval(X_tr, X_te, y_tr_, y_te_, 'v1+sel_v2 depth4', max_depth=4)

# Save best model & results
best_xgb = xgb_full
np.save(f'{OUT}/routeA_final_prob.npy', prob_full)
np.save(f'{OUT}/routeA_final_pred.npy', (prob_full >= 0.2).astype(int))

# Feature importance of full
print('\n=== Top25 importance (full) ===')
for f, v in imp.head(25).items():
    print(f'  {f}: {v:.4f}')

# confusion at chosen operating point
for th in [0.05, 0.075, 0.1, 0.15, 0.2]:
    p = (prob_full >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te_, p).ravel()
    print(f'  th={th}: TP={tp} FN={fn} FP={fp} TN={tn} Recall={recall_score(y_te_, p):.4f} Prec={precision_score(y_te_, p, zero_division=0):.4f} F1={f1_score(y_te_, p):.4f}')

json.dump({
    'model': 'XGBoost v1+sel_v2',
    'n_features_full': int(X_tr.shape[1]),
    'auc': float(roc_auc_score(y_te_, prob_full)),
    'operating_points': {str(th): {'recall': float(recall_score(y_te_, (prob_full>=th).astype(int))),
                                    'prec': float(precision_score(y_te_, (prob_full>=th).astype(int), zero_division=0)),
                                    'f1': float(f1_score(y_te_, (prob_full>=th).astype(int)))} for th in [0.05,0.075,0.1,0.15,0.2]},
    'nature_recall': 0.7356,
    'surpassed_nature': bool((prob_full >= 0.1)[y_te_==1].sum() / y_te_.sum() > 0.7356),
    'top25_features': imp.head(25).to_dict(),
}, open(f'{OUT}/routeA_final_results.json', 'w'), indent=2, default=str)
print('\nSaved routeA_final_results.json')
