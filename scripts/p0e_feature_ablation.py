#!/usr/bin/env python3
"""P0e: 特征组消融 —— 逐组去掉 5 类特征重训 Token-Attn, 量化各组贡献。

特征分组与 make_tokenattn_attn_figures.py / 实验探索全记录 3.2 完全一致:
  基础统计量 32 / 温度变化 6 / 分段端点差分 15 / 超温标志 4 / 电池类型 5  = 62
做法: 固定序列分支不动, 只从特征分支移除某一组(其余 4 组保留), 重训 30 epoch。
  若某组移除后 PR-AUC 明显下降 → 该组对检测是主信号; 下降小 → 冗余。
输出: docs/p0e_feature_ablation.json
用法: DEVICE=cuda BATCH=64 SEED=42 [ONLY=basic,tempchg,segdiff,overtemp,batt] python p0e_feature_ablation.py
"""
import os, sys, pickle, warnings, json, time
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_num_threads(4)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_c1c2 as T

SEED = int(os.environ.get('SEED', 42))
EPOCHS = int(os.environ.get('EPOCHS', 30))
BATCH = int(os.environ.get('BATCH', 64))
torch.manual_seed(SEED); np.random.seed(SEED)
D = T.D
device = T.device
FEAT_COLS = D['feat_cols']

GROUPS = {
    'basic':    {'v_mean','v_std','v_min','v_max','v_first','v_last','v_slope',
                 'a_mean','a_std','a_min','a_max','a_first','a_last',
                 'p_mean','p_std','p_max','p_min','p_first','p_last',
                 'soc_first','soc_last','soc_delta','t1_mean','t1_max',
                 't2_mean','t2_max','t1_last','t2_last','n_points',
                 'duration_min','total_kwh','p_v_ratio'},
    'tempchg':  {'t1_slope','t2_slope','t1_max_jump','t2_max_jump','t1_std_2nd','t2_std_2nd'},
    'segdiff':  {'v_first_third_min','v_sag_from_mean','a_last_third_max','a_first_last_ratio',
                 'p_max_jump','soc_rate','soc_last_rate','v_seg_change_1to2','v_seg_change_2to3',
                 'a_seg_change_1to2','a_seg_change_2to3','a_seg3_max',
                 'v_last3_vs_first3','a_last3_vs_first3','p_last3_vs_first3'},
    'overtemp': {'t1_over_40','t2_over_40','t1_over_45','t2_over_45'},
    'batt':     {'bt_LFP','bt_NMC','bt_LMO','bt_LCO','bt_LP'},
}
# 校验分组覆盖全部 62 列且不重不漏
all_grouped = set().union(*GROUPS.values())
missing = set(FEAT_COLS) - all_grouped
extra = all_grouped - set(FEAT_COLS)
if missing or extra:
    print(f'[WARN] group mismatch: missing={missing} extra={extra}', flush=True)

from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

def train_with_feats(name, feat_dim, Ftr, Fva, Fte):
    """用给定特征子集重训 Token-Attn, 返回 PR-AUC 等指标。"""
    Xtr, ltr = T.pad(D['train']['X_tensor']); ytr = D['train']['y']
    Xva, lva = T.pad(D['val']['X_tensor']); yva = D['val']['y']
    Xte, lte = T.pad(D['test']['X_tensor']); yte = D['test']['y']
    model = T.TokenAttnFusion(feat_dim).to(device)
    pos_w = float((ytr == 0).sum()) / max(1, int((ytr == 1).sum()))
    w = torch.tensor([1.0, pos_w], dtype=torch.float32).to(device)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    Xtr_t = torch.FloatTensor(Xtr).to(device); ltr_t = torch.LongTensor(ltr)
    Xva_t = torch.FloatTensor(Xva).to(device); lva_t = torch.LongTensor(lva)
    Xte_t = torch.FloatTensor(Xte).to(device); lte_t = torch.LongTensor(lte)
    Ftr_t = torch.FloatTensor(Ftr).to(device); Fva_t = torch.FloatTensor(Fva).to(device); Fte_t = torch.FloatTensor(Fte).to(device)
    ytr_t = torch.LongTensor(ytr).to(device); yva_t = torch.LongTensor(yva).to(device)

    best_f1, best_ep, best_state = 0, 0, None
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(len(Xtr_t)); tot = 0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            if len(idx) < 2:
                continue
            out = model(Xtr_t[idx], ltr_t[idx], Ftr_t[idx])
            loss = crit(out, ytr_t[idx])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += loss.item()
        sched.step(); model.eval()
        with torch.no_grad():
            vo = model(Xva_t, lva_t, Fva_t); vf1 = f1_score(yva, vo.argmax(1).cpu().numpy(), zero_division=0)
        if vf1 > best_f1:
            best_f1 = vf1; best_ep = ep + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        prob = torch.softmax(model(Xte_t, lte_t, Fte_t), 1)[:, 1].cpu().numpy()
    return {'PR-AUC': float(average_precision_score(yte, prob)),
            'AUC': float(roc_auc_score(yte, prob)),
            'F1_05': float(f1_score(yte, (prob >= 0.5).astype(int), zero_division=0)),
            'n_feat': feat_dim, 'best_epoch': best_ep, 'sec': round(time.time() - t0, 1)}

# 全量特征索引 + 各消融索引
full_idx = list(range(len(FEAT_COLS)))
name_to_idx = {c: i for i, c in enumerate(FEAT_COLS)}

results = {}
print(f'=== full (62) ===', flush=True)
results['full'] = train_with_feats('full', len(full_idx),
                                   D['train']['X_feat'].astype(np.float32),
                                   D['val']['X_feat'].astype(np.float32),
                                   D['test']['X_feat'].astype(np.float32))
print(f'  full PR-AUC={results["full"]["PR-AUC"]:.4f} ({results["full"]["sec"]}s)', flush=True)

only = os.environ.get('ONLY', '')
for gname in ['basic', 'tempchg', 'segdiff', 'overtemp', 'batt']:
    if only and gname not in only:
        continue
    keep_idx = [i for i, c in enumerate(FEAT_COLS) if c not in GROUPS[gname]]
    if len(keep_idx) == len(full_idx):
        print(f'[WARN] {gname} removed nothing, skip', flush=True); continue
    Ftr = D['train']['X_feat'][:, keep_idx].astype(np.float32)
    Fva = D['val']['X_feat'][:, keep_idx].astype(np.float32)
    Fte = D['test']['X_feat'][:, keep_idx].astype(np.float32)
    print(f'=== remove {gname} ({len(keep_idx)} feat) ===', flush=True)
    results[f'remove_{gname}'] = train_with_feats(f'remove_{gname}', len(keep_idx), Ftr, Fva, Fte)
    print(f'  remove_{gname} PR-AUC={results[f"remove_{gname}"]["PR-AUC"]:.4f}', flush=True)

full_pr = results['full']['PR-AUC']
print('\n=== P0e 特征组消融 ===', flush=True)
for k, v in results.items():
    drop = full_pr - v['PR-AUC']
    print(f'  {k:16s} PR-AUC={v["PR-AUC"]:.4f}  Δ(相对full)={-drop:+.4f}  n_feat={v["n_feat"]}', flush=True)
json.dump({'full_PR-AUC': full_pr, 'results': results,
           'group_sizes': {k: len(v) for k, v in GROUPS.items()}},
          open(f'{T.OUT}/p0e_feature_ablation.json', 'w'), indent=2, ensure_ascii=False)
print('DONE', flush=True)
