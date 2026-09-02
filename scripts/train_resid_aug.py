#!/usr/bin/env python3
"""P1c: 分解残差化 + 正类条件增强 组合 (复刻导师综述 U-Net-DeWA 制胜逻辑)。

导师综述 Table 2 的制胜点是「分解(DeW) + 数据增强(DeWA)」组合, 而非单独任一:
  U-Net-Raw 0.403 → U-Net-DeW 0.662 → U-Net-DeWA 0.693 (+0.29 F1)。
今晚单独测过 P1a 增强(负) 和 P1b 残差化(负), 但没测组合。本脚本:
  1) 对 6 通道序列去趋势(居中滑动均值)得到残差;
  2) 对残差序列的正类(642 故障)做条件增强(每条 k 份, 随机 jitter/scaling/magwarp/timewarp/winslice);
  3) 残差+增强数据一起训练 Token-Attn。62 维特征不变。
用法: DEVICE=cuda BATCH=64 SEED=42 WINDOW=20 AUG_K=4 python train_resid_aug.py
输出: docs/c1c2_tokenattn_resid_w{WINDOW}_augK{AUG_K}_s{seed}_results.json
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
WINDOW = int(os.environ.get('WINDOW', 20))
AUG_K = int(os.environ.get('AUG_K', 4))
EPOCHS = int(os.environ.get('EPOCHS', 30))
BATCH = int(os.environ.get('BATCH', 64))
torch.manual_seed(SEED); np.random.seed(SEED)
D = T.D
device = T.device


def detrend(x, window):
    L = x.shape[0]
    if L < 3:
        return x - x.mean(0, keepdims=True)
    w = min(window, L)
    kernel = np.ones(w, dtype=np.float64) / w
    trend = np.stack([np.convolve(x[:, c], kernel, mode='same') for c in range(x.shape[1])], axis=1)
    return x - trend.astype(np.float32)


def aug_jitter(x, sigma=0.08):
    return x + np.random.normal(0, sigma, size=x.shape).astype(np.float32)

def aug_scaling(x, lo=0.75, hi=1.25):
    s = np.random.uniform(lo, hi, size=(1, x.shape[1])).astype(np.float32)
    return (x * s).astype(np.float32)

def aug_magnitude_warp(x, sigma=0.25):
    L, C = x.shape
    steps = np.random.normal(0, sigma / max(1, np.sqrt(L)), size=(L, C))
    curve = 1 + np.cumsum(steps, axis=0)
    curve = curve / (curve.mean(axis=0, keepdims=True) + 1e-8)
    return (x * curve.astype(np.float32)).astype(np.float32)

def aug_time_warp(x, sigma=0.2):
    L, C = x.shape
    if L < 8:
        return x
    t = np.arange(L); n_knots = 4
    knots = np.linspace(0, L - 1, n_knots)
    vals = np.clip(knots + np.random.normal(0, sigma * L / n_knots, size=n_knots), 0, L - 1)
    vals[0] = 0; vals[-1] = L - 1
    new_t = np.clip(np.interp(t, knots, vals), 0, L - 1)
    return np.stack([np.interp(new_t, t, x[:, c]) for c in range(C)], axis=1).astype(np.float32)

def aug_window_slice(x):
    L, C = x.shape
    if L < 10:
        return x
    w = np.random.randint(int(L * 0.5), L)
    start = np.random.randint(0, L - w + 1)
    seg = x[start:start + w]
    t_old = np.linspace(0, 1, w); t_new = np.linspace(0, 1, L)
    return np.stack([np.interp(t_new, t_old, seg[:, c]) for c in range(C)], axis=1).astype(np.float32)

AUG = [aug_jitter, aug_scaling, aug_magnitude_warp, aug_time_warp, aug_window_slice]

def apply_aug(x):
    return AUG[np.random.randint(len(AUG))](x)

# 1) 残差化所有序列
print(f'Residualize (window={WINDOW}) ...', flush=True)
resid_tr = [detrend(s, WINDOW) for s in D['train']['X_tensor']]
resid_va = [detrend(s, WINDOW) for s in D['val']['X_tensor']]
resid_te = [detrend(s, WINDOW) for s in D['test']['X_tensor']]

# 2) 正类条件增强
Ftr = D['train']['X_feat'].astype(np.float32)
ytr = D['train']['y']
n_pos = int((ytr == 1).sum())
print(f'Train: {len(resid_tr)} seqs ({n_pos} positive), AUG_K={AUG_K}', flush=True)
raw_list = list(resid_tr); feat_list = list(Ftr); y_list = list(ytr)
for i in range(len(resid_tr)):
    if ytr[i] != 1:
        continue
    for _ in range(AUG_K):
        raw_list.append(apply_aug(resid_tr[i]))
        feat_list.append(Ftr[i])
        y_list.append(1)

Xtr, ltr = T.pad(raw_list)
Ftr_aug = np.stack(feat_list, 0)
ytr_aug = np.array(y_list, dtype=np.int64)
Xva, lva = T.pad(resid_va); yva = D['val']['y']
Xte, lte = T.pad(resid_te); yte = D['test']['y']
Fva = D['val']['X_feat'].astype(np.float32)
Fte = D['test']['X_feat'].astype(np.float32)
print(f'Aug train: {Xtr.shape} feat {Ftr_aug.shape} y+={(ytr_aug==1).sum()}', flush=True)

from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, recall_score, precision_score, accuracy_score

model = T.TokenAttnFusion(T.FEAT_DIM).to(device)
pos_w = float((ytr_aug == 0).sum()) / max(1, int((ytr_aug == 1).sum()))
w = torch.tensor([1.0, pos_w], dtype=torch.float32).to(device)
crit = nn.CrossEntropyLoss(weight=w)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

Xtr_t = torch.FloatTensor(Xtr).to(device); ltr_t = torch.LongTensor(ltr)
Xva_t = torch.FloatTensor(Xva).to(device); lva_t = torch.LongTensor(lva)
Xte_t = torch.FloatTensor(Xte).to(device); lte_t = torch.LongTensor(lte)
Ftr_t = torch.FloatTensor(Ftr_aug).to(device); Fva_t = torch.FloatTensor(Fva).to(device); Fte_t = torch.FloatTensor(Fte).to(device)
ytr_t = torch.LongTensor(ytr_aug).to(device); yva_t = torch.LongTensor(yva).to(device)

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
    if (ep + 1) % 5 == 0 or ep == EPOCHS - 1:
        print(f'  [resid_aug w{WINDOW} k{AUG_K}] ep{ep+1}/{EPOCHS} loss={tot:.3f} val_f1={vf1:.4f} best={best_f1:.4f}(ep{best_ep})', flush=True)

model.load_state_dict(best_state); model.eval()
with torch.no_grad():
    prob = torch.softmax(model(Xte_t, lte_t, Fte_t), 1)[:, 1].cpu().numpy()
    pred = prob >= 0.5
    vprob = torch.softmax(model(Xva_t, lva_t, Fva_t), 1)[:, 1].cpu().numpy()
best_th, best_cf1 = 0.5, 0
for th in np.arange(0.05, 0.96, 0.01):
    f1 = f1_score(yva, (vprob >= th).astype(int), zero_division=0)
    if f1 > best_cf1: best_cf1, best_th = f1, float(th)
cpred = (prob >= best_th).astype(int)
res = {
    'PR-AUC': float(average_precision_score(yte, prob)),
    'AUC': float(roc_auc_score(yte, prob)),
    'F1_05': float(f1_score(yte, pred, zero_division=0)),
    'Recall_05': float(recall_score(yte, pred)),
    'best_epoch': best_ep, 'best_th': best_th,
    'calib_F1': float(f1_score(yte, cpred, zero_division=0)),
    'sec': round(time.time() - t0, 1), 'WINDOW': WINDOW, 'AUG_K': AUG_K,
}
out = {'tokenattn_resid_aug': res,
       'meta': {'seed': SEED, 'WINDOW': WINDOW, 'AUG_K': AUG_K, 'pos_weight': pos_w,
                'device': str(device), 'epochs': EPOCHS}}
sfix = f'_s{SEED}' if SEED != 42 else ''
name = f'c1c2_tokenattn_resid_w{WINDOW}_augK{AUG_K}{sfix}'
json.dump(out, open(f'{T.OUT}/{name}_results.json', 'w'), indent=2)
np.save(f'{T.OUT}/{name}_prob.npy', prob)
print(f'\n=== tokenattn_resid_aug (W={WINDOW}, K={AUG_K}, seed={SEED}) ===', flush=True)
for k in ['PR-AUC', 'AUC', 'F1_05', 'Recall_05']:
    print(f'  {k}: {res[k]:.4f}', flush=True)
print(f'  best_th={best_th:.2f} calib_F1={res["calib_F1"]:.4f} ({res["sec"]}s)', flush=True)
print('DONE', flush=True)
