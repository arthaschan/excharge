#!/usr/bin/env python3
"""P1a: 正类条件数据增强 → Token-Attn (导师方向主线)。

思路: 训练集正类(故障)仅 642 条, 深度模型在小样本+强类不平衡上吃亏。
对每条正类序列做条件增强(只增强正类), 生成 k 份变体, 与原数据一起训练 Token-Attn。
增强方法(时域, 长度不变): jitter / scaling / magnitude_warp / time_warp / window_slice,
每条正类随机选一种(可组合)。增强副本沿用原序列的 62 维特征(不变)。

用法:
  DEVICE=cuda BATCH=64 SEED=42 AUG_K=4 AUG_METHODS=all python train_aug.py
输出:
  docs/c1c2_tokenattn_augK{k}_s{seed}_results.json / _prob.npy / _model.pt
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
import train_c1c2 as T   # 复用模型类/数据/pad/device

SEED = int(os.environ.get('SEED', 42))
AUG_K = int(os.environ.get('AUG_K', 4))
EPOCHS = int(os.environ.get('EPOCHS', 30))
BATCH = int(os.environ.get('BATCH', 64))
torch.manual_seed(SEED); np.random.seed(SEED)

D = T.D
device = T.device
MAXLEN = T.MAXLEN

# ---------------- 增强方法 (输入 z-score 后的 [L,6] numpy, 长度不变) ----------------
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
    t = np.arange(L)
    n_knots = 4
    knots = np.linspace(0, L - 1, n_knots)
    vals = knots + np.random.normal(0, sigma * L / n_knots, size=n_knots)
    vals = np.clip(vals, 0, L - 1); vals[0] = 0; vals[-1] = L - 1
    new_t = np.interp(t, knots, vals)
    new_t = np.clip(new_t, 0, L - 1)
    return np.stack([np.interp(new_t, t, x[:, c]) for c in range(C)], axis=1).astype(np.float32)

def aug_window_slice(x):
    L, C = x.shape
    if L < 10:
        return x
    w = np.random.randint(int(L * 0.5), L)
    start = np.random.randint(0, L - w + 1)
    seg = x[start:start + w]
    t_old = np.linspace(0, 1, w)
    t_new = np.linspace(0, 1, L)
    return np.stack([np.interp(t_new, t_old, seg[:, c]) for c in range(C)], axis=1).astype(np.float32)

AUG_METHODS = {
    'jitter': aug_jitter, 'scaling': aug_scaling,
    'magnitude_warp': aug_magnitude_warp, 'time_warp': aug_time_warp,
    'window_slice': aug_window_slice,
}
METHOD_NAMES = list(AUG_METHODS.keys())

def apply_aug(x):
    """随机选 1~2 种方法串行叠加。"""
    k = np.random.randint(1, 3)
    names = np.random.choice(METHOD_NAMES, size=k, replace=False)
    for n in names:
        x = AUG_METHODS[n](x)
    return x

# ---------------- 构建增强训练集 ----------------
raw_tr = D['train']['X_tensor']
Ftr = D['train']['X_feat'].astype(np.float32)
ytr = D['train']['y']
n_pos = int((ytr == 1).sum())
print(f'Train: {len(raw_tr)} seqs ({n_pos} positive), AUG_K={AUG_K}', flush=True)

raw_list = list(raw_tr)
feat_list = list(Ftr)
y_list = list(ytr)
for i in range(len(raw_tr)):
    if ytr[i] != 1:
        continue
    for _ in range(AUG_K):
        raw_list.append(apply_aug(raw_tr[i]))
        feat_list.append(Ftr[i])       # 增强副本沿用原特征
        y_list.append(1)

Xtr_aug, ltr_aug = T.pad(raw_list)
Ftr_aug = np.stack(feat_list, 0)
ytr_aug = np.array(y_list, dtype=np.int64)
print(f'Augmented train: {Xtr_aug.shape} feat {Ftr_aug.shape} y+={(ytr_aug==1).sum()}', flush=True)

Xva, lva = T.pad(D['val']['X_tensor']); yva = D['val']['y']
Xte, lte = T.pad(D['test']['X_tensor']); yte = D['test']['y']
Fva = D['val']['X_feat'].astype(np.float32)
Fte = D['test']['X_feat'].astype(np.float32)

# ---------------- 训练 (与 train_c1c2.run_one 同协议) ----------------
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

model = T.TokenAttnFusion(T.FEAT_DIM).to(device)
nparams = sum(p.numel() for p in model.parameters())
pos_w = float((ytr_aug == 0).sum()) / max(1, int((ytr_aug == 1).sum()))
w = torch.tensor([1.0, pos_w], dtype=torch.float32).to(device)
crit = nn.CrossEntropyLoss(weight=w)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

Xtr_t = torch.FloatTensor(Xtr_aug).to(device); ltr_t = torch.LongTensor(ltr_aug)
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
        vo = model(Xva_t, lva_t, Fva_t); vp = vo.argmax(1).cpu().numpy()
        vf1 = f1_score(yva, vp, zero_division=0)
    if vf1 > best_f1:
        best_f1 = vf1; best_ep = ep + 1
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if (ep + 1) % 5 == 0 or ep == EPOCHS - 1:
        print(f'  [tokenattn_augK{AUG_K}] ep{ep+1}/{EPOCHS} loss={tot:.3f} val_f1={vf1:.4f} best={best_f1:.4f}(ep{best_ep})', flush=True)

model.load_state_dict(best_state); model.eval()
with torch.no_grad():
    te = model(Xte_t, lte_t, Fte_t)
    prob = torch.softmax(te, 1)[:, 1].cpu().numpy(); pred = te.argmax(1).cpu().numpy()
    vprob = torch.softmax(model(Xva_t, lva_t, Fva_t), 1)[:, 1].cpu().numpy()

best_th, best_cf1 = 0.5, 0
for th in np.arange(0.05, 0.96, 0.01):
    f1 = f1_score(yva, (vprob >= th).astype(int), zero_division=0)
    if f1 > best_cf1: best_cf1, best_th = f1, float(th)
cpred = (prob >= best_th).astype(int)
res = {
    'PR-AUC': average_precision_score(yte, prob),
    'AUC': roc_auc_score(yte, prob),
    'F1': f1_score(yte, pred),
    'Recall': recall_score(yte, pred),
    'Prec': precision_score(yte, pred, zero_division=0),
    'Acc': accuracy_score(yte, pred),
    'best_epoch': best_ep, 'params': nparams, 'best_th': best_th,
    'calib_F1': f1_score(yte, cpred), 'calib_Recall': recall_score(yte, cpred),
    'calib_Prec': precision_score(yte, cpred, zero_division=0),
    'sec': round(time.time() - t0, 1),
    'AUG_K': AUG_K, 'n_train_pos_aug': int((ytr_aug == 1).sum()),
    'n_train_total_aug': len(ytr_aug),
}
out = {'tokenattn_aug': res,
       'meta': {'seed': SEED, 'AUG_K': AUG_K, 'methods': METHOD_NAMES,
                'pos_weight': pos_w, 'device': str(device), 'epochs': EPOCHS}}
sfix = f'_s{SEED}' if SEED != 42 else ''
name = f'c1c2_tokenattn_augK{AUG_K}{sfix}'
json.dump(out, open(f'{T.OUT}/{name}_results.json', 'w'), indent=2)
np.save(f'{T.OUT}/{name}_prob.npy', prob)
torch.save({'state': best_state, 'meta': {'name': name, 'seed': SEED,
            'AUG_K': AUG_K, 'best_epoch': best_ep, 'PR-AUC': res['PR-AUC'], 'device': str(device)}},
           f'{T.OUT}/{name}_model.pt')
print(f'\n=== tokenattn_aug (K={AUG_K}, seed={SEED}) === (th=0.5)', flush=True)
for k in ['Acc', 'Prec', 'Recall', 'F1', 'AUC', 'PR-AUC']:
    print(f'  {k}: {res[k]:.4f}', flush=True)
print(f'  best_th={best_th:.2f} calib_F1={res["calib_F1"]:.4f} (params={nparams:,}, {res["sec"]}s)', flush=True)
print('DONE', flush=True)
