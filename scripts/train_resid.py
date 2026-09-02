#!/usr/bin/env python3
"""P1b: 分解残差化预处理 → 序列分支 (复刻 U-Net-DeW 逻辑)。

思路(来自导师给的时序增强综述 Wen et al.): 先去掉确定性/趋势分量, 再对残差建模。
充电曲线有强确定性形状(SOC 决定功率曲线), 异常是叠加其上的局部突变(超温/SOC跳变/电压塌陷)。
对 6 通道序列做去趋势(减去居中滑动均值), 得到残差 = 局部偏离, 喂给序列分支,
帮助模型聚焦"局部异常"而非"正常充电形状"。62 维特征不变。

用法:
  DEVICE=cuda BATCH=64 SEED=42 WINDOW=20 python train_resid.py
输出: docs/c1c2_tokenattn_resid_w{WINDOW}_s{seed}_results.json / _prob.npy / _model.pt
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
WINDOW = int(os.environ.get('WINDOW', 20))   # 滑动均值窗口(去趋势尺度)
EPOCHS = int(os.environ.get('EPOCHS', 30))
BATCH = int(os.environ.get('BATCH', 64))
torch.manual_seed(SEED); np.random.seed(SEED)

D = T.D
device = T.device


def detrend(x, window):
    """x: [L,6] z-scored 序列; 返回去趋势残差 (x - 居中滑动均值)。"""
    L = x.shape[0]
    if L < 3:
        return x - x.mean(0, keepdims=True)
    w = min(window, L)
    kernel = np.ones(w, dtype=np.float64) / w
    trend = np.stack([np.convolve(x[:, c], kernel, mode='same') for c in range(x.shape[1])], axis=1)
    return x - trend.astype(np.float32)


print(f'Residualizing (detrend window={WINDOW}) ...', flush=True)
raw_tr = [detrend(s, WINDOW) for s in D['train']['X_tensor']]
raw_va = [detrend(s, WINDOW) for s in D['val']['X_tensor']]
raw_te = [detrend(s, WINDOW) for s in D['test']['X_tensor']]

Xtr, ltr = T.pad(raw_tr); ytr = D['train']['y']
Xva, lva = T.pad(raw_va); yva = D['val']['y']
Xte, lte = T.pad(raw_te); yte = D['test']['y']
Ftr = D['train']['X_feat'].astype(np.float32)
Fva = D['val']['X_feat'].astype(np.float32)
Fte = D['test']['X_feat'].astype(np.float32)
print(f'Train {Xtr.shape} feat {Ftr.shape} | Val {Xva.shape} | Test {Xte.shape}', flush=True)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

model = T.TokenAttnFusion(T.FEAT_DIM).to(device)
nparams = sum(p.numel() for p in model.parameters())
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
        vo = model(Xva_t, lva_t, Fva_t); vp = vo.argmax(1).cpu().numpy()
        vf1 = f1_score(yva, vp, zero_division=0)
    if vf1 > best_f1:
        best_f1 = vf1; best_ep = ep + 1
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if (ep + 1) % 5 == 0 or ep == EPOCHS - 1:
        print(f'  [tokenattn_resid_w{WINDOW}] ep{ep+1}/{EPOCHS} loss={tot:.3f} val_f1={vf1:.4f} best={best_f1:.4f}(ep{best_ep})', flush=True)

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
    'F1': f1_score(yte, pred), 'Recall': recall_score(yte, pred),
    'Prec': precision_score(yte, pred, zero_division=0), 'Acc': accuracy_score(yte, pred),
    'best_epoch': best_ep, 'params': nparams, 'best_th': best_th,
    'calib_F1': f1_score(yte, cpred), 'calib_Recall': recall_score(yte, cpred),
    'calib_Prec': precision_score(yte, cpred, zero_division=0),
    'sec': round(time.time() - t0, 1), 'WINDOW': WINDOW,
}
out = {'tokenattn_resid': res,
       'meta': {'seed': SEED, 'WINDOW': WINDOW, 'pos_weight': pos_w,
                'device': str(device), 'epochs': EPOCHS}}
sfix = f'_s{SEED}' if SEED != 42 else ''
name = f'c1c2_tokenattn_resid_w{WINDOW}{sfix}'
json.dump(out, open(f'{T.OUT}/{name}_results.json', 'w'), indent=2)
np.save(f'{T.OUT}/{name}_prob.npy', prob)
torch.save({'state': best_state, 'meta': {'name': name, 'seed': SEED, 'WINDOW': WINDOW,
            'best_epoch': best_ep, 'PR-AUC': res['PR-AUC'], 'device': str(device)}},
           f'{T.OUT}/{name}_model.pt')
print(f'\n=== tokenattn_resid (WINDOW={WINDOW}, seed={SEED}) === (th=0.5)', flush=True)
for k in ['Acc', 'Prec', 'Recall', 'F1', 'AUC', 'PR-AUC']:
    print(f'  {k}: {res[k]:.4f}', flush=True)
print(f'  best_th={best_th:.2f} calib_F1={res["calib_F1"]:.4f} (params={nparams:,}, {res["sec"]}s)', flush=True)
print('DONE', flush=True)
