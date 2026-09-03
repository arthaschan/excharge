#!/usr/bin/env python3
"""R3: 案例深描 —— 挑 test 异常样本, 画"曲线 + 注意力热力 + 特征关注"并排。

对 4 个 test 异常样本(2 个高置信真阳 + 2 个高温度跳变), 各出一张 2 面板图:
  左: 6 通道充电曲线(z-score, 填充到实际长度)
  右: 该样本 [CLS] 注意力对 62 特征 token 的 top-12 关注(异常时模型依赖哪些特征)
输出: docs/r3_figs/case_{i}_{tag}.png
"""
import os, sys, pickle, warnings
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
import torch
torch.backends.mha.set_fastpath_enabled(False)   # 抓注意力必须禁 fastpath
torch.set_num_threads(4)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_c1c2 as T

D = T.D
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FEAT_COLS = D['feat_cols']
SEQ_FEATS = D['seq_feats']

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs(f'{T.OUT}/r3_figs', exist_ok=True)

# ---------- 载入 seed42 模型 ----------
model = T.TokenAttnFusion(62).to(device)
ckpt = torch.load(f'{T.OUT}/c1c2_tokenattn_model.pt', map_location='cpu')
model.load_state_dict(ckpt['state']); model.eval()

# ---------- 抓注意力 (monkey-patch) ----------
attn_buf = {}
for i, layer in enumerate(model.encoder.layers):
    sa = layer.self_attn
    orig = sa.forward
    def make(idx, o):
        def w(*a, **k):
            k['need_weights'] = True; k['average_attn_weights'] = False
            out = o(*a, **k)
            attn_buf[idx] = out[1].detach().float()
            return out
        return w
    sa.forward = make(i, orig)

# ---------- 数据 ----------
Xte, lte = T.pad(D['test']['X_tensor'])
Fte = D['test']['X_feat'].astype(np.float32)
yte = np.asarray(D['test']['y'])
K = model.K

# 预测概率
with torch.no_grad():
    probs = torch.softmax(model(torch.FloatTensor(Xte).to(device),
                                torch.LongTensor(lte).to(device),
                                torch.FloatTensor(Fte).to(device)), 1)[:, 1].cpu().numpy()

# ---------- 挑样本 ----------
fault_idx = np.where(yte == 1)[0]
# 2 个最高置信真阳
conf_order = fault_idx[np.argsort(probs[fault_idx])[::-1]]
picks = list(conf_order[:2])
# 2 个最高 t1_max_jump (温度跳变型异常), 避免与上面重复
tj_col = FEAT_COLS.index('t1_max_jump')
tj_order = fault_idx[np.argsort(Fte[fault_idx, tj_col])[::-1]]
for i in tj_order:
    if i not in picks:
        picks.append(i)
    if len(picks) >= 4:
        break
print(f'Picked test fault samples: {picks}', flush=True)

def get_sample_attention(idx):
    with torch.no_grad():
        model(torch.FloatTensor(Xte[idx:idx+1]).to(device),
              torch.LongTensor(lte[idx:idx+1]).to(device),
              torch.FloatTensor(Fte[idx:idx+1]).to(device))
        A = attn_buf[len(model.encoder.layers) - 1].mean(1).cpu().numpy()[0]  # [S,S]
    n_seg = K; n_feat = A.shape[0] - 1 - n_seg
    a_cls = A[0, :]
    seg_w = a_cls[1:1 + n_seg]
    feat_w = a_cls[1 + n_seg:]
    return seg_w, feat_w

CH_LABELS = ['Voltage', 'Current', 'Power', 'GunTemp1', 'GunTemp2', 'SOC']
for idx in picks:
    seg_w, feat_w = get_sample_attention(idx)
    topf = np.argsort(feat_w)[::-1][:12]
    L = int(lte[idx])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={'width_ratios': [1.4, 1]})
    # 左: 曲线
    ax = axes[0]
    t = np.arange(L)
    for c in range(6):
        ax.plot(t, Xte[idx, :L, c], lw=1.0, label=CH_LABELS[c])
    ax.set_xlabel('time step'); ax.set_ylabel('z-scored'); ax.set_title(f'Fault sample #{idx} (pred={probs[idx]:.2f})')
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    # 右: 特征关注 top-12
    ax = axes[1]
    names = [FEAT_COLS[j] for j in topf][::-1]
    vals = feat_w[topf][::-1]
    ax.barh(range(len(names)), vals, color='#C44E52')
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('[CLS] attention to feature token')
    ax.set_title(f'top feature attention (seg attn: {seg_w.argmax()+1}/8 peak)')
    plt.tight_layout()
    plt.savefig(f'{T.OUT}/r3_figs/case_{idx}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  saved case_{idx}.png  (top features: {[FEAT_COLS[j] for j in topf[:5]]})', flush=True)

print('DONE', flush=True)
