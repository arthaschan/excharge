#!/usr/bin/env python3
"""绘制 Bi-LSTM 1D-GradCAM 可视化图（出版级，英文标注）。
图1: 特征重要性条形图(故障样本)
图2: 时间重要性曲线(故障 vs 正常, 对齐到序列末端)
图3: 单样本故障 GradCAM 热力图示例 (时间×特征)
"""
import pickle, numpy as np, json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA = '/Users/arthas/git/excharge/data/real/'
OUT = '/Users/arthas/git/excharge/docs/'

with open(f'{DATA}/seq_tensors.pkl', 'rb') as f:
    d = pickle.load(f)
X_te, y_te = d['X_te'], d['y_te']
FEATS = d['feats']

r = json.load(open(f'{OUT}/routeC_gradcam_results.json'))
feat_imp = np.array(r['feat_imp'])
time_fault = np.array(r['time_fault'])
time_normal = np.array(r['time_normal'])
feats_en = r['feats_en']

# 样式
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
                     'axes.spines.top': False, 'axes.spines.right': False})

# ---- 图1: 特征重要性 ----
fig, ax = plt.subplots(figsize=(6.5, 4))
order = np.argsort(feat_imp)
colors = ['#e74c3c' if i == order[-1] else '#5b8ff9' for i in range(len(FEATS))]
ax.barh([feats_en[i] for i in order], feat_imp[order], color=colors, edgecolor='none')
ax.set_xlabel('Feature importance  |∂logit/∂x|  (summed over fault samples)')
ax.set_title('1D-GradCAM Feature Attribution (Bi-LSTM, fault n=129)')
for i, j in enumerate(order):
    ax.text(feat_imp[j] + 0.03, i, f'{feat_imp[j]:.2f}', va='center', fontsize=10)
ax.set_xlim(0, feat_imp.max() * 1.25)
fig.tight_layout()
fig.savefig(f'{OUT}/routeC_gradcam_feat_importance.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# ---- 图2: 时间重要性(对齐末端) ----
def align_tail(imp):
    """把重要性按相对位置(距末端)重采样到 0-100"""
    # 找到非零段(实际序列长度内的有效段)
    return imp

# 直接画原始时间轴 + 加末端对齐子图
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# 左: 原始时间步(0-200)
ax = axes[0]
ax.plot(time_fault[:160], label='Fault (n=129)', color='#e74c3c', lw=1.8)
ax.plot(time_normal[:160], label='Normal (n=2647)', color='#5b8ff9', lw=1.8)
ax.set_xlabel('Time step (from charging start)')
ax.set_ylabel('GradCAM importance')
ax.set_title('Temporal importance (absolute)')
ax.legend(frameon=False)

# 右: 对齐末端(最后40步)
ax = axes[1]
tail = 40
ax.plot(np.arange(tail), time_fault[-tail:], label='Fault', color='#e74c3c', lw=1.8)
ax.plot(np.arange(tail), time_normal[-tail:], label='Normal', color='#5b8ff9', lw=1.8)
ax.set_xlabel('Time step (aligned to charging end)')
ax.set_ylabel('GradCAM importance')
ax.set_title('Temporal importance (tail-aligned)')
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(f'{OUT}/routeC_gradcam_time_importance.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# ---- 图3: 单样本热力图示例 ----
# 取一个典型故障样本, 计算逐时间步×特征的梯度贡献
import torch, torch.nn as nn
torch.set_num_threads(4)
MAXLEN = 200

class BiLSTM(nn.Module):
    def __init__(self, n_feat=6, hidden=64, n_layers=2, dropout=0.2):
        super().__init__()
        self.hidden = hidden
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=n_layers, batch_first=True,
                            bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden*2), nn.Linear(hidden*2, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 2))
    def forward(self, x, L):
        B, T, F = x.shape
        packed = nn.utils.rnn.pack_padded_sequence(x, L.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=T)
        fwd_last = out[torch.arange(B), L-1, :self.hidden]
        bwd_last = out[:, 0, self.hidden:]
        last = torch.cat([fwd_last, bwd_last], dim=1)
        mask = torch.arange(T, device=x.device).unsqueeze(0) < L.unsqueeze(1).to(x.device)
        out_masked = out.masked_fill(~mask.unsqueeze(-1), -1e9)
        maxp = out_masked.max(dim=1).values
        return self.head(last + maxp)

model = BiLSTM()
model.load_state_dict(torch.load(f'{OUT}/routeC_bilstm_model.pt', map_location='cpu'))
model.eval()

def pad_one(s):
    X = np.zeros((1, MAXLEN, len(FEATS)), dtype=np.float32)
    L = min(len(s), MAXLEN)
    X[0, :L] = s[:L]
    return X, L

fault_idx = np.where(y_te == 1)[0]
# 选最长的一条故障样本作示例
idx = fault_idx[int(np.argmax([len(X_te[i]) for i in fault_idx]))]
s = X_te[idx]
X_p, L = pad_one(s)
x = torch.FloatTensor(X_p).requires_grad_(True)
l = torch.LongTensor([L])
logit = model(x, l)[0, 1]
model.zero_grad(); logit.backward()
gx = x.grad[0].detach().numpy()[:L]  # (L, F)

fig, axes = plt.subplots(2, 1, figsize=(9, 5), gridspec_kw={'height_ratios': [1, 2.2]})
# 上: 原始信号
ax = axes[0]
sig_colors = ['#2c3e50', '#e67e22', '#27ae60', '#8e44ad', '#c0392b', '#16a085']
for j in range(6):
    ax.plot(s[:L, j], color=sig_colors[j], lw=1.0, label=feats_en[j])
ax.set_ylabel('Normalized signal')
ax.set_title(f'Example fault sequence (sample #{idx}, L={L}) + input-gradient attribution')
ax.legend(frameon=False, ncol=3, fontsize=8, loc='upper right')

# 下: 输入梯度绝对值热力图 (时间×特征)
ax = axes[1]
gx_abs = np.abs(gx)
gx_norm = gx_abs / (gx_abs.max() + 1e-9)
im = ax.imshow(gx_norm.T, aspect='auto', cmap='Reds', interpolation='nearest')
ax.set_xlabel('Time step')
ax.set_yticks(range(6))
ax.set_yticklabels(feats_en)
ax.set_title('|∂logit/∂x| attribution heatmap (1D-GradCAM, input gradient)')
fig.colorbar(im, ax=ax, label='Normalized |gradient|')
fig.tight_layout()
fig.savefig(f'{OUT}/routeC_gradcam_heatmap.png', dpi=200, bbox_inches='tight')
plt.close(fig)

print('3 figures saved.')
