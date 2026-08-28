#!/usr/bin/env python3
"""敏感性分析（批注2落地）：
1. 输入通道敏感性: 6 个传感器通道逐一置零, 看 AUC/Recall/F1 变化
   (通道越重要, 置零后性能下降越多)
2. 序列长度阈值敏感性: 30/50/80/120 点
3. 输出结果 JSON 供论文引用
模型: 需传入已训练的 Bi-LSTM 模型路径
"""
import pickle, numpy as np, time, os, warnings, json, sys
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'
import torch, torch.nn as nn
torch.set_num_threads(4)

DATA = '/Users/arthas/git/excharge/data/real/'
OUT = '/Users/arthas/git/excharge/docs/'
MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else f'{OUT}/routeC_bilstm_model.pt'
TAG = sys.argv[2] if len(sys.argv) > 2 else 'bilstm'

with open(f'{DATA}/seq_tensors.pkl', 'rb') as f:
    d = pickle.load(f)
X_te, y_te = d['X_te'], d['y_te']
FEATS = d['feats']
FEATS_EN = ['Voltage(V)', 'Current(A)', 'Power(kW)', 'GunTemp1(°C)', 'GunTemp2(°C)', 'SOC(%)']

MAXLEN = 200
def pad(seqs):
    B = len(seqs); F = len(FEATS)
    X = np.zeros((B, MAXLEN, F), dtype=np.float32)
    L_arr = np.zeros(B, dtype=np.int64)
    for i, s in enumerate(seqs):
        L = min(len(s), MAXLEN)
        X[i, :L] = s[:L]; L_arr[i] = L
    return X, L_arr

X_te_p, l_te = pad(X_te)
print(f'Test seqs: {len(X_te)} (fault {y_te.sum()})', flush=True)

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
        out, (hn, cn) = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=T)
        fwd_last = out[torch.arange(B), L-1, :self.hidden]
        bwd_last = out[:, 0, self.hidden:]
        last = torch.cat([fwd_last, bwd_last], dim=1)
        mask = torch.arange(T, device=x.device).unsqueeze(0) < L.unsqueeze(1).to(x.device)
        out_masked = out.masked_fill(~mask.unsqueeze(-1), -1e9)
        maxp = out_masked.max(dim=1).values
        feat = last + maxp
        return self.head(feat)

model = BiLSTM()
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()
print(f'Model loaded from {MODEL_PATH}', flush=True)

from sklearn.metrics import recall_score, f1_score, roc_auc_score, average_precision_score

def evaluate(X_p, l_arr):
    with torch.no_grad():
        X_t = torch.FloatTensor(X_p)
        L_t = torch.LongTensor(l_arr)
        out = model(X_t, L_t)
        prob = torch.softmax(out, 1)[:, 1].cpu().numpy()
        pred = out.argmax(1).cpu().numpy()
    return {
        'Recall': recall_score(y_te, pred, zero_division=0),
        'F1': f1_score(y_te, pred, zero_division=0),
        'AUC': roc_auc_score(y_te, prob),
        'PR-AUC': average_precision_score(y_te, prob),
    }

# 基线
base = evaluate(X_te_p, l_te)
print('\n=== 基线(全6通道) ===', flush=True)
for k, v in base.items(): print(f'  {k}: {v:.4f}', flush=True)

# 1. 输入通道敏感性: 逐通道置零
print('\n=== 输入通道敏感性分析(逐通道置零) ===', flush=True)
channel_sens = {}
for j in range(len(FEATS)):
    X_zero = X_te_p.copy()
    X_zero[:, :, j] = 0.0  # z-score 后置零 = 均值
    r = evaluate(X_zero, l_te)
    channel_sens[FEATS[j]] = {
        'en': FEATS_EN[j],
        'AUC': r['AUC'], 'AUC_drop': base['AUC'] - r['AUC'],
        'Recall': r['Recall'], 'Recall_drop': base['Recall'] - r['Recall'],
        'F1': r['F1'], 'PR-AUC': r['PR-AUC'],
    }
    print(f"  {FEATS_EN[j]:16s}: AUC {r['AUC']:.4f} (Δ {base['AUC']-r['AUC']:+.4f})  Recall {r['Recall']:.4f} (Δ {base['Recall']-r['Recall']:+.4f})", flush=True)

# 2. 特征分组敏感性: 电气(V/I/P) vs 温度(枪温1/2) vs SOC
print('\n=== 特征分组敏感性 ===', flush=True)
groups = {
    'electrical_VIP': [0, 1, 2],
    'temperature': [3, 4],
    'soc': [5],
}
group_sens = {}
for gname, cols in groups.items():
    X_zero = X_te_p.copy()
    for j in cols:
        X_zero[:, :, j] = 0.0
    r = evaluate(X_zero, l_te)
    group_sens[gname] = {'AUC': r['AUC'], 'AUC_drop': base['AUC'] - r['AUC'], 'Recall': r['Recall'], 'Recall_drop': base['Recall'] - r['Recall']}
    print(f"  {gname:16s}: AUC {r['AUC']:.4f} (Δ {base['AUC']-r['AUC']:+.4f})  Recall {r['Recall']:.4f} (Δ {base['Recall']-r['Recall']:+.4f})", flush=True)

# 保存
result = {'base': base, 'channel_sensitivity': channel_sens, 'group_sensitivity': group_sens,
          'model': MODEL_PATH, 'feats': FEATS, 'feats_en': FEATS_EN}
json.dump(result, open(f'{OUT}/routeC_{TAG}_sensitivity.json', 'w'), indent=2, ensure_ascii=False)
print(f'\nSaved routeC_{TAG}_sensitivity.json', flush=True)
