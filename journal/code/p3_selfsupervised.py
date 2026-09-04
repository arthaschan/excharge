#!/usr/bin/env python3
"""P3/方案B — 特征级自监督预训练（老师方向最后一条路）

用全量无标签序列（含 owner7-8 测试域）做掩码特征重建预训练，学一个更好的特征表征，
再微调到异常检测，看是否提升跨站 PR-AUC。

预训练任务（VIME 风格掩码重建）：
  - 输入 62 维特征（已 z-score，掩码 = 置 0 即均值）
  - Encoder MLP(62→128→64) + Decoder MLP(64→128→62)
  - 掩码 ~30% 特征，MSE 重建被掩码位置
  - 变体 B0 随机掩码；变体 B1 归因引导掩码（LightGBM top-10 重要特征以更高概率被掩码，
    强制编码器"从次要特征推断重要特征"，注入归因知识）

微调：Encoder + 分类头(64→2)，owner1-6 训 / owner7-8 测。
对照：随机初始化 encoder（无预训练）vs B0 vs B1，各单 seed 42。

判据：预训练版 PR-AUC > 随机初始化版，且 B1（归因引导）≥ B0（随机掩码）。
输出：journal/docs/p3_selfsupervised.json
"""
import pickle, numpy as np, time, os, json, warnings
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'
import torch, torch.nn as nn
import torch.nn.functional as F
torch.set_num_threads(4)
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE)
DATA = os.path.join(ROOT, 'data', 'real')
OUT = os.path.join(BASE, 'docs')
os.makedirs(OUT, exist_ok=True)

SEED = int(os.environ.get('SEED', 42))
DEVICE = os.environ.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS_FT = int(os.environ.get('EPOCHS', 30))
PRETRAIN_EPOCHS = int(os.environ.get('PT_EPOCHS', 30))
BATCH = 256
MASK_RATIO = 0.30
TOPK_IMPORTANT = 10

torch.manual_seed(SEED); np.random.seed(SEED)
device = torch.device(DEVICE)

# ---------- 1. 数据 ----------
D = pickle.load(open(f'{DATA}/fusion_data.pkl', 'rb'))
cols = D['feat_cols']; FEAT_DIM = len(cols)
Ftr = D['train']['X_feat'].astype(np.float32); ytr = D['train']['y']
Fva = D['val']['X_feat'].astype(np.float32);   yva = D['val']['y']
Fte = D['test']['X_feat'].astype(np.float32);  yte = D['test']['y']
# 全量无标签（自监督预训练用，含测试域）
Fall = np.vstack([Ftr, Fva, Fte]).astype(np.float32)
print(f'[1/4] 特征: train {Ftr.shape} val {Fva.shape} test {Fte.shape} | 无标签全量 {Fall.shape}', flush=True)

# LightGBM gain（归因引导掩码用）
m_lgb = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31,
                           subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=4, verbosity=-1)
m_lgb.fit(Ftr, ytr, eval_set=[(Fva, yva)], eval_metric='auc', callbacks=[lgb.early_stopping(100, verbose=False)])
gain = m_lgb.booster_.feature_importance(importance_type='gain')
topk_idx = set(np.argsort(-gain)[:TOPK_IMPORTANT].tolist())
print(f'  归因 top-{TOPK_IMPORTANT}: {[cols[i] for i in np.argsort(-gain)[:TOPK_IMPORTANT]]}', flush=True)

# ---------- 2. 模型 ----------
class Encoder(nn.Module):
    def __init__(self, in_dim=62, hid=128, out_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, hid), nn.BatchNorm1d(hid), nn.ReLU(),
                                 nn.Linear(hid, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU())
    def forward(self, x):
        return self.mlp(x)

class Decoder(nn.Module):
    def __init__(self, in_dim=64, hid=128, out_dim=62):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Linear(hid, out_dim))
    def forward(self, z):
        return self.mlp(z)

# ---------- 3. 预训练（掩码重建）----------
def pretrain(guided):
    torch.manual_seed(SEED); np.random.seed(SEED)
    enc = Encoder(FEAT_DIM).to(device); dec = Decoder().to(device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=1e-3, weight_decay=1e-4)
    X = torch.FloatTensor(Fall).to(device)
    n = len(X)
    t0 = time.time()
    for ep in range(PRETRAIN_EPOCHS):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            if len(idx) < 2:
                continue
            x = X[idx]
            # 生成掩码
            mask = torch.zeros_like(x)
            for j in range(FEAT_DIM):
                p = 0.5 if (guided and j in topk_idx) else 0.15 if guided else MASK_RATIO
                mask[:, j] = torch.rand(x.shape[0]) < p
            xm = x * (1 - mask)                      # 掩码位置置 0（均值）
            z = enc(xm)
            rec = dec(z)
            loss = F.mse_loss(rec * mask, x * mask)  # 只对被掩码位置算 MSE
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
    print(f'  预训练{"(归因引导)" if guided else "(随机)"}完成: {PRETRAIN_EPOCHS}ep loss末={tot/(n//BATCH+1):.4f} {time.time()-t0:.0f}s', flush=True)
    return {k: v.cpu().clone() for k, v in enc.state_dict().items()}

print('[2/4] 预训练（掩码特征重建）...', flush=True)
enc_b0 = pretrain(guided=False)
enc_b1 = pretrain(guided=True)

# ---------- 4. 微调 + 评估 ----------
def finetune(enc_state):
    torch.manual_seed(SEED); np.random.seed(SEED)
    enc = Encoder(FEAT_DIM).to(device)
    if enc_state is not None:
        enc.load_state_dict(enc_state)
    head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2)).to(device)
    pos_w = float((ytr == 0).sum()) / max(1, int((ytr == 1).sum()))
    crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_w], device=device))
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_FT)
    Ftr_t = torch.FloatTensor(Ftr).to(device); Fva_t = torch.FloatTensor(Fva).to(device); Fte_t = torch.FloatTensor(Fte).to(device)
    ytr_t = torch.LongTensor(ytr).to(device); yva_t = torch.LongTensor(yva).to(device)
    best_f1, best_ep, best_state = 0, 0, None
    for ep in range(EPOCHS_FT):
        enc.train(); head.train(); perm = torch.randperm(len(Ftr_t))
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            if len(idx) < 2:
                continue
            loss = crit(head(enc(Ftr_t[idx])), ytr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step(); enc.eval(); head.eval()
        with torch.no_grad():
            vp = head(enc(Fva_t)).argmax(1).cpu().numpy()
            vf1 = float(f1_score(yva, vp, zero_division=0))
        if vf1 > best_f1:
            best_f1, best_ep = vf1, ep + 1
            best_state = {k: v.cpu().clone() for k, v in list(enc.state_dict().items()) + list(head.state_dict().items())}
    enc.load_state_dict({k: v for k, v in best_state.items() if k in enc.state_dict()})
    head.load_state_dict({k: v for k, v in best_state.items() if k in head.state_dict()})
    enc.eval(); head.eval()
    with torch.no_grad():
        prob = torch.softmax(head(enc(Fte_t)), 1)[:, 1].cpu().numpy()
    return {'PR-AUC': float(average_precision_score(yte, prob)), 'AUC': float(roc_auc_score(yte, prob)), 'best_ep': best_ep}

print('[3/4] 微调对照（随机初始化 vs B0 vs B1）...', flush=True)
res = {}
res['random_init'] = finetune(None)
res['B0_random_mask'] = finetune(enc_b0)
res['B1_guided_mask'] = finetune(enc_b1)

# ---------- 5. 汇总 ----------
summary = {'meta': {'date': '2026-09-04', 'env': str(device), 'seed': SEED,
                    'pretrain': f'掩码重建 {PRETRAIN_EPOCHS}ep, 全量 {len(Fall)} 无标签(含owner7-8)',
                    'mask_ratio': MASK_RATIO, 'topk_guided': TOPK_IMPORTANT,
                    'protocol': 'owner1-6 微调 / owner7-8 测, 特征级MLP(62→128→64)+head'},
           'results': res}
with open(f'{OUT}/p3_selfsupervised.json', 'w', encoding='utf-8') as fp:
    json.dump(summary, fp, ensure_ascii=False, indent=2)

print('\n=== 方案B 判据（随机初始化 vs 预训练）===', flush=True)
base = res['random_init']['PR-AUC']
for k, v in res.items():
    print(f'  {k:18s}: PR-AUC={v["PR-AUC"]:.4f} (Δ{v["PR-AUC"]-base:+.4f}) AUC={v["AUC"]:.4f}', flush=True)
print('  结果已保存 journal/docs/p3_selfsupervised.json', flush=True)
