#!/usr/bin/env python3
"""融合模型输入通道敏感性分析: 逐通道置零(6 通道原始时序), 观察测试集性能变化。
与 1D-GradCAM 通道归因交叉印证。输出: docs/fusion_sensitivity.json + fusion_sensitivity.png
"""
import pickle, numpy as np, json, time, os, warnings
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'
import torch, torch.nn as nn
torch.set_num_threads(4)
from sklearn.metrics import recall_score, roc_auc_score

DATA = '/Users/arthas/git/excharge/data/real/'
OUT  = '/Users/arthas/git/excharge/docs/'
MAXLEN = 200
SEQ_FEATS = ['chargingv','charginga','out_power','charging_gun_temperature1','charging_gun_temperature2','current_soc']
FEATS_EN = ['Voltage(V)','Current(A)','Power(kW)','GunTemp1(°C)','GunTemp2(°C)','SOC(%)']

with open(f'{DATA}/fusion_data.pkl','rb') as f: D = pickle.load(f)
def pad(seqs):
    B=len(seqs); X=np.zeros((B,MAXLEN,6),dtype=np.float32); L=np.zeros(B,dtype=np.int64)
    for i,s in enumerate(seqs):
        n=min(len(s),MAXLEN); X[i,:n]=s[:n]; L[i]=n
    return X,L
Xte,lte = pad(D['test']['X_tensor']); yte=D['test']['y']; Fte=D['test']['X_feat'].astype(np.float32)
print(f'Test {len(Xte)} (fault {yte.sum()})', flush=True)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
class FusionModel(nn.Module):
    def __init__(self, n_seq=6, hidden=64, n_layers=2, feat_dim=62, dropout=0.2):
        super().__init__(); self.hidden=hidden
        self.lstm=nn.LSTM(n_seq,hidden,num_layers=n_layers,batch_first=True,bidirectional=True,dropout=dropout)
        self.head=nn.Sequential(nn.LayerNorm(hidden*2+feat_dim),nn.Linear(hidden*2+feat_dim,64),
                                nn.ReLU(),nn.Dropout(dropout),nn.Linear(64,2))
    def seq_repr(self,x,L):
        B,T,_=x.shape
        packed=nn.utils.rnn.pack_padded_sequence(x,L.cpu(),batch_first=True,enforce_sorted=False)
        out,_=self.lstm(packed); out,_=nn.utils.rnn.pad_packed_sequence(out,batch_first=True,total_length=T)
        fwd=out[torch.arange(B),L-1,:self.hidden]; bwd=out[:,0,self.hidden:]
        last=torch.cat([fwd,bwd],dim=1)
        mask=torch.arange(T,device=x.device).unsqueeze(0)<L.unsqueeze(1).to(x.device)
        maxp=out.masked_fill(~mask.unsqueeze(-1),-1e9).max(dim=1).values
        return last+maxp
    def forward(self,x,L,f): return self.head(torch.cat([self.seq_repr(x,L),f],dim=1))
model=FusionModel(feat_dim=D['meta']['n_features'])
model.load_state_dict(torch.load(f'{OUT}/fusion_model.pt',map_location='cpu')); model.to(device).eval()

Xte_t=torch.FloatTensor(Xte).to(device); lte_t=torch.LongTensor(lte).to(device); Fte_t=torch.FloatTensor(Fte).to(device)
def eval_(X):
    with torch.no_grad():
        o=model(X,lte_t,Fte_t); prob=torch.softmax(o,1)[:,1].cpu().numpy(); pred=o.argmax(1).cpu().numpy()
    return recall_score(yte,pred), roc_auc_score(yte,prob), prob
baseR,baseA,_=eval_(Xte_t)
print(f'Baseline Recall={baseR:.4f} AUC={baseA:.4f}', flush=True)
rows=[]
for j in range(6):
    Xz=Xte_t.clone(); Xz[:,:,j]=0.0
    r,a,_=eval_(Xz)
    rows.append({'channel':FEATS_EN[j],'zeroed_Recall':float(r),'dRecall':float(r-baseR),
                 'zeroed_AUC':float(a),'dAUC':float(a-baseA)})
    print(f'  zero {FEATS_EN[j]:12s}: Recall {r:.4f} (Δ{r-baseR:+.4f}) | AUC {a:.4f} (Δ{a-baseA:+.4f})', flush=True)
# 分组
g_elec=[0,1,2]; g_temp=[3,4]
def grp(idxs):
    Xz=Xte_t.clone()
    for j in idxs: Xz[:,:,j]=0.0
    r,a,_=eval_(Xz); return r,a
re,ae=grp(g_elec); rt,at=grp(g_temp)
print(f'  [Electrical group zeroed] Recall {re:.4f} (Δ{re-baseR:+.4f}) AUC {ae:.4f} (Δ{ae-baseA:+.4f})', flush=True)
print(f'  [Temperature group zeroed] Recall {rt:.4f} (Δ{rt-baseR:+.4f}) AUC {at:.4f} (Δ{at-baseA:+.4f})', flush=True)

# 图
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(11,4))
names=[r['channel'] for r in rows]
dR=[r['dRecall'] for r in rows]; dA=[r['dAUC'] for r in rows]
colors=['#e74c3c' if v<0 else '#5b8ff9' for v in dR]
axes[0].bar(names,dR,color=colors); axes[0].axhline(0,color='k',lw=0.8)
axes[0].set_ylabel('Δ Recall'); axes[0].set_title('Channel-zeroed Δ Recall (fusion)'); axes[0].tick_params(axis='x',rotation=30)
colors2=['#e74c3c' if v<0 else '#5b8ff9' for v in dA]
axes[1].bar(names,dA,color=colors2); axes[1].axhline(0,color='k',lw=0.8)
axes[1].set_ylabel('Δ AUC'); axes[1].set_title('Channel-zeroed Δ AUC (fusion)'); axes[1].tick_params(axis='x',rotation=30)
plt.tight_layout(); fig.savefig(f'{OUT}/fusion_sensitivity.png',dpi=200,bbox_inches='tight'); plt.close(fig)

out={'baseline_Recall':float(baseR),'baseline_AUC':float(baseA),
     'per_channel':rows,'electrical_group':{'Recall':float(re),'dRecall':float(re-baseR),'AUC':float(ae),'dAUC':float(ae-baseA)},
     'temperature_group':{'Recall':float(rt),'dRecall':float(rt-baseR),'AUC':float(at),'dAUC':float(at-baseA)}}
json.dump(out,open(f'{OUT}/fusion_sensitivity.json','w'),indent=2)
print('Saved fusion_sensitivity.json + fusion_sensitivity.png', flush=True)
