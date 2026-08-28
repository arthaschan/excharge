#!/usr/bin/env python3
"""独立评估融合模型 (加载 fusion_model.pt, 不重新训练)。
计算测试集(owner7-8)指标 + 阈值扫描 + 与纯 BiLSTM 对比, 写 fusion_results.json。
用于训练被中断时从 checkpoint 恢复评估, 也可单独重跑评估。
"""
import pickle, numpy as np, json, os, warnings
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'
import torch, torch.nn as nn
torch.set_num_threads(4)
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score)

DATA='/Users/arthas/git/excharge/data/real/'; OUT='/Users/arthas/git/excharge/docs/'
MAXLEN=200

with open(f'{DATA}/fusion_data.pkl','rb') as f: D=pickle.load(f)
def pad(seqs):
    B=len(seqs); X=np.zeros((B,MAXLEN,6),dtype=np.float32); L=np.zeros(B,dtype=np.int64)
    for i,s in enumerate(seqs):
        n=min(len(s),MAXLEN); X[i,:n]=s[:n]; L[i]=n
    return X,L
Xte,lte=pad(D['test']['X_tensor']); yte=D['test']['y']; Fte=D['test']['X_feat'].astype(np.float32)
print(f'Test {len(Xte)} (fault {yte.sum()})', flush=True)

device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
class FusionModel(nn.Module):
    def __init__(self,n_seq=6,hidden=64,n_layers=2,feat_dim=62,dropout=0.2):
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
with torch.no_grad():
    o=model(Xte_t,lte_t,Fte_t); prob=torch.softmax(o,1)[:,1].cpu().numpy(); pred=o.argmax(1).cpu().numpy()
res={'Acc':accuracy_score(yte,pred),'Prec':precision_score(yte,pred,zero_division=0),'Recall':recall_score(yte,pred),
     'F1':f1_score(yte,pred),'AUC':roc_auc_score(yte,prob),'PR-AUC':average_precision_score(yte,prob)}
print('\n=== Fusion Model on Test (owners7-8), th=0.5 ===')
for k,v in res.items(): print(f'  {k}: {v:.4f}')
sweep={}
print('\n=== Threshold sweep ===')
for th in [0.3,0.4,0.5,0.6,0.7,0.8]:
    p=(prob>=th).astype(int); r=recall_score(yte,p,zero_division=0); pr=precision_score(yte,p,zero_division=0); f1=f1_score(yte,p,zero_division=0)
    sweep[th]={'Recall':float(r),'Prec':float(pr),'F1':float(f1)}
    print(f'  th={th}: Recall={r:.4f} Prec={pr:.4f} F1={f1:.4f}')
try:
    base=json.load(open(f'{OUT}/routeC_bilstm_results.json'))
    print('\n=== Compare: Fusion vs Pure Bi-LSTM (routeC) ===')
    for k in ['Recall','Prec','F1','AUC','PR-AUC']:
        print(f'  {k}: fusion={res[k]:.4f}  pure={base[k]:.4f}  Δ={res[k]-base[k]:+.4f}')
except Exception as e:
    base=None; print('baseline compare skipped:', e)
best=json.load(open(f'{OUT}/fusion_best.json')) if os.path.exists(f'{OUT}/fusion_best.json') else {}
out={'fusion':res,'sweep':sweep,'baseline_pure_bilstm':base,'meta':{'feat_dim':D['meta']['n_features'],'device':str(device),**best}}
json.dump(out,open(f'{OUT}/fusion_results.json','w'),indent=2)
np.save(f'{OUT}/fusion_prob.npy',prob); np.save(f'{OUT}/fusion_pred.npy',pred)
print('\nSaved fusion_results.json, fusion_prob.npy, fusion_pred.npy')
