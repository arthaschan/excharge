#!/usr/bin/env python3
"""绘制融合模型 1D-GradCAM 三张图(出版级英文标注):
  图A: 通道特征重要性条形图
  图B: 时间重要性(绝对 + 末端对齐)
  图C: 单样本故障序列 + 输入梯度热力图
读取 docs/fusion_gradcam_results.json + 加载 fusion_model.pt。
"""
import pickle, numpy as np, json, os, warnings
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'
import torch, torch.nn as nn
torch.set_num_threads(4)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

DATA='/Users/arthas/git/excharge/data/real/'; OUT='/Users/arthas/git/excharge/docs/'
MAXLEN=200
SEQ_FEATS=['chargingv','charginga','out_power','charging_gun_temperature1','charging_gun_temperature2','current_soc']
FEATS_EN=['Voltage(V)','Current(A)','Power(kW)','GunTemp1(°C)','GunTemp2(°C)','SOC(%)']

r=json.load(open(f'{OUT}/fusion_gradcam_results.json'))
feat_imp=np.array(r['feat_imp']); time_fault=np.array(r['time_fault']); time_normal=np.array(r['time_normal'])
n_fault=r['n_fault']; n_normal=r['n_normal']

plt.rcParams.update({'font.size':11,'axes.labelsize':12,'axes.titlesize':13,
                     'axes.spines.top':False,'axes.spines.right':False})

# 图A: 特征重要性
fig,ax=plt.subplots(figsize=(6.5,4))
order=np.argsort(feat_imp)
colors=['#e74c3c' if i==order[-1] else '#5b8ff9' for i in range(len(SEQ_FEATS))]
ax.barh([FEATS_EN[i] for i in order],feat_imp[order],color=colors,edgecolor='none')
ax.set_xlabel('Feature importance  |∂logit/∂x|  (summed over fault samples)')
ax.set_title('1D-GradCAM Feature Attribution (fusion, fault n=%d)'%n_fault)
for i,j in enumerate(order): ax.text(feat_imp[j]+0.03,i,f'{feat_imp[j]:.2f}',va='center',fontsize=10)
ax.set_xlim(0,feat_imp.max()*1.25); fig.tight_layout()
fig.savefig(f'{OUT}/fusion_gradcam_feat_importance.png',dpi=200,bbox_inches='tight'); plt.close(fig)

# 图B: 时间重要性
fig,axes=plt.subplots(1,2,figsize=(11,4))
ax=axes[0]; ax.plot(time_fault[:160],label='Fault (n=%d)'%n_fault,color='#e74c3c',lw=1.8)
ax.plot(time_normal[:160],label='Normal (n=%d)'%n_normal,color='#5b8ff9',lw=1.8)
ax.set_xlabel('Time step (from charging start)'); ax.set_ylabel('GradCAM importance')
ax.set_title('Temporal importance (absolute)'); ax.legend(frameon=False)
ax=axes[1]; tail=40; ax.plot(np.arange(tail),time_fault[-tail:],label='Fault',color='#e74c3c',lw=1.8)
ax.plot(np.arange(tail),time_normal[-tail:],label='Normal',color='#5b8ff9',lw=1.8)
ax.set_xlabel('Time step (aligned to charging end)'); ax.set_ylabel('GradCAM importance')
ax.set_title('Temporal importance (tail-aligned)'); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(f'{OUT}/fusion_gradcam_time_importance.png',dpi=200,bbox_inches='tight'); plt.close(fig)

# 图C: 单样本热力图
class FusionModel(nn.Module):
    def __init__(self,n_seq=6,hidden=64,n_layers=2,feat_dim=62,dropout=0.2):
        super().__init__(); self.hidden=hidden
        self.lstm=nn.LSTM(n_seq,hidden,num_layers=n_layers,batch_first=True,bidirectional=True,dropout=dropout)
        self.head=nn.Sequential(nn.LayerNorm(hidden*2+feat_dim),nn.Linear(hidden*2+feat_dim,64),
                                nn.ReLU(),nn.Dropout(dropout),nn.Linear(64,2))
    def lstm_out(self,x,L):
        B,T,_=x.shape
        packed=nn.utils.rnn.pack_padded_sequence(x,L.cpu(),batch_first=True,enforce_sorted=False)
        out,_=self.lstm(packed); out,_=nn.utils.rnn.pad_packed_sequence(out,batch_first=True,total_length=T)
        return out
    def forward(self,x,L,f): return self.head(torch.cat([self.seq_repr(x,L),f],dim=1))
    def seq_repr(self,x,L):
        B,T,_=x.shape; A=self.lstm_out(x,L)
        fwd=A[torch.arange(B),L-1,:self.hidden]; bwd=A[:,0,self.hidden:]
        last=torch.cat([fwd,bwd],dim=1)
        mask=torch.arange(T,device=x.device).unsqueeze(0)<L.unsqueeze(1).to(x.device)
        maxp=A.masked_fill(~mask.unsqueeze(-1),-1e9).max(dim=1).values
        return last+maxp
device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
with open(f'{DATA}/fusion_data.pkl','rb') as f: D=pickle.load(f)
model=FusionModel(feat_dim=D['meta']['n_features'])
model.load_state_dict(torch.load(f'{OUT}/fusion_model.pt',map_location='cpu')); model.to(device).eval()
Xte,_=D['test']['X_tensor'],D['test']['y']
yt=D['test']['y']
fault_idx=np.where(yt==1)[0]
idx=fault_idx[int(np.argmax([len(Xte[i]) for i in fault_idx]))]
s=Xte[idx]; L=min(len(s),MAXLEN)
Xp=np.zeros((1,MAXLEN,6),dtype=np.float32); Xp[0,:L]=s[:L]
x=torch.FloatTensor(Xp).to(device).requires_grad_(True); l=torch.LongTensor([L]).to(device)
logit=model(x,l,torch.zeros((1,D['meta']['n_features']),device=device))[0,1]
model.zero_grad(); logit.backward(); gx=x.grad[0].detach().cpu().numpy()[:L]
fig,axes=plt.subplots(2,1,figsize=(9,5),gridspec_kw={'height_ratios':[1,2.2]})
ax=axes[0]; sig_colors=['#2c3e50','#e67e22','#27ae60','#8e44ad','#c0392b','#16a085']
for j in range(6): ax.plot(s[:L,j],color=sig_colors[j],lw=1.0,label=FEATS_EN[j])
ax.set_ylabel('Normalized signal'); ax.set_title(f'Example fault sequence (#{idx}, L={L}) + input-gradient attribution')
ax.legend(frameon=False,ncol=3,fontsize=8,loc='upper right')
ax=axes[1]; gx_abs=np.abs(gx); gx_norm=gx_abs/(gx_abs.max()+1e-9)
im=ax.imshow(gx_norm.T,aspect='auto',cmap='Reds',interpolation='nearest')
ax.set_xlabel('Time step'); ax.set_yticks(range(6)); ax.set_yticklabels(FEATS_EN)
ax.set_title('|∂logit/∂x| attribution heatmap (1D-GradCAM, input gradient)')
fig.colorbar(im,ax=ax,label='Normalized |gradient|')
fig.tight_layout(); fig.savefig(f'{OUT}/fusion_gradcam_heatmap.png',dpi=200,bbox_inches='tight'); plt.close(fig)
print('3 fusion GradCAM figures saved.')
