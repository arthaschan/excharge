#!/usr/bin/env python3
"""Route B: Train baseline models on real Nature dataset sequences.
Models: XGBoost, RandomForest, MLP, Transformer (PyTorch).
Metrics: Accuracy, Recall, Precision, F1, AUC.
"""
import pandas as pd, numpy as np, json, os, time, sys, warnings
warnings.filterwarnings('ignore')

DATA = '/Users/arthas/git/excharge/data/real/'
OUT  = '/Users/arthas/git/excharge/docs/'

# Load data
X_train = pd.read_parquet(f'{DATA}/seq_X_train.parquet')
X_val   = pd.read_parquet(f'{DATA}/seq_X_val.parquet')
X_test  = pd.read_parquet(f'{DATA}/seq_X_test.parquet')
y_train = pd.read_parquet(f'{DATA}/seq_y_train.parquet')['label'].values
y_val   = pd.read_parquet(f'{DATA}/seq_y_val.parquet')['label'].values
y_test  = pd.read_parquet(f'{DATA}/seq_y_test.parquet')['label'].values

feat_cols = list(X_train.columns)
print(f'Train: {len(X_train):,} ({y_train.sum()} fault, {y_train.mean()*100:.2f}%)')
print(f'Val:   {len(X_val):,} ({y_val.sum()} fault)')
print(f'Test:  {len(X_test):,} ({y_test.sum()} fault)')
print(f'Features: {len(feat_cols)}')

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

results = {}

# ============================================================
# 1. XGBoost
# ============================================================
print('\n=== Training XGBoost ===', flush=True)
from xgboost import XGBClassifier
t0 = time.time()
xgb = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
    use_label_encoder=False, eval_metric='logloss',
    random_state=42, n_jobs=-1
)
xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
xgb_pred = xgb.predict(X_test)
xgb_prob = xgb.predict_proba(X_test)[:,1]
results['XGBoost'] = {
    'Acc': accuracy_score(y_test, xgb_pred),
    'Prec': precision_score(y_test, xgb_pred, zero_division=0),
    'Recall': recall_score(y_test, xgb_pred),
    'F1': f1_score(y_test, xgb_pred),
    'AUC': roc_auc_score(y_test, xgb_prob),
    'time_s': time.time()-t0,
}
print(f'XGBoost done in {time.time()-t0:.1f}s')
print(classification_report(y_test, xgb_pred, target_names=['Normal','Fault'], digits=4))

# Feature importance
fi = pd.Series(xgb.feature_importances_, index=feat_cols).sort_values(ascending=False)
print('Top15 XGBoost features:')
for f, v in fi.head(15).items():
    print(f'  {f}: {v:.4f}')

# ============================================================
# 2. RandomForest
# ============================================================
print('\n=== Training RandomForest ===', flush=True)
t0 = time.time()
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:,1]
results['RandomForest'] = {
    'Acc': accuracy_score(y_test, rf_pred),
    'Prec': precision_score(y_test, rf_pred, zero_division=0),
    'Recall': recall_score(y_test, rf_pred),
    'F1': f1_score(y_test, rf_pred),
    'AUC': roc_auc_score(y_test, rf_prob),
    'time_s': time.time()-t0,
}
print(f'RF done in {time.time()-t0:.1f}s')
print(classification_report(y_test, rf_pred, target_names=['Normal','Fault'], digits=4))

# ============================================================
# 3. MLP
# ============================================================
print('\n=== Training MLP ===', flush=True)
t0 = time.time()
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(256,128,64), max_iter=300, early_stopping=True,
                    validation_fraction=0.1, random_state=42)
mlp.fit(X_train_s, y_train)
mlp_pred = mlp.predict(X_test_s)
mlp_prob = mlp.predict_proba(X_test_s)[:,1]
results['MLP'] = {
    'Acc': accuracy_score(y_test, mlp_pred),
    'Prec': precision_score(y_test, mlp_pred, zero_division=0),
    'Recall': recall_score(y_test, mlp_pred),
    'F1': f1_score(y_test, mlp_pred),
    'AUC': roc_auc_score(y_test, mlp_prob),
    'time_s': time.time()-t0,
}
print(f'MLP done in {time.time()-t0:.1f}s')
print(classification_report(y_test, mlp_pred, target_names=['Normal','Fault'], digits=4))

# ============================================================
# 4. PyTorch Transformer
# ============================================================
try:
    import torch, torch.nn as nn
    has_torch = True
except:
    has_torch = False
    print('PyTorch not available, skipping Transformer')

if has_torch:
    print('\n=== Training Transformer ===', flush=True)
    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Build windowed sequences from raw data for Transformer
    df = pd.read_parquet(f'{DATA}/all_data.parquet')
    # Filter to train owners and >= 30 points
    train_owners = ['Sheet1','Sheet2','Sheet3','Sheet4']
    # Build tx->label map from seq-level
    tx_label = {}
    for _, row in pd.concat([
        pd.read_parquet(f'{DATA}/seq_X_train.parquet').assign(label=y_train),
        pd.read_parquet(f'{DATA}/seq_X_val.parquet').assign(label=y_val),
    ]).iterrows():
        pass  # we already have labels
    # Use sequence-level aggregate features as tabular — no windowing for simplicity
    # Transformer on aggregated features
    class TransformerClassifier(nn.Module):
        def __init__(self, n_features, d_model=64, nhead=4, n_layers=2, dropout=0.1):
            super().__init__()
            self.input_proj = nn.Linear(n_features, d_model)
            self.cls_token = nn.Parameter(torch.randn(1,1,d_model))
            self.pos_embed = nn.Parameter(torch.randn(1, 10, d_model))
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                        dropout=dropout, batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 2)
            )
        def forward(self, x):
            # x: (batch, n_features) → repeat to fake sequence
            B = x.shape[0]
            x = x.unsqueeze(1)  # (B, 1, d_feat)
            # pad to 10
            x = torch.nn.functional.pad(x, (0,0,0, 10-1))  # (B,10,d_feat)
            # project to d_model
            x = self.input_proj(x)
            # add cls token
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)  # (B, 11, d_model)
            # add pos embed (truncate if needed)
            n_pos = min(x.shape[1], self.pos_embed.shape[1])
            x[:, :n_pos] += self.pos_embed[:, :n_pos]
            x = self.encoder(x)
            cls_out = x[:,0]
            return self.head(cls_out)

    model = TransformerClassifier(n_features=len(feat_cols)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, (y_train==0).sum()/(y_train==1).sum()]).to(device)
    )

    X_tr_t = torch.FloatTensor(X_train_s).to(device)
    y_tr_t = torch.LongTensor(y_train).to(device)
    X_va_t = torch.FloatTensor(X_val_s).to(device)
    y_va_t = torch.LongTensor(y_val).to(device)
    X_te_t = torch.FloatTensor(X_test_s).to(device)
    y_te_t = torch.LongTensor(y_test).to(device)

    best_f1, best_state = 0, None
    batch_size = 256
    for epoch in range(60):
        model.train()
        perm = torch.randperm(len(X_tr_t))
        epoch_loss = 0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i+batch_size]
            out = model(X_tr_t[idx])
            loss = criterion(out, y_tr_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_out = model(X_va_t)
            val_pred = val_out.argmax(1).cpu().numpy()
            val_f1 = f1_score(y_val, val_pred)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        if (epoch+1) % 10 == 0:
            print(f'  Epoch {epoch+1}: loss={epoch_loss:.4f} val_f1={val_f1:.4f} best={best_f1:.4f}', flush=True)

    # Load best, test
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_out = model(X_te_t)
        test_prob = torch.softmax(test_out, 1)[:,1].cpu().numpy()
        test_pred = test_out.argmax(1).cpu().numpy()
    results['Transformer'] = {
        'Acc': accuracy_score(y_test, test_pred),
        'Prec': precision_score(y_test, test_pred, zero_division=0),
        'Recall': recall_score(y_test, test_pred),
        'F1': f1_score(y_test, test_pred),
        'AUC': roc_auc_score(y_test, test_prob),
        'time_s': time.time()-t0,
    }
    print(f'Transformer done in {time.time()-t0:.1f}s')
    print(classification_report(y_test, test_pred, target_names=['Normal','Fault'], digits=4))

# ============================================================
# Summary
# ============================================================
print('\n' + '='*60)
print('=== Model Comparison ===')
print(f'{"Model":<18} {"Acc":>8} {"Prec":>8} {"Recall":>8} {"F1":>8} {"AUC":>8} {"Time(s)":>8}')
print('-'*60)
for name, r in sorted(results.items(), key=lambda x: -x[1]['F1']):
    print(f'{name:<18} {r["Acc"]:>8.4f} {r["Prec"]:>8.4f} {r["Recall"]:>8.4f} {r["F1"]:>8.4f} {r["AUC"]:>8.4f} {r["time_s"]:>8.1f}')

# Save
results_df = pd.DataFrame(results).T
results_df.to_csv(f'{OUT}/routeB_baseline_results.csv')
print(f'\nSaved results to {OUT}/routeB_baseline_results.csv')

# Nature paper baseline comparison
print('\n=== Comparison with Nature Paper (73.56% Recall) ===')
for name, r in results.items():
    gap = r['Recall']*100 - 73.56
    print(f'{name}: Recall={r["Recall"]*100:.2f}% (gap: {gap:+.2f}pp vs Nature)')

with open(f'{OUT}/routeB_baseline_report.md','w') as f:
    f.write('# Route B 基线模型结果\n\n')
    f.write(f'**数据集**: Nature 论文配套真实数据集（深圳 Autosun 30 座充电站 2020-2023）\n\n')
    f.write(f'- 训练集（owners 1-4）: {len(X_train):,} 序列，故障 {y_train.sum():,}（{y_train.mean()*100:.2f}%）\n')
    f.write(f'- 验证集（owners 5-6）: {len(X_val):,} 序列，故障 {y_val.sum():,}\n')
    f.write(f'- 测试集（owners 7-8）: {len(X_test):,} 序列，故障 {y_test.sum():,}\n\n')
    f.write(f'**特征数**: {len(feat_cols)}\n\n')
    f.write('## 模型对比\n\n')
    f.write(results_df.to_markdown() + '\n\n')
    f.write('## Nature Paper 基线对比\n\n')
    f.write('| Model | Recall | Gap vs Nature |\n')
    f.write('|-------|--------|-------------|\n')
    for name, r in sorted(results.items(), key=lambda x: -x[1]['Recall']):
        gap = r['Recall']*100 - 73.56
        f.write(f'| {name} | {r["Recall"]*100:.2f}% | {gap:+.2f}pp |\n')
    f.write('\n## 特征重要性（XGBoost Top10）\n\n')
    fi_df = fi.head(10).to_frame('importance')
    f.write(fi_df.to_markdown() + '\n')
    f.write('\n---\n*Generated by excharge Route B pipeline*\n')

print('\nAll done!')
