"""
重新计算全部基线的 PR-AUC，保证口径一致：
- 序列级（2776 序列）：XGBoost(v1)、XGBoost(57特征)、RF、MLP
- 窗口级（7768 窗口）：Transformer（routeB 复现）

输出：docs/routeA_pr_auc_results.json 更新版
"""
import numpy as np
import pandas as pd
import json
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DATA = 'data/real'
y_test = pd.read_parquet(f'{DATA}/seq_y_test.parquet')['label'].values
print(f"序列级测试集: {len(y_test)} 样本, 故障 {y_test.sum()} ({y_test.mean()*100:.2f}%)")

results = {}

# ---- 序列级模型 ----
X_test = pd.read_parquet(f'{DATA}/seq_X_test.parquet')
print(f"X_test 形状: {X_test.shape}")

# 1. XGBoost v1 (37特征) - 用 routeA 已有概率
for name, path in [
    ('XGBoost v1 (37特征)', 'docs/routeA_sweep_v1_baseline_prob.npy'),
    ('XGBoost v1+sel_v2 (57特征)', 'docs/routeA_final_prob.npy'),
]:
    prob = np.load(path)
    assert len(prob) == len(y_test), f"{name}: {len(prob)} != {len(y_test)}"
    results[name] = {
        'pr_auc': round(float(average_precision_score(y_test, prob)), 4),
        'auc': round(float(roc_auc_score(y_test, prob)), 4),
        'level': 'sequence'
    }
    print(f"{name}: PR-AUC={results[name]['pr_auc']}  AUC={results[name]['auc']}")

# 2. RF 重新训练（与 train_baseline_final.py 相同参数）
X_train = pd.read_parquet(f'{DATA}/seq_X_train.parquet')
y_train = pd.read_parquet(f'{DATA}/seq_y_train.parquet')['label'].values
print(f"训练集: {X_train.shape}, 故障 {y_train.sum()}")

rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
prob_rf = rf.predict_proba(X_test)[:, 1]
results['RandomForest'] = {
    'pr_auc': round(float(average_precision_score(y_test, prob_rf)), 4),
    'auc': round(float(roc_auc_score(y_test, prob_rf)), 4),
    'level': 'sequence'
}
print(f"RandomForest: PR-AUC={results['RandomForest']['pr_auc']}  AUC={results['RandomForest']['auc']}")

# 3. MLP 重新训练
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
mlp = MLPClassifier(hidden_layer_sizes=(256,128,64), max_iter=300, early_stopping=True,
                    random_state=42)
mlp.fit(X_train_s, y_train)
prob_mlp = mlp.predict_proba(X_test_s)[:, 1]
results['MLP'] = {
    'pr_auc': round(float(average_precision_score(y_test, prob_mlp)), 4),
    'auc': round(float(roc_auc_score(y_test, prob_mlp)), 4),
    'level': 'sequence'
}
print(f"MLP: PR-AUC={results['MLP']['pr_auc']}  AUC={results['MLP']['auc']}")

# ---- 窗口级 Transformer（routeB 复现，7768 窗口）----
tp = np.load('docs/routeB_transformer_prob.npy')
# 窗口级标签需从 routeB 脚本确认；先用 routeB 结果 JSON 的 AUC 反查
# 这里只记录概率文件，标签在窗口构建脚本里
print(f"\nTransformer 窗口级概率: {tp.shape}")
# 窗口级标签：从 train_transformer_real.py 找测试标签生成方式
try:
    import subprocess
    # 检查是否有保存的窗口标签
    import os
    for f in ['data/real/window_y_test.npy', 'data/real/y_test_windows.npy']:
        if os.path.exists(f):
            wy = np.load(f)
            print(f'找到窗口标签 {f}: {wy.shape}, 故障 {wy.sum()}')
            results['Transformer (windowed)'] = {
                'pr_auc': round(float(average_precision_score(wy, tp)), 4),
                'auc': round(float(roc_auc_score(wy, tp)), 4),
                'level': 'window'
            }
            print(f"Transformer(windowed): PR-AUC={results['Transformer (windowed)']['pr_auc']}  AUC={results['Transformer (windowed)']['auc']}")
except Exception as e:
    print('窗口标签查找失败:', e)

# 随机基线
results['random_baseline'] = {'pr_auc': round(float(y_test.mean()), 4), 'level': 'sequence'}
results['_meta'] = {
    'y_test_size': int(len(y_test)),
    'n_fault': int(y_test.sum()),
    'fault_rate': round(float(y_test.mean()), 4),
}

with open('docs/routeA_pr_auc_results.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print('\n已保存 docs/routeA_pr_auc_results.json')
print(json.dumps(results, indent=2, ensure_ascii=False))
