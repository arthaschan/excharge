import pandas as pd, numpy as np, json
from datetime import datetime

OUT = '/Users/arthas/.qclaw/workspace-dhj4e57a67drnnbd/simulated_data'
print("Phase 2: Data Preprocessing")

# Load data
hourly = pd.read_parquet(f'{OUT}/hourly_3stations.parquet')
hourly['timestamp'] = pd.to_datetime(hourly['timestamp'])
piles = pd.read_csv(f'{OUT}/piles.csv')
faults = pd.read_csv(f'{OUT}/fault_records.csv')
faults['fault_start'] = pd.to_datetime(faults['fault_start'])
faults['fault_end'] = pd.to_datetime(faults['fault_end'])
print(f"Loaded: {len(hourly)} hourly rows, {len(piles)} piles, {len(faults)} faults")

# ====== 1. Label Engineering ======
print("Building labels...")
hourly['label_binary'] = 0
hourly['label_multiclass'] = 0

fault_type_map = {'接触不良':1, '过温保护':2, '通讯故障':3, '硬件损坏':4, '功率异常':5, '线缆问题':6}

for i, f in faults.iterrows():
    mask = (hourly['pile_id'] == f['pile_id']) & (hourly['timestamp'] >= f['fault_start']) & (hourly['timestamp'] <= f['fault_end'])
    hourly.loc[mask, 'label_binary'] = 1
    hourly.loc[mask, 'label_multiclass'] = fault_type_map.get(f['fault_type'], 0)

print(f"Labels: abnormal_records={hourly['label_binary'].sum()} ({100*hourly['label_binary'].sum()/len(hourly):.4f}%)")

# ====== 2. Time Features ======
hourly['hour'] = hourly['timestamp'].dt.hour
hourly['dayofweek'] = hourly['timestamp'].dt.dayofweek
hourly['month'] = hourly['timestamp'].dt.month
hourly['is_weekend'] = (hourly['dayofweek'] >= 5).astype(int)

# ====== 3. Merge power_rating ======
hourly = hourly.merge(piles[['pile_id','power_rating_kw']], on='pile_id', how='left')

# ====== 4. Sliding window features (per pile) ======
print("Computing window features...")
hourly = hourly.sort_values(['pile_id','timestamp']).reset_index(drop=True)

window_sizes = [3, 6, 12, 24]
result_parts = []
for pid, grp in hourly.groupby('pile_id'):
    grp = grp.sort_values('timestamp').copy()
    for w in window_sizes:
        grp[f'power_roll_mean_{w}h'] = grp['active_power_kw'].rolling(w, min_periods=1).mean().shift(1)
        grp[f'power_roll_std_{w}h'] = grp['active_power_kw'].rolling(w, min_periods=1).std().shift(1).fillna(0)
        grp[f'power_roll_max_{w}h'] = grp['active_power_kw'].rolling(w, min_periods=1).max().shift(1)
        grp[f'power_roll_min_{w}h'] = grp['active_power_kw'].rolling(w, min_periods=1).min().shift(1)
    grp[f'temp_roll_mean_{w}h'] = grp['temperature_c'].rolling(w, min_periods=1).mean().shift(1)
    grp[f'temp_roll_max_{w}h'] = grp['temperature_c'].rolling(w, min_periods=1).max().shift(1)
    
    # Lag features
    grp['lag_1'] = grp['active_power_kw'].shift(1)
    grp['lag_6'] = grp['active_power_kw'].shift(6)
    grp['lag_12'] = grp['active_power_kw'].shift(12)
    grp['lag_24'] = grp['active_power_kw'].shift(24)
    
    result_parts.append(grp)
    if len(result_parts) % 10 == 0:
        print(f"  Processed {len(result_parts)}/60 piles...")

hourly = pd.concat(result_parts, ignore_index=True)
# Fill remaining NaNs
hourly = hourly.fillna(0)
print(f"Processed all piles. Shape: {hourly.shape}")

# ====== 5. Train/Val/Test Split ======
print("Splitting...")
hourly = hourly.sort_values('timestamp')
train_end = pd.Timestamp('2026-04-12')
val_end = pd.Timestamp('2026-05-31')

train = hourly[hourly['timestamp'] <= train_end].copy()
val = hourly[(hourly['timestamp'] > train_end) & (hourly['timestamp'] <= val_end)].copy()
test = hourly[hourly['timestamp'] > val_end].copy()

print(f"Train: {len(train):,}, Val: {len(val):,}, Test: {len(test):,}")
print(f"Train abnormal: {train['label_binary'].sum():,} ({100*train['label_binary'].sum()/len(train):.4f}%)")
print(f"Val abnormal:   {val['label_binary'].sum():,} ({100*val['label_binary'].sum()/len(val):.4f}%)")
print(f"Test abnormal:  {test['label_binary'].sum():,} ({100*test['label_binary'].sum()/len(test):.4f}%)")

# ====== 6. Define feature columns ======
feature_cols = ['hour','dayofweek','month','is_weekend','power_rating_kw',
    'active_power_kw','voltage_v','current_a','temperature_c','efficiency']
for w in window_sizes:
    feature_cols += [f'power_roll_mean_{w}h',f'power_roll_std_{w}h',f'power_roll_max_{w}h',f'power_roll_min_{w}h']
for w in window_sizes:
    feature_cols += [f'temp_roll_mean_{w}h',f'temp_roll_max_{w}h']
feature_cols += ['lag_1','lag_6','lag_12','lag_24']
# Filter to existing columns only
feature_cols = [c for c in feature_cols if c in hourly.columns]

print(f"Feature columns: {len(feature_cols)}")

# ====== 7. Save ======
train.to_parquet(f'{OUT}/processed_hourly_train.parquet', index=False)
val.to_parquet(f'{OUT}/processed_hourly_val.parquet', index=False)
test.to_parquet(f'{OUT}/processed_hourly_test.parquet', index=False)
json.dump({'feature_columns': feature_cols, 'n_features': len(feature_cols)}, open(f'{OUT}/feature_columns.json','w'))
print(f"Saved train/val/test parquets + feature_columns.json")

# ====== 8. Report ======
report = f"""# Phase 2 预处理报告

**执行时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## 数据规模
| 数据集 | 行数 | 异常数 | 异常占比 |
|--------|------|--------|---------|
| Train | {len(train):,} | {train['label_binary'].sum():,} | {100*train['label_binary'].sum()/len(train):.4f}% |
| Val   | {len(val):,} | {val['label_binary'].sum():,} | {100*val['label_binary'].sum()/len(val):.4f}% |
| Test  | {len(test):,} | {test['label_binary'].sum():,} | {100*test['label_binary'].sum()/len(test):.4f}% |
| **Total** | **{len(hourly):,}** | **{hourly['label_binary'].sum():,}** | **{100*hourly['label_binary'].sum()/len(hourly):.4f}%** |

## 多分类标签分布
```
{hourly['label_multiclass'].value_counts().sort_index().to_string()}
```

## 特征工程
- 时间特征：hour, dayofweek, month, is_weekend
- 设备特征：power_rating_kw
- 滑动窗口（3/6/12/24h）：power_mean/std/max/min, temp_mean/max
- 滞后特征：lag_1, lag_6, lag_12, lag_24
- **特征总数：{len(feature_cols)}**
"""
with open(f'{OUT}/PREPROCESSING_REPORT.md','w') as f:
    f.write(report)
print("Report saved.")
print("\nPhase 2 complete!")
