# 数据来源与文件说明

## 1. 原始数据集（外部，需下载）

论文使用的原始数据来自 Yang 等人的公开数据集（Nature Communications 论文，
见论文参考文献 [3]）：

- 数据集托管：Mendeley Data —— DOI: `10.17632/c7gg94tmvz.3`
- 作者代码托管：Zenodo —— DOI: `10.5281/zenodo.17423221`

原始文件名为 `processed_data.xlsx`，含 8 个 sheet（Sheet1–Sheet8，对应 8 个
数据所有者 owner1–owner8）。全量 155.6 万采样点（2020-07 至 2024-03）。

### 放置方式（二选一）

- 默认：把 `processed_data.xlsx` 放到本文件夹的 `data/raw/processed_data.xlsx`；
- 或用环境变量指定：`export RAW_XLSX=/path/to/processed_data.xlsx`。

然后运行 `scripts/convert_real_data.py` 生成 `data/real/all_data.parquet`
（以及 `sequences.parquet`）。

## 2. 中间数据（本文件夹已内置，可直接用于复现）

本文件夹 `data/real/` 已内置三个中间文件，可直接复现论文全部结果，无需重新下载
原始 xlsx 或重跑预处理：

| 文件 | 大小 | 来源脚本 | 用途 |
|------|------|---------|------|
| `all_data.parquet` | ~35 MB | convert_real_data.py | 全量采样点（原始单位），供 §4.5 机制分析按 transaction_id 取原始信号 |
| `seq_tensors.pkl` | ~35 MB | build_seq_tensors.py | 序列级 6 通道张量（z-score, 变长 list），供端到端 Bi-LSTM 基线 |
| `fusion_data.pkl` | ~41 MB | build_fusion_data.py | 融合对齐数据：X_tensor[L,6] + X_feat[62维 z-score] + y + tx，供 Token-Attn / GBDT / 归因 / 消融 |

三者均可由 `all_data.parquet` 重新生成（seed=42 分层抽样，确定性）：

```
python scripts/build_seq_tensors.py      # → data/real/seq_tensors.pkl
python scripts/build_fusion_data.py      # → data/real/fusion_data.pkl
```

## 3. 数据划分口径（不可变）

- 训练集：owner1–6（Sheet1–Sheet6），内部按 80/20 分层切出 val
  （train 13,505 序列 / 642 故障；val 3,377 / 160 故障）
- 测试集：owner7–8（Sheet7–Sheet8），全新站点跨域
  （2,776 序列 / 129 故障）
- 序列筛选：每条充电序列 ≥ 30 采样点，共 19,658 条有效序列
- 特征：6 通道原始时序 + 62 维手工特征（基础统计量 32 / 温度变化 6 /
  分段端点差分 15 / 超温标志 4 / 电池类型 5）
- 归一化：序列每通道 z-score（每序列内）；62 维特征 z-score（fit on train）
- 采样点级故障率：5.68%（全量 155.6 万采样点，88,394 个故障采样点）
- 电池类型分布（序列级，19,658 条有效序列）：LFP 61.7% / NMC 36.7% /
  LMO 1.2% / LCO 0.1% / LP 0.2%（types 编码：3=LFP, 6=NMC, 4=LMO,
  5=LCO, 7=LP，见 build_fusion_data.py 的 `bt_map`）

## 4. 复现口径注意事项

- 序列 padding 至 MAXLEN=200，超长序列取前 200 步（超长占比 <2%，对故障
  影响 <1%，见论文 §5.3 局限性）。
- 主指标 PR-AUC（类别不平衡下的阈值无关排序指标）。
- 重跑训练会因硬件浮点差异（CUDA vs 原环境 MPS）产生小幅波动，结论不变；
  详见 README「复现注意事项」。
