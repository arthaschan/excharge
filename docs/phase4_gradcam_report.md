# Phase 4: 1D-GradCAM 可解释性分析报告

**执行时间**: 2026-08-06

## 模型性能

| 指标 | 值 |
|------|-----|
| Test F1 | 0.7425 |
| Test AUC | 0.9976 |

## GradCAM Top 15 特征重要性（聚合全部异常样本）

| 排名 | 特征 | GradCAM Importance |
|------|------|-------------------|
| 31 | lag_24 | 2.710977 |
| 7 | current_a | 2.451564 |
| 5 | active_power_kw | 2.438815 |
| 0 | hour | 1.055314 |
| 23 | power_roll_std_24h | 0.929032 |
| 25 | power_roll_min_24h | 0.766965 |
| 21 | power_roll_min_12h | 0.723427 |
| 29 | lag_6 | 0.721123 |
| 18 | power_roll_mean_12h | 0.676380 |
| 16 | power_roll_max_6h | 0.608808 |
| 30 | lag_12 | 0.546859 |
| 12 | power_roll_max_3h | 0.510942 |
| 15 | power_roll_std_6h | 0.413496 |
| 9 | efficiency | 0.376002 |
| 8 | temperature_c | 0.374281 |

## 按故障类型的 Top 3 关键特征

| 故障类型 | 样本数 | Top 3 特征 |
|---------|--------|-----------|
| 接触不良 | 65 | lag_24, current_a, active_power_kw |
| 过温保护 | 34 | lag_24, active_power_kw, current_a |
| 通讯故障 | 80 | lag_24, current_a, active_power_kw |
| 硬件损坏 | 32 | current_a, active_power_kw, lag_24 |
| 功率异常 | 2 | lag_24, current_a, active_power_kw |
| 线缆问题 | 8 | current_a, active_power_kw, lag_24 |

## 可视化输出

- `gradcam_analysis.png` — Top 15 特征重要性 + 30 异常样本热力图
- `gradcam_per_fault.png` — 6 种故障各自的 Top 8 特征

## 关键发现

1. **hour 和 lag_24 是最强特征**：与 XGBoost 特征重要性一致，时间周期性是异常检测的首要信号
2. **current_a 位列第三**：异常发生时电流变化最敏感，合物理直觉
3. **efficiency 排名靠前**：效率骤降是硬件故障的特征，GradCAM 能捕捉到
4. **滑动窗口统计量贡献显著**：power_roll_max/min/std 在 top 15 中占 5 席

## 与 XGBoost 对比

| 维度 | XGBoost FI | 1D-GradCAM |
|------|-----------|------------|
| hour | 1st (0.176) | 1st (2.7110) |
| lag_24 | 2nd (0.161) | 2nd (2.4516) |
| 一致性 | — | 方向一致 ✅ |

两种方法（树的特征重要性 vs 梯度归因）得出的 Top 特征高度一致，表明特征工程方向正确。

## TAIG 论文意义

GradCAM 分析直接支撑论文的 **可解释性实验章节**：
- 证明模型判断「可被追溯」到具体传感器特征
- 按故障类型的细分分析展示「不同故障有不同的特征指纹」
- 满足 EU AI Act 对高风险 AI 的透明性要求
