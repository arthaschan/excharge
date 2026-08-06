# TAIG 研究计划：充电站可解释异常检测
## —— 模拟数据生成 + 研究全流程

> 启动日期：2026-08-06

---

## 一、模拟数据生成（已完成 ✅）

### 1.1 数据规模

| 数据集 | 路径 | 规模 | 说明 |
|--------|------|------|------|
| `stations.csv` | `simulated_data/` | 100 行 | 充电站基本信息（编号、行政区、GPS） |
| `piles.csv` | `simulated_data/` | 2,000 行 | 充电桩清单（编号、所属站、额定功率、投运日期） |
| `fault_records.csv` | `simulated_data/` | 6,949 行 | 故障记录（站长标注用，含开始/结束时间、故障类型、维修措施） |
| `daily_summaries.parquet` | `simulated_data/` | 730,000 行 | 日级汇总（2,000 桩 × 365 天） |
| `hourly_3stations.parquet` | `simulated_data/` | 525,600 行 | 小时级时序数据（3 站 × 60 桩 × 8,760 小时）|

### 1.2 关键参数

| 参数 | 值 |
|------|-----|
| 充电站数量 | 100 |
| 每站充电桩数 | 20 |
| 总充电桩数 | 2,000 |
| 时间跨度 | 2025-08-01 ~ 2026-07-31（365 天）|
| 时序粒度 | 15 分钟（原始）/ 1 小时（模拟抽样）|
| **故障率** | **~1%**（异常天数占比 1.08%）|
| 每桩平均故障次数 | 3.5 次/年 |
| 故障平均持续时间 | 4.0 小时（中位数 2.7h）|

### 1.3 故障类型分布

| 故障类型 | 数量 | 占比 | 模拟原因 |
|---------|------|------|---------|
| 接触不良 | 2,379 | 34.2% | 充电枪头接触不良/松动 |
| 过温保护 | 1,765 | 25.4% | 设备过热保护触发 |
| 通讯故障 | 1,432 | 20.6% | 模块通讯中断/丢包 |
| 硬件损坏 | 664 | 9.6% | 功率模块/继电器/屏幕损坏 |
| 功率异常 | 360 | 5.2% | 输出功率异常波动 |
| 线缆问题 | 349 | 5.0% | 线缆老化/破损/断股 |
| **合计** | **6,949** | **100%** | |

### 1.4 充电站地理分布

```
宝安 13 | 南山 18 | 福田 6 | 罗湖 10 | 龙岗 13 | 龙华 9 | 光明 7 | 坪山 10 | 盐田 14
```

### 1.5 充电桩额定功率分布

```
60kW: 383 | 120kW: 393 | 180kW: 400 | 240kW: 428 | 360kW: 396
```

---

## 二、小时级时序数据字段说明（`hourly_3stations.parquet`）

| 字段 | 含义 | 单位 | 类型 |
|------|------|------|------|
| `pile_id` | 充电桩编号 | — | 字符串 |
| `station_id` | 所属充电站编号 | — | 字符串 |
| `timestamp` | 时间戳 | yyyy-MM-dd HH:mm | datetime |
| `active_power_kw` | 有功功率 | kW | float |
| `voltage_v` | 电压 | V | float |
| `current_a` | 电流 | A | float |
| `temperature_c` | 设备温度 | °C | float |
| `efficiency` | 转换效率 | 0~1 | float |
| `is_fault` | 是否故障 | 0/1 | int |
| `fault_type` | 故障类型 | — | string (故障时非空) |

**功率统计（非零时段）：min=0.0kW, max=439.4kW, mean=76.7kW**
**温度范围：17.4 ~ 69.5°C**

---

## 三、研究全流程

### Phase 1：文献调研与背景研究 📚（进行中）

```
Step 1.1：GradCAM 及可解释性方法调研
   ├── GradCAM 原始论文（Selvaraju et al., 2017）
   ├── 时序 GradCAM 变体（GradCAM-TS, 1D GradCAM）
   ├── Integrated Gradients, SHAP 对比
   └── 输出：可解释性方法文献综述

Step 1.2：充电站异常检测现状
   ├── EV 充电基础设施故障分析
   ├── 时序异常检测公共数据集
   └── 输出：充电站故障检测现有方法评估

Step 1.3：AI 治理会议热点
   ├── TAIG Workshop 近 2 年论文主题分析
   ├── 透明性与可解释性话题趋势
   └── 输出：投稿策略建议

Step 1.4：有监督时序异常检测进展
   ├── LSTM/Transformer/TCN 分类性能对比
   ├── 极度不平衡数据处理（SMOTE, Focal Loss）
   └── 输出：模型选型建议
```

→ 输出文档：`simulated_data/literature_review.md`

### Phase 2：数据准备与标注 🏷️

```
Step 2.1：数据预处理
   ├── 时间戳对齐、缺失值填充
   ├── 特征工程（滑动窗口统计、日周期编码、设备特征）
   └── 输出：标准化特征矩阵

Step 2.2：标注数据集构建
   ├── 从 fault_records.csv 反向打标签
   ├── 非故障时段自动标注为「正常」
   └── 输出：完整标注数据集（feature_matrix + labels）

Step 2.3：数据划分
   ├── Train 70% / Val 15% / Test 15%
   ├── 按时间划分（避免数据泄漏）
   └── 输出：train/val/test splits
```

### Phase 3：模型训练 🤖

```
Step 3.1：Baseline 模型
   ├── Random Forest + 特征工程
   ├── XGBoost + 滑动窗口
   └── 建立性能基准

Step 3.2：深度学习模型
   ├── LSTM（双向，2 层）
   ├── Transformer Encoder（8 heads, 4 layers）
   ├── TCN（dilated convolutions）
   └── 对比实验

Step 3.3：不平衡处理
   ├── Focal Loss（α=0.25, γ=2）
   ├── SMOTE 过采样
   ├── 加权损失函数
   └── 输出：最优模型
```

### Phase 4：可解释性分析 🔍

```
Step 4.1：GradCAM 实现
   ├── 1D GradCAM：对时序卷积层进行梯度反向传播
   ├── 时间维度可视化（哪个时间段贡献最大）
   ├── 特征维度可视化（功率/电压/温度/效率的贡献度）
   └── 输出：GradCAM 热力图

Step 4.2：SHAP 对比分析
   ├── SHAP 全局特征重要性排名
   ├── SHAP 局部解释（单条异常预测的归因）
   └── 与 GradCAM 结果对比验证

Step 4.3：可解释性质量评估
   ├── Fidelity（忠实度）：特征移除后预测变化量
   ├── AOPC（Average Percentage Point Change）
   ├── 专家意见（导师/工程师主观评估）
```

### Phase 5：论文撰写 ✍️

```
Step 5.1：论文结构
   ├── Abstract（150~200 词）
   ├── Introduction（问题背景、研究动机、贡献点）
   ├── Related Work（文献综述）
   ├── Methodology（数据 → 模型 → 解释）
   ├── Experiments（数据集、实验设置、结果、消融实验）
   ├── Analysis（可解释性分析与讨论）
   └── Conclusion（总结与展望）

Step 5.2：投稿准备
   ├── TAIG Workshop 模板排版
   ├── 作者列表确认
   └── 导师审阅
```

---

## 四、论文核心贡献点（预期）

1. **数据贡献**：首个融合充电站多维时序数据与 GradCAM 可解释性分析的模拟/实测数据集
2. **方法贡献**：提出针对充电站时序数据的 GradCAM 可解释异常检测框架
3. **应用贡献**：将 AI 治理的透明性原则落地到新能源基础设施运维场景
4. **对比贡献**：GradCAM vs SHAP 在时序异常解释任务上的系统性比较

---

## 五、文件结构

```
simulated_data/
├── stations.csv                 # 100 充电站基本信息
├── piles.csv                    # 2,000 充电桩清单
├── fault_records.csv            # 6,949 故障记录（站长标注用）
├── daily_summaries.parquet      # 730K 行日级汇总
├── hourly_3stations.parquet     # 525.6K 行小时级时序（3站×60桩）
├── generate.py                  # 数据生成脚本
└── literature_review.md         # 文献调研报告（待完成）
```
