# Phase 1 门控报告（gate_report_phase1.md）

**项目**：充电过程在线早期预警（earlywarning）｜**日期**：2026-09-03
**执行人**：陈天元（本机 Mac）｜**依据**：《研究方案_充电过程在线早期预警_v1.0.md》§5.2 + gate_report_phase0 §8/§9
**数据**：`data/real/all_data.parquet`（31,449 事务）→ **prefix_dataset_full.parquet**（固化数据集）→ **prefix_feats_v1.parquet**（52 维特征，已冻结）
**代码**：`code/build_prefix_dataset.py` → `code/build_prefix_features.py` → `code/structure_ab.py`
**原始结果**：`data/prefix_features_v1.json`（schema 冻结）+ `docs/structure_ab_results.json`（A/B 结构判定，3-seed）

---

## 1. 结论先行：✅ Phase 1 数据管线定型完成 + 结构判定选 **A（两阶段级联）**

> **Phase 1 两大交付完成**：(1) 数据管线正式版全链路（固化数据集 + 特征 v1 冻结 + E11 泄漏自检通过）；(2) 多任务结构 A vs B 判定——**A 两阶段级联胜出**，作为 Phase 2 深度量产的默认结构。
> 判定理由（诚实表述）：A/B 在识别能力上高度接近（差异大部分在 3-seed std 内），但 **A 在终止检测主任务上 6/6 同分布口径一致略优**，且具备三重复合优势（阈值独立可调 / Stage2 训练池天然平衡 / 与论文双谱系叙事一一对应）。B 作为消融对照保留。

---

## 2. 交付物一：数据管线正式版（固化 + 冻结 + 自检）

### 2.1 固化数据集 prefix_dataset_full.parquet（19.5 MB，31,449 事务 = 每事务一行）

| 列 | 含义 |
|---|---|
| transaction_id / begin_time / owner / label / family / class_judge / types | 事务级元数据（family 双谱系：startup 4,827 / run 853 / normal 25,769） |
| dur_min / n_rows | 会话时长(min) / 采样行数 |
| offsets | 逐行偏移 (n,)：距 begin_time 的分钟（float32，读回误差 ≤3e-6） |
| vals_flat | 6 通道完整序列展平 (n*6,) float32，读回 reshape(-1,6) |

- **序列张量化关键设计**：存完整序列 + offsets，Phase 2 深度模型按 τ 现场截段 `[offsets ≤ τ]` + length mask——**无需重读 111MB 原始文件**，且任意 τ 网格可在线重切（E7 口径一致）；
- **读回正确性抽查**：5 事务与源数据逐值比对，通道误差 0.0、offsets 误差 ≤3e-6 ✅；
- **时间语义固化**：begin_time=会话开始（事务内恒定）、end_time=逐行采样戳（+60s）——前缀可见行 = `offsets ≤ τ`；
- 内含 E11 防线：offsets 非单调即抛错（构建期拦截）。

### 2.2 统一切分 split.json（E11-2 固化）

- 主口径 **跨站冷启动**：train = owner1-6（27,335 事务）→ test = owner7-8（4,114 事务）；
- 同分布随机分层（owner×label）作为方法学内参，由实验脚本按 seed 现场切分（与 Phase 0 同协议）；
- per-owner 故障统计固化进 split.json，后续任何脚本引用同一口径，防 owner 判定漂移。

### 2.3 特征 v1 冻结 prefix_feats_v1.parquet + prefix_features_v1.json

| 项 | 值 |
|---|---|
| 特征行 | 208,358（每事务 × 每前缀口径） |
| 特征维 | **52**（6 通道 × 8 统计 + SOC 增量 + 功率活跃度/达峰位置 + 枪温爬升） |
| 口径 | 时间 τ∈{1,2,3,5,10,20}min（主）+ 进度 p∈{10%,25%,50}%（稳健性对照） |
| 与 Phase 0 样本量对照 | **逐 τ 完全一致**（time@1: 3,524 … time@20: 21,917）→ gate_phase0 数字可复现 ✅ |
| E11 泄漏自检 | 0 违规；特征名黑名单（dur/n_rows/end_time…）0 命中 ✅ |
| 速度 | 17s（相对 Phase 0 逐行扫 111MB 提速 ~8×） |

> ⚠️ 版本纪律（E7）：**prefix_feats_v1 已冻结**。Phase 2 若增特征须升 v2 并重跑本报告的 A/B 判定与 gate_phase0 对照，否则数字不可比。

---

## 3. 交付物二：多任务结构 A vs B 判定（3-seed LightGBM，prefix_feats_v1）

### 3.1 判定表（同分布随机分层，主判据）

| τ | AP_term A / B | AP_run_fault A / B | AP_startup_fault A / B | AP_startup_e2e A / B |
|---|---|---|---|---|
| 1 min | **0.877** / 0.869 | 0.665 / 0.665 | 0.921 / 0.926 | **0.814** / 0.780 |
| 2 min | **0.883** / 0.875 | 0.704 / 0.702 | 0.926 / 0.926 | **0.740** / 0.726 |
| 3 min | **0.882** / 0.875 | 0.741 / 0.746 | 0.870 / 0.871 | **0.704** / 0.688 |
| 5 min | **0.899** / 0.897 | 0.770 / 0.785 | 0.852 / 0.849 | 0.673 / 0.675 |
| 10 min | **0.897** / 0.892 | 0.818 / 0.827 | 0.791 / 0.799 | 0.605 / 0.609 |
| 20 min | **0.900** / 0.889 | 0.886 / 0.888 | 0.590 / 0.559 | 0.448 / 0.476 |
| **均值差 A-B** | **+0.0071** | −0.0050 | +0.0010 | +0.0032 |

### 3.2 跨站（owner1-6 → 7-8，辅判据）

| τ | AP_term A / B | AP_run_fault A / B | AP_startup_e2e A / B |
|---|---|---|---|
| 1 min | 0.893 / 0.896 | 0.117 / 0.137 | 0.875 / 0.877 |
| 2 min | **0.927** / 0.912 | **0.274** / 0.233 | **0.916** / 0.906 |
| 3 min | **0.924** / 0.915 | 0.244 / 0.269 | **0.901** / 0.890 |
| 5 min | **0.887** / 0.882 | **0.299** / 0.278 | **0.863** / 0.849 |
| 10 min | **0.858** / 0.848 | **0.474** / 0.416 | **0.830** / 0.807 |
| 20 min | **0.747** / 0.739 | **0.687** / 0.674 | 0.595 / 0.652 |

### 3.3 判定解读（诚实版）

1. **统计上**：A-B 差（+0.002~+0.012）与 3-seed std（0.005~0.014）同量级，**不是显著胜出**——两者都是合格的表示；
2. **但 A 在"终止检测"主任务上 6/6 同分布 + 5/6 跨站口径一致占优**，方向无翻转；
3. **结构性优势（选 A 的核心理由）**：
   - **Stage1/Stage2 阈值独立可调**：运维可分别设"预警灵敏度"（Stage1）与"族报告置信度"（Stage2），B 单模型做不到；
   - **Stage2 训练池 = 纯故障子集**（startup 4,827 / run 853），天然回避 B 全量中 normal 25,769 对 run 的淹没（class_weight='balanced' 只是补偿，非根治）；
   - **与论文双谱系叙事一一对应**："先判会否终止，命中者再分启动/运行型" = 预警 + 分型两段式故事，可解释性/归因可分段做；
4. **附带观察（对 Phase 2/3 有价值）**：
   - run 族可分性随 τ 单调升（同分布 0.66@1min → 0.89@20min），跨站短前缀弱（≤5min 仅 0.12~0.30、≥10min 0.47~0.69）→ **run 型 EAR 天然晚于 startup**（物理合理：运行型故障前几分钟信号弱），双谱系分开报 EAR 的设定再次被支持；
   - startup 可分性随 τ 递减（0.92@1-2min → 0.59@20min）：能撑过 20min 的"启动型"本就稀少（301 例）且更难判——τ 越短对启动型越有利，EAR 叙事成立。

---

## 4. 对 Phase 2 的输入规格（冻结协议）

| 项 | 规格 |
|---|---|
| 结构 | **A 两阶段级联**（Stage1 终止检测 → Stage2 族分类）；B 三分类仅作消融对照保留 |
| 表格侧输入 | prefix_feats_v1.parquet（52 维，冻结） |
| 序列侧输入 | prefix_dataset_full.parquet：按 τ 截 `[offsets≤τ]` + length mask；padding 上限 = τ 对应行数上界 |
| 主口径 | 跨站 owner1-6 → 7-8（与会议论文一致）；同分布为内参 |
| 网格 | τ∈{1,2,3,5,10,20}min（τ=1 是 EAR 最大卖点） |
| 深度模型存在门槛 | 必须相对 LightGBM（同分布 AP_term 0.877~0.900 / 跨站 0.75~0.93）**显著增益 + 提供归因**，否则无存在意义（R1/R3/E1 纪律） |
| 量产纪律 | 7-seed 概率集成 + bootstrap CI + **逐 owner 报告**（Sheet8 样本少，gate_phase0 §4.3 协议） |
| 环境 | Mac 探路已结 → H100 CUDA 量产（Token-Attn 前缀变体迁移自 train_c1c2.py） |

---

## 5. 文件清单

```
earlywarning/
├── code/
│   ├── build_prefix_dataset.py     ← Phase 1 正式版（固化序列 + split.json）
│   ├── build_prefix_features.py    ← Phase 1 正式版（特征 v1 + schema 冻结 + E11 自检）
│   └── structure_ab.py             ← 结构 A/B 判定实验
├── data/
│   ├── prefix_dataset_full.parquet (19.5 MB)   ← 固化数据集（31,449 事务完整序列）
│   ├── split.json                              ← 跨站统一切分
│   ├── prefix_feats_v1.parquet (22.9 MB)       ← 特征 v1（208,358 × 52）
│   └── prefix_features_v1.json                 ← schema 冻结
└── docs/
    ├── gate_report_phase1.md       ← 本报告
    └── structure_ab_results.json   ← A/B 原始结果（3-seed）
```

---

## 6. 下一步（Phase 2 起点）

1. **H100 上 Token-Attn 前缀变体**：结构 A（双头/双模型级联），序列张量按 §4 规格构建；
2. 树基线定靶：LightGBM × 跨站主口径（表侧特征）作为深度模型的必须超越线；
3. 1-seed 探路（R9）→ 有效才 7-seed 量产 + 逐 owner + bootstrap CI；
4. 变体消融（paired）：交互方式（Token-Attn vs 拼接 vs Gated）、mask 必要性、特征组消融（哪些通道贡献 EAR）。

---
*本报告全部数字来自本机实测（2026-09-03，prefix_feats_v1，3-seed，LightGBM binary/multiclass）。A/B 差异显著性有限，选 A 综合统计方向 + 结构优势。原始 JSON 可复现。*
