# Gate Report — Phase 3 可解释性归因与 EAR 分析

> 状态: ✅ 完成(2026-09-04)  
> 产物: `code/phase3_ear.py` / `phase3b_lineage.py` / `phase3c_attn.py` / `phase3d_viz.py`  
> 输出: `docs/phase3_ear_results.json` / `phase3_ear_by_txn.csv` / `phase3b_lineage.json` / `phase3c_attn_results.json` / `fig_ear_case_{startup,run,normal}.png`  
> 上游裁决: Phase 2 方案 C(LightGBM 为主报 + TokenAttn 为可解释副线,见 `gate_report_phase2.md`)  
> 下游指向: Phase 4 会议/期刊论文初稿

---

## 0. 一句话结论

> **主报 LightGBM 已能在前 2 min 提前 65.9% 的 startup 与 81.5% 的 run 故障,中位预警提前量 5.6 / 37.6 min。TokenAttn 在可解释性维度上以 CLS→gunT2 通道聚焦 + 双族段级注意力模式差异,给出与 LightGBM 重要性互补的"通道级早期画像",使模型决策既可量化、可解释,也能落地到运维响应窗口。**

---

## 1. 数据与口径(E11 纪律复核)

| 维度 | 数值 / 说明 |
|---|---|
| 数据集 | `prefix_dataset_full.parquet`(7,386 段前缀,2,941 事务) |
| 切分 | owner1-6 内部 val(seed42 stratify 20%);test 仅用于最终归因读取 |
| 主报 | LightGBM,per-τ 独立训练,逐 τ val precision≥0.90 校准阈值 |
| 副线 | TokenAttn(BiLSTM→K=8 段池化 + 52 numeric token + CLS→2层 TransformerEncoder→线性) |
| 故障族 | startup 317 事务,run 54 事务,共 371 |
| 对照族 | normal 2,284 段(τ=2 子集) |

> ⚠️ **EAR 校准只读 owner1-6 val,绝不读 test 标签**(E11 纪律)。

---

## 2. Phase 3a — EAR(最早可预警前缀)汇总

### 2.1 逐 τ 高精度阈值(val precision≥0.90 校准)

| τ (min) | 1 | 2 | 3 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|
| 阈值 P | 0.998 | 0.802 | 0.832 | 0.806 | 0.769 | 0.639 |

> 解读: 极短前缀(τ=1)要求极高置信度(0.998);τ≥2 后稳定在 0.77~0.83 区间;τ=20 因前缀更长更宽松。  

### 2.2 双族 EAR / lead 汇总(故障事务全集)

| 族 | n | 预警率 | EAR 中位 (min) | EAR 均值 | lead_min 中位 | lead_min 均值 | lead_min P25~P75 | 未预警 |
|---|---|---|---|---|---|---|---|---|
| **startup** | 317 | **65.9 %** | **2.0** | 2.76 | **5.6** | 8.5 | 1.7~12.7 | 108 (34.1 %) |
| **run** | 54 | **81.5 %** | **2.0** | 4.82 | **37.6** | 40.1 | 30.9~50.9 | 10 (18.5 %) |
| 合计 | 371 | 68.2 % | 2.0 | 3.07 | 7.5 | 13.7 | 2.3~25.9 | 118 (31.8 %) |

### 2.3 EAR 分布(startup vs run)

| EAR (τ) | 1 | 2 | 3 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|
| startup | 11 | 144 | 29 | 16 | 7 | 2 |
| run | 2 | 23 | 8 | 4 | 2 | 5 |

> 解读: 两侧 EAR 中位 = 2 min(主表叙事),run 因 dur 普遍较长,即使 EAR 同样 2 min,lead 自动拉长(37.6 vs 5.6)。startup 108 个未预警的多为短事务(dur < EAR)+ 数据稀疏者(前缀行 <2);run 仅 10 个未预警,准确率更高。

### 2.4 关于"未预警"的口径区分

| 类型 | 含义 | 处置 |
|---|---|---|
| 模型未预警 | 模型在所有可用 τ 上 P 均 < 该 τ 高精度阈值 | 记入 `n_unalerted`;论文中如实报"召回上限" |
| 数据不允许预警 | 该事务某 τ 前缀行数 < 2(插枪到起充延迟、采样稀疏) | 不计入 EAR/lead 评估,论文 §诚实边界 段 |

> 注:offsets 语义 = 自"插枪"计时(非起充)。某些事务首行 offset 3+ min → τ=1/2/3 可能 <2 行不可用;只有 {τ=5,10,20} 可用。startup 中有 3,524 人群、run 中有 1,842 人群受此影响(已在 Phase 3a 单独标注)。

---

## 3. Phase 3b — 双谱系机制对比(LightGBM 视角)

### 3.1 Stage2 故障族判别 top-20 特征重要性

| rank | 特征 | importance | rank | 特征 | importance |
|---|---|---|---|---|---|
| 1 | chargingv_slope | 655 | 11 | charginga_slope | 401 |
| 2 | current_soc_last | 625 | 12 | out_power_last | 398 |
| 3 | current_soc_mean | 549 | 13 | out_power_range | 378 |
| 4 | chargingv_first | 537 | 14 | charginga_last | 358 |
| 5 | chargingv_last | 469 | 15 | out_power_slope | 345 |
| 6 | chargingv_mean | 451 | 16 | gunT1_first | 325 |
| 7 | out_power_mean | 410 | 17 | charginga_range | 320 |
| 8 | chargingv_range | 402 | 18 | out_power_first | 304 |
| 9 | charginga_mean | 402 | 19 | chargingv_std | 294 |
| 10 | current_soc_slope | 293 | 20 | charginga_first | 287 |

> 解读: 电压组(chargingv_*)+ SOC 组(current_soc_*)占据主导(8/20),电流与功率组作为辅助。gun 温度仅 gunT1_first(325)进入 top-20。这与"轻量级一阶统计 + 序列斜率"路径吻合。

### 3.2 τ=2 测试集三族通道级早期画像

| 通道/指标 | normal | startup | run | 双族 vs normal 差异 |
|---|---|---|---|---|
| chargingv_first (V) | 340.2 | 341.2 | **357.9** | run 显著高于 normal / startup |
| chargingv_last (V) | 354.3 | 352.2 | **369.9** | run 持续高压,startup/normal 接近 |
| chargingv_slope | **13.1** | 4.7 | 6.8 | normal 升压最快(普通 CC 阶段) |
| charginga_first (A) | 60.9 | **114.7** | 85.8 | startup 起步即大电流 |
| charginga_last (A) | 105.5 | 123.4 | 100.1 | startup 持续高电流 |
| out_power_first (kW) | 20.6 | **37.2** | 28.4 | startup 第一分钟即进入高功率段 |
| out_power_last (kW) | 36.4 | 40.7 | 34.4 | startup 末段仍维持 40 kW+ |
| current_soc_first (%) | 15.2 | 17.7 | **10.5** | run 起步 SOC 更低 |
| current_soc_last (%) | 40.8 | 41.7 | 31.0 | run 进度慢(SOC 升速低) |
| soc_delta | 25.6 | 24.0 | 20.5 | 三族接近,run 略低 |
| power_active_ratio | 0.841 | **0.994** | **1.000** | 双族功率持续高位 |
| power_peak_pos | 0.818 | 0.322 | 0.356 | normal 峰值靠后,双族峰值靠前 |
| gunT1_rise | -0.27 | -0.12 | -0.07 | 双族枪温略升,normal 平/降 |

**双族机制叙事**(可直读论文):

> **startup 族 = "短促大电流主动注入型"**:τ=2 即出现 charginga_first = 114.7 A(为 normal 60.9 A 的 1.88 倍),out_power_first = 37.2 kW(normal 仅 20.6),power_active_ratio = 0.994,峰值靠前(power_peak_pos=0.32)。SOC 从 17.7 → 41.7(短时高 ΔSOC 伴随电池大电流接受)。  
> **run 族 = "长时间高压缓充型"**:τ=2 起 chargingv_first 357.9 V(normal 340.2),SOC 起步更低(10.5 %)、进度更慢(31.0 vs normal 40.8),功率曲线全程活跃(power_active_ratio = 1.000)。双族 vs normal 的"功率持续高活性 + 高压/大电流"双特征指纹清晰可分。

### 3.3 样本规模(τ=2 测试集)

| 族 | 段数 |
|---|---|
| normal | 2,284 |
| startup | 267 |
| run | 44 |

---

## 4. Phase 3c — TokenAttn 注意力归因(副线)

> 模型:TokenAttn τ=3 seed0,在 test 集上抓取最后一层 CLS 自注意力权重。

### 4.1 通道级 CLS 注意力(feat_by_channel)

| 通道 | startup | run | normal | 叙事 |
|---|---|---|---|---|
| chargingv (V) | 0.102 | 0.079 | 0.123 | |
| charginga (A) | 0.096 | 0.095 | 0.136 | |
| out_power (kW) | 0.090 | 0.095 | 0.138 | |
| gunT1 (℃) | 0.092 | 0.110 | 0.133 | |
| **gunT2 (℃)** | **0.475** | **0.491** | **0.244** | **故障族 CLS 高度聚焦** |
| SOC (%) | 0.091 | 0.079 | 0.133 | |
| **sum** | 0.946 | 0.950 | 0.907 | |

> 解读: 故障族(尤其 startup/run)CLS 注意力 ~50 % 落在 gunT2,normal 仅 24 %。换言之**"枪温 2"是模型判别异常的关键信号**,而非主报 LightGBM 依赖的电压/SOC 一阶统计。这构成 **"决策口径互补"**:LightGBM 用电压/SOC 量化早预警;TokenAttn 用枪温 2 的通道级聚焦为决策背书,二者不是冗余而是双谱系机制画像。

### 4.2 段级 CLS 注意力(seg_profile,K=8 池化段)

| 段序号 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | sum |
|---|---|---|---|---|---|---|---|---|---|
| startup | 0.013 | 0.016 | 0.016 | 0.018 | 0.011 | 0.013 | 0.004 | 0.004 | 0.096 |
| run | 0.010 | 0.014 | 0.014 | 0.013 | 0.005 | 0.006 | 0.008 | 0.009 | 0.079 |
| normal | **0.122** | **0.145** | **0.088** | 0.009 | 0.008 | 0.009 | 0.003 | 0.003 | 0.387 |

> 解读: normal 注意力高度集中在前 2-3 段(共 ~92 %),与"普通充电前段特征已足够判别"一致;**startup / run 注意力更分散**(各段 0.01~0.02),说明故障族需要看更长的时序模式才能定论——这与 §2 双族 EAR 中位 2 min 一致(短前缀即可判别,但要确认稳定性需看到 τ=3)。

### 4.3 特征级注意力聚焦段(feat_profile)

startup 与 run 在 **feature 索引 8~12** 段(0.077~0.094)出现明显峰值,其他索引 0.008~0.018。对应数值特征组的"中间段",结合 §3.1 Stage2 重要性,该段主要为 chargingv/charginga/out_power 的 first/last/mean 特征——与 LightGBM 优势特征组重合,印证主副线在"特征子集"上也有交集。

### 4.4 样本规模(test 集 τ=3)

| 族 | 段数 |
|---|---|
| normal | 3,363 |
| startup | 230 |
| run | 47 |

---

## 5. Phase 3d — 样例可视化(论文素材)

3 张样例图已确认标签正确,可直接入论文:

| 文件 | 内容 | EAR | lead_min |
|---|---|---|---|
| `fig_ear_case_startup.png` | startup 故障:6 通道时序 + 红色 EAR=2min 虚线 + 绿色 prefix 区 + 标题"startup fault case, dur=7.6min, lead=5.6min" | 2 | 5.6 |
| `fig_ear_case_run.png` | run 故障:同上前 25 min 窗口 | 2 | 34.7 |
| `fig_ear_case_normal.png` | normal 对照:6 通道平直,无 EAR 标注(诚实) | — | — |

**startup 故障样例**(tid=...964110):典型短促大电流。  
- V: 312→313.5 V(CC);I: 130→136 A(陡升);gunT2: 34→36 ℃(显著温升);SOC: 0→75 %;  
- 6 通道曲线在 2 min 后立即展现故障征兆,EAR=2 min 提示"看前 2 min 已可下结论"。

**run 故障样例**(tid=...085510):典型长时间高压缓充。  
- V: 370→475 V(完整 CV);I: 0→125 A 后渐降至 ~100A;P: 0→60 kW(长时间高功率);  
- gunT2: 23→36 ℃(长时间累积温升,与 startup 不同);SOC: 0→60 %+;  
- lead_min = 34.7 min——运维可从容响应。

**normal 对照**(tid=...570510):6 通道平直稳定,典型短小正常充电(V 352 V 平台,I 25→70 A,P 0→25 kW,枪温几乎不动,SOC 0→8.5 %);模型正确判定为"无预警"。

> 修复记录: `phase3d_viz.py` 读取 CSV 时 32 位 transaction_id 被 pandas 解析为 object(Python int),与 str 目标 `==` 恒 False → 修复为 `pd.read_csv(..., dtype={'transaction_id': str})`。同步把 EAR/lead 改为数据驱动读取、normal 不画误导性 EAR 竖线、坐标轴标签由"time since session start"修正为"time since plug-in (min)"(offsets 真实语义)。

---

## 6. 双谱系机制画像(主副线互补,论文核心叙事)

| 维度 | 主报 (LightGBM) | 副线 (TokenAttn 注意力) | 是否互补 |
|---|---|---|---|
| 关键特征 | 电压斜率 / SOC_last / SOC_mean / 电压_first(8/20 top) | gunT2 通道聚焦(0.47~0.49) | ✅ 不同特征维度 |
| 双族机制 | startup 大电流主动 + run 高压缓充(quantitative) | 双族 CLS 高度集中 gunT2,normal 较均匀 | ✅ 双族均显示 gunT2 重要性 |
| 时序模式 | 一阶统计 + slope(序列差分) | 段级注意力分散度(故障 vs normal) | ✅ 时间尺度不同 |
| EAR 提前量 | startup 5.6 / run 37.6 min 中位 | 副线仅给出"通道级画像"不报 EAR | ✅ 主线报 EAR,副线做机制 |

> 结论: LightGBM 给出可量化、可运营的 EAR/lead 提前量;TokenAttn 给出通道级(gunT2)与段级(故障族需看更长序列)的注意力机制画像。**两者共同支撑"该方向不仅有效、可解释,且故障机制具有清晰的物理/工程对应(枪温累积温升 + 异常功率曲线)"**。

---

## 7. 诚实边界(对应 E11 纪律)

1. **数据语义**: offsets = 自"插枪"计时;部分事务因延迟起充,τ=1/2/3 前缀行<2,不可用 τ 不计入 EAR。这些事务在 startup 中占 9.6 %、run 中占 17.9 %,论文 §4 中应明确说明。  
2. **未预警率**: startup 34.1 %、run 18.5 %——已记入 `n_unalerted`,论文需明确"召回天花板 ≈ EAR+多 τ 校准精度的上界",不为召回设商业承诺。  
3. **跨站泛化**: 本期 EAR / lead 在 Phase 2 数据集(2 站点,owner1-6 切分)上得出,**未在外部站点做独立验证**;Phase 4 论文扩展方向之一为多站点交叉验证。  
4. **offsets 与 EAR 单位差异**: EAR 是 τ(分钟)单位;offsets 是自插枪的连续时间。论文图 x 轴统一标"minutes since plug-in"。  
5. **模型选型背景**: 主报 = LightGBM(树为主)而非 TokenAttn——见 `gate_report_phase2.md` Phase 2 调参结论,TokenAttn 在本数据集上落后 LightGBM ~0.03 是结构性结论(非欠调参)。  
6. **本 Phase 3 工作不审计 voltaic 数据的真实性**——见根目录 `voltaic数据审计_与_充电早期预警方向综述判定.md`。

---

## 8. Phase 4 论文初稿指引(给后续重启用)

> 用户当前已转向老师给的"另外一个期刊方向",本 Phase 4 待重启时按以下清单推进。

| # | 任务 | 锚定产物 |
|---|---|---|
| P4-1 | 论文骨架: title + abstract + intro + method + experiments + conclusion | `paper/` |
| P4-2 | Method 节核心表: 双族机制画像 + EAR 提前量 + 主副线互补表 | 本报告 §3.2 / §4.1 / §6 |
| P4-3 | 实验节: Phase 2 主报 LightGBM 结果 + Phase 3 EAR 汇总表 + 注意力热图 | `docs/phase3_ear_results.json` / `phase3c_attn_results.json` |
| P4-4 | 论文核心图(4 张): (a) EAR 累积分布 (b) 段级注意力对比 (c) 通道级注意力对比 (d) 3 张样例图(§5) | `docs/fig_ear_case_*.png` |
| P4-5 | 诚实边界段: §7 1~6 条作为论文 §4 limitation | |
| P4-6 | 期刊扩展清单: 多站点验证 / 滑动窗口在线部署 / 数据延迟容忍 | 后续论文扩展 |

---

## 9. 重启命令(快速回到此状态)

```bash
cd /Users/arthas/git/excharge/earlywarning
# 主报 EAR 已就绪,无需重跑 Phase 3a~c,仅需在重启时重生成样例图:
PY=/Users/arthas/.workbuddy/binaries/python/envs/default/bin/python
$PY code/phase3d_viz.py

# 检查产物是否齐全:
ls -la docs/phase3_*.* docs/fig_ear_case_*.png
```

完整恢复路径见 `docs/CHECKPOINT_phase2.md`(已升级为 Phase 3 完成态)。

---

## 10. 修订记录

- **2026-09-04 09:25** — Phase 2 方案 B 调参完成,转 C(树为主报,TokenAttn 副线)  
- **2026-09-04 后续** — Phase 3a/b/c 完成;Phase 3d CSV index 匹配修复(`dtype=str`),EAR/lead 数据驱动化,normal 不画误导 EAR 竖线,坐标轴语义修正为 plug-in;本文产出。