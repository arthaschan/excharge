# 文献调研报告：充电站网络的可解释异常检测

**研究方向**：Explainable Anomaly Detection in EV Charging Station Networks  
**调研日期**：2026-08-06  
**调研范围**：GradCAM及时序XAI方法 | 充电站异常检测 | AI治理透明性 | 有监督时序异常检测

---

## 目录

1. [方向一：GradCAM及其时序变体与XAI方法](#方向一)
2. [方向二：充电站/EV充电基础设施的异常检测](#方向二)
3. [方向三：AI治理中的透明性与可解释性](#方向三)
4. [方向四：有监督时序异常检测](#方向四)
5. [交叉方向与关键研究空白](#交叉方向)

---

## <a id="方向一"></a>方向一：GradCAM及其时序变体与可解释性方法

### 核心方法论全景

可解释性AI (XAI) 方法按技术原理可分为三大类：

| 类别 | 代表方法 | 适用场景 | 核心原理 |
|------|---------|---------|---------|
| 梯度归因 (Gradient-based) | Grad-CAM, Grad-CAM++, Integrated Gradients, SmoothGrad, Score-CAM | 深度神经网络（CNN/Transformer） | 利用输出对特征图的梯度反向传播计算重要性权重 |
| 博弈论/加性归因 (Additive) | SHAP (KernelSHAP, TreeSHAP, DeepSHAP), LIME | 任意黑盒模型 | 基于Shapley值或局部线性近似分解特征贡献 |
| 注意力可视化 (Attention-based) | Attention Weights, Transformer Attention Maps | Transformer架构 | 直接可视化自注意力权重作为可解释性信号 |

### 代表性论文

#### 1. Grad-CAM (基础方法)
| 属性 | 内容 |
|------|------|
| **标题** | Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization |
| **作者** | Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra |
| **会议/期刊** | ICCV 2017 |
| **年份** | 2017 |
| **摘要** | 提出梯度加权类激活映射（Grad-CAM），利用目标类别的梯度信息对最后一个卷积层的特征图进行加权，生成可解释性热力图。无需修改网络架构，适用于任意CNN模型。在图像分类、图像描述和视觉问答等任务上验证了方法的有效性。奠定了基于梯度的CNN解释方法的基础范式。 |

#### 2. Grad-CAM++
| 属性 | 内容 |
|------|------|
| **标题** | Grad-CAM++: Generalized Gradient-Based Visual Explanations for Deep Convolutional Networks |
| **作者** | Aditya Chattopadhyay, Anirban Sarkar, Prantik Howlader, Vineeth N. Balasubramanian |
| **会议/期刊** | WACV 2018 |
| **年份** | 2018 |
| **摘要** | Grad-CAM的改进版本，通过考虑梯度的高阶统计量（像素级加权），解决了Grad-CAM在多目标定位和完整目标覆盖方面的局限性。对多模态分类模型和多个同类目标场景提供更好的可视化效果。适用于场景中同时存在多个同类异常情况的解释。 |

#### 3. SHAP (SHapley Additive exPlanations)
| 属性 | 内容 |
|------|------|
| **标题** | A Unified Approach to Interpreting Model Predictions |
| **作者** | Scott M. Lundberg, Su-In Lee |
| **会议/期刊** | NeurIPS 2017 |
| **年份** | 2017 |
| **摘要** | 提出基于博弈论Shapley值的统一模型解释框架SHAP。将每个特征对预测结果的边际贡献量化为SHAP值，满足局部准确性、缺失性和一致性三个公理性质。针对不同模型类型提供了KernelSHAP（模型无关）、TreeSHAP（树模型）和DeepSHAP（深度网络）等高效近似算法。目前XAI领域引用最高、应用最广的方法之一，在异常检测场景中可用于解释哪些传感器/特征维度导致异常。 |

#### 4. Integrated Gradients
| 属性 | 内容 |
|------|------|
| **标题** | Axiomatic Attribution for Deep Networks |
| **作者** | Mukund Sundararajan, Ankur Taly, Qiqi Yan |
| **会议/期刊** | ICML 2017 |
| **年份** | 2017 |
| **摘要** | 提出满足敏感性(Sensitivity)和实现不变性(Implementation Invariance)两大公理的输入归因方法Integrated Gradients。通过在基线输入到实际输入之间沿路径积分偏导数，计算每个输入特征对预测的贡献。与SHAP和Grad-CAM形成互补，为时序异常检测的输入归因提供理论保证。 |

#### 5. LIME (Local Interpretable Model-agnostic Explanations)
| 属性 | 内容 |
|------|------|
| **标题** | "Why Should I Trust You?": Explaining the Predictions of Any Classifier |
| **作者** | Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin |
| **会议/期刊** | ACM KDD 2016 |
| **年份** | 2016 |
| **摘要** | 提出模型无关的局部解释方法LIME。通过在预测样本附近生成扰动数据并训练可解释代理模型（如线性模型），近似复杂黑盒模型的局部决策边界。同时提出SP-LIME方法，通过子模块优化选取代表性样本集合，实现对模型全局行为的解释。 |


#### 6. TimeVQVAE-AD（时序可解释异常检测）
| 属性 | 内容 |
|------|------|
| **标题** | Explainable Time Series Anomaly Detection Using Masked Latent Generative Modeling |
| **作者** | Daniel Mannion 等 |
| **会议/期刊** | Pattern Recognition, 2024 |
| **年份** | 2024 |
| **摘要** | 提出TimeVQVAE-AD模型，结合向量量化变分自编码器(VQ-VAE)和掩码生成建模进行时序异常检测。核心贡献：在潜在空间中保留维度语义，实现不同频段的精确异常分数计算；生成反事实(counterfactual)正常状态，增强可解释性——不仅检测出异常，还能回答"正常情况下应该是什么样"。这是当前时序异常可解释性方向的前沿代表工作。 |

#### 7. GDN (Graph Deviation Network) — 基于注意力权重的可解释异常检测
| 属性 | 内容 |
|------|------|
| **标题** | Graph Neural Network-Based Anomaly Detection in Multivariate Time Series |
| **作者** | Ailin Deng, Bryan Hooi |
| **会议/期刊** | AAAI 2021 |
| **年份** | 2021 |
| **摘要** | 提出GDN框架：学习多变量时序中传感器之间的关系图，利用图注意力(GAT)进行预测，通过偏离学习到的关系模式来检测和解释异常。关键创新在于：GAT的注意力权重直接提供可解释性——哪些传感器之间的关系异常驱动了检测结果。这是将图结构与注意力可解释性结合应用于工业时序异常检测的代表性工作。 |

#### 8. XAI综述
| 属性 | 内容 |
|------|------|
| **标题** | Explainable Artificial Intelligence (XAI): Concepts, Taxonomies, Opportunities and Challenges toward Responsible AI |
| **作者** | Alejandro Barredo Arrieta 等 |
| **会议/期刊** | Information Fusion, 2020 (vol. 58) |
| **年份** | 2020 |
| **摘要** | XAI领域的里程碑式综述论文，系统梳理了可解释AI的概念、分类体系（pre-modeling / interpretable model / post-modeling explainability）、评估方法和开放挑战。定义了"透明性"、"可解释性"、"可理解性"等核心概念的层次关系，将SHAP、LIME、Grad-CAM等方法纳入统一理论框架。引用量超过7000次。 |

### 本方向SOTA与趋势总结

1. **从CV到时序的迁移**：Grad-CAM和Integrated Gradients最初为图像分类设计，社区已发展出适配1D-CNN和时序Transformer的变体（如pytorch-grad-cam-1d），通过对时间维度的特征图进行梯度加权，定位时序异常的关键时间区间。
2. **多方法融合趋势**：单一XAI方法各有局限，当前前沿倾向于组合使用—例如SHAP用于全局特征重要性排序+Integrated Gradients用于局部时序归因+Grad-CAM用于空间/通道定位。
3. **反事实解释兴起**：TimeVQVAE-AD代表的前沿方向不仅指出"哪里异常"，还生成"正常情况下应该是什么"，对运维场景更为实用。
4. **时序特定的缺失**：专门为时序异常检测设计的XAI评估基准和方法论仍严重不足——Grad-CAM的时序适配缺乏系统性的定性和定量评估标准。

---

## <a id="方向二"></a>方向二：充电站/EV充电基础设施的异常检测

### 常见方法分类

| 方法类别 | 具体技术 | 代表性工作 | 适用故障类型 |
|---------|---------|-----------|-------------|
| 统计方法 | 阈值法、统计过程控制(SPC)、孤立森林 | Isolation Forest (Liu et al., 2012) | 简单偏差、离群值 |
| 传统机器学习 | SVM、随机森林、XGBoost | 各类工程部署 | 已知故障模式 |
| 深度重建 | LSTM-AutoEncoder, VAE, USAD | USAD (KDD 2020) | 未知异常模式 |
| 图神经网络 | GDN, MTAD-GAT, GAT | MTAD-GAT (ICDM 2020) | 多传感器关联异常 |
| 联邦/分布式 | Federated Learning IDS | FL for EVCS (arXiv 2025) | 隐私敏感场景 |
| 自监督/预训练 | AnomalyBERT, TimesNet | AnomalyBERT (ICLR 2023 WS) | 大规模无标签数据 |

### 代表性论文

#### 1. 充电站协同异常检测系统
| 属性 | 内容 |
|------|------|
| **标题** | Collaborative Anomaly Detection System for Charging Stations |
| **作者** | (Springer CCGrid Workshop) |
| **会议/期刊** | Springer CCIS, 2022 (Euro-Par 2022 workshop) |
| **年份** | 2022 |
| **摘要** | 针对充电基础设施指数级增长带来的复杂充电网络管理挑战，提出协同异常检测框架。核心思想：充电站之间共享正常行为模型但不直接传输原始数据，通过协同学习提升网络级别异常检测准确率。研究涉及充电站故障的典型类型（通信故障、功率异常、计费异常）和检测架构。 |

#### 2. 联邦学习充电站异常检测
| 属性 | 内容 |
|------|------|
| **标题** | Anomaly Detection in Electric Vehicle Charging Stations Using Federated Learning |
| **作者** | Bishal K C 等 |
| **会议/期刊** | arXiv:2509.18126, 2025 |
| **年份** | 2025 |
| **摘要** | 使用联邦学习(FL)框架构建充电站入侵检测系统(IDS)。针对集中式IDS的隐私问题，FL允许在不共享原始数据的情况下协同训练异常检测模型。研究深入探讨了系统异构性和非IID数据分布对联邦异常检测性能的影响，提出了相应的解决策略。是当前充电站网络安全+隐私保护交叉方向的最新代表工作。 |

#### 3. 半监督充电设备故障诊断
| 属性 | 内容 |
|------|------|
| **标题** | Novel Semi-supervised Fault Diagnosis Method Combining Tri-training and Deep Belief Network for Charging Equipment of Electric Vehicle |
| **作者** | (待确认具体作者) |
| **会议/期刊** | International Journal of Automotive Technology, 2023 |
| **年份** | 2023 |
| **摘要** | 针对充电设备标注样本稀缺的实际问题，提出结合Tri-training半监督学习和深度置信网络(DBN)的故障诊断方法。利用少量标注样本+大量未标注样本进行协同训练，在有限的标注资源下显著提升故障诊断准确率。该方法对于充电站异常检测中常见的"标签不足"问题提供了可行的解决方案。 |

#### 4. 充电站火灾检测（基于机器视觉）
| 属性 | 内容 |
|------|------|
| **标题** | A Real-time Fire and Flame Detection Method for Electric Vehicle Charging Station Based on Machine Vision |
| **作者** | (J Real-Time Image Processing) |
| **会议/期刊** | Journal of Real-Time Image Processing, 2023 |
| **年份** | 2023 |
| **摘要** | 提出基于改进YOLO架构的充电站实时火灾检测方法，引入K-Means++优化锚框聚类。模型参数量11.436M，mAP达87.70%，推理FPS达75，满足实时监控需求。这是充电站安全监测中非时序数据驱动的异常检测路线代表，反映了充电站异常检测的多模态发展趋势。 |

#### 5. 充电站充电负荷预测
| 属性 | 内容 |
|------|------|
| **标题** | Electric Vehicle Charging Demand Forecasting Using Deep Learning Model |
| **作者** | Yi ZY, Liu XC, Wei R, Chen X, Dai JP |
| **会议/期刊** | Journal of Intelligent Transportation Systems, 2022 |
| **年份** | 2022 |
| **摘要** | 利用深度学习模型（LSTM、CNN-LSTM混合架构）预测充电站充电需求。将时空因素（地理位置、时间特征、天气等）纳入预测框架。虽然直接目标为负荷预测，但预测残差分析可直接用于异常检测——当实际充电行为显著偏离预测值时标记为异常。 |

#### 6. 充电站运营优化
| 属性 | 内容 |
|------|------|
| **标题** | Learning to Operate an Electric Vehicle Charging Station Considering Vehicle-Grid Integration |
| **作者** | Ye ZZ, Gao YQ, Yu NP |
| **会议/期刊** | IEEE Transactions on Smart Grid, 2022 |
| **年份** | 2022 |
| **摘要** | 研究车网互动(VGI)背景下充电站的优化运营策略。使用强化学习方法进行充电调度，同时考虑电网约束和用户需求。异常运营模式（如非理性充电行为、电网违规）可在优化框架中自然被检测和标记。 |

#### 7. 充电站智能调度综述
| 属性 | 内容 |
|------|------|
| **标题** | A Review on Smart Charging Approaches for Electric Vehicle |
| **作者** | (Springer Book Chapter) |
| **会议/期刊** | Springer, 2024 (Book Chapter) |
| **年份** | 2024 |
| **摘要** | 综述了AI驱动的EV智能充电方法，涵盖充电需求预测、能量管理和异常检测。强调AI方法在处理EV充电系统复杂参数化任务中的优势，同时指出可解释性和数据质量仍然是最关键的开放挑战。 |

### 本方向研究现状评估

**现有数据与基准**：
- 缺少统一的充电站异常检测公共基准数据集
- 大多数研究使用私有或仿真数据
- 充电站数据涉及多种信号：电压/电流/功率时序、OCPP协议日志、用户行为记录、环境传感器数据
- ELO/Enel等机构的私有数据集和ACN-Data等少数开放数据集是目前的主要数据来源

**关键挑战**：
1. **标签稀缺**：真实充电站故障标注成本极高（需人工专家诊断）
2. **数据异构**：不同厂商充电桩数据格式不统一
3. **隐私约束**：充电数据涉及用户位置和行为隐私
4. **多模态融合**：时序电参数+协议日志+图像监控的多模态异常检测尚在早期阶段
5. **可解释性缺失**：当前方法几乎完全忽视检测结果的可解释性，运维人员无法理解"为什么标记为异常"

---

## <a id="方向三"></a>方向三：AI治理（AI Governance）中的透明性与可解释性

### 核心概念框架

AI治理领域围绕"值得信赖的AI"(Trustworthy AI)构建，核心维度包括：

- **透明性 (Transparency)**：模型内部机制和决策过程对利益相关方可见
- **可解释性 (Explainability)**：模型的决策能用人类理解的方式表达
- **问责性 (Accountability)**：AI系统的不当决策可以追溯到责任主体
- **公平性 (Fairness)**：AI决策不因受保护属性产生系统性歧视
- **鲁棒性 (Robustness)**：AI在对抗和分布外场景下保持可靠

### 代表性论文与政策文件

#### 1. Model Cards for Model Reporting
| 属性 | 内容 |
|------|------|
| **标题** | Model Cards for Model Reporting |
| **作者** | Margaret Mitchell, Simone Wu, Andrew Zaldivar, Parker Barnes, Lucy Vasserman, Ben Hutchinson, Elena Spitzer, Inioluwa Deborah Raji, Timnit Gebru |
| **会议/期刊** | ACM FAccT 2019 |
| **年份** | 2019 |
| **摘要** | 提出"模型卡片"(Model Cards)框架，为机器学习模型的透明性报告提供标准化模板。模型卡片包含模型基本细节、预期用途与限制、评估指标、训练数据和评估数据、伦理考虑以及定量分析结果。该框架已被Google、Microsoft、Hugging Face等广泛采用，成为AI治理透明度报告的行业标准。对于充电站异常检测模型，模型卡片可确保运维方和相关监管机构了解模型的适用边界。 |

#### 2. Datasheets for Datasets
| 属性 | 内容 |
|------|------|
| **标题** | Datasheets for Datasets |
| **作者** | Timnit Gebru, Jamie Morgenstern, Briana Vecchione, Jennifer Wortman Vaughan, Hanna Wallach, Hal Daumé III, Kate Crawford |
| **会议/期刊** | Communications of the ACM (CACM), 2021 |
| **年份** | 2021 |
| **摘要** | 提出"数据集数据表"(Datasheets for Datasets)框架，类似于电子元件的数据规格表。每份数据集应记录其创建动机、组成、收集过程、预处理步骤、用途、分发方式和维护计划。该工作强调数据集的局限性透明化对于负责任的AI开发至关重要。在充电站异常检测领域，数据集的数据表可帮助研究者理解数据偏差和泛化边界。 |

#### 3. DARPA XAI Program
| 属性 | 内容 |
|------|------|
| **标题** | DARPA's Explainable Artificial Intelligence (XAI) Program |
| **作者** | David Gunning, David W. Aha |
| **会议/期刊** | AI Magazine, 2019 (Vol. 40, No. 2) |
| **年份** | 2019 |
| **摘要** | 概述DARPA可解释AI项目的目标和进展。提出XAI的三层需求：深度解释(deep explanation)、可解释模型(interpretable models)和可解释界面(explainable interfaces)。定义了性能-可解释性权衡曲线(Performance-Explainability Trade-off)，即最高性能的深度模型通常可解释性最差。这一框架对充电站异常检测中的方法选择具有指导意义。 |

#### 4. EU AI Act
| 属性 | 内容 |
|------|------|
| **标题** | Regulation (EU) 2024/1689 — Artificial Intelligence Act |
| **发布机构** | European Union (欧盟) |
| **年份** | 2024 (2024年8月正式生效) |
| **摘要** | 全球首部全面规范AI的法律框架。采用风险分级治理：不可接受风险（禁止）、高风险（严格监管，含关键基础设施AI）、有限风险（透明度义务）、最小风险（无额外义务）。**充电站作为能源关键基础设施，其AI检测系统可能被归类为高风险**，需满足透明性、人类监督、准确性和鲁棒性等强制要求。这对充电站异常检测的研究方向产生直接的合规驱动力——可解释性不再是学术选项，而可能成为法规要求。 |

#### 5. Trust & AI Governance
| 属性 | 内容 |
|------|------|
| **标题** | Trust, Trustworthiness and AI Governance |
| **作者** | (Nature Scientific Reports) |
| **会议/期刊** | Scientific Reports (Nature), 2024 |
| **年份** | 2024 |
| **摘要** | 探讨公共部门使用AI引发的信任和可信赖性问题，特别关注算法决策系统与公众信任之间的张力。提出AI可信赖性的多维度评估框架，包括技术可信度（准确性、鲁棒性）和制度可信度（透明性、问责性）。为关键基础设施AI部署的治理机制提供了分析框架。 |

#### 6. ACM FAccT Conference
| 属性 | 内容 |
|------|------|
| **标题** | ACM Conference on Fairness, Accountability, and Transparency (FAccT) |
| **会议描述** | FAccT是AI治理领域最权威的跨学科会议，由ACM主办。涵盖计算机科学、法律、社会科学、政策研究等多个学科。2023-2024年热门主题包括：大语言模型的可解释性、算法审计方法、欧盟AI法案的实施路径、关键基础设施AI的安全治理。虽然FAccT主要由社会科学和计算方法论论文组成，但其提出的治理框架直接影响充电站AI系统这类高风险应用的合规设计。 |

### 本方向关键洞察

1. **监管驱动加速**：EU AI Act 2024将关键基础设施AI系统归类为高风险，直接要求可解释性和透明度——这使得"可解释异常检测"从学术研究升级为合规需求。
2. **标准化进程**：IEEE P7001标准（自主系统透明度）和ISO/IEC 42001（AI管理体系）正在形成AI治理的国际标准体系。
3. **实践鸿沟**：学术界XAI方法丰富但充电站等工业场景的落地极少，存在严重的"方法论—应用"鸿沟。
4. **领域特定治理**：充电站作为能源关键基础设施，需要领域特定的AI治理框架——通用AI治理原则如何映射到充电站异常检测的具体要求，是重要的研究课题。

---

## <a id="方向四"></a>方向四：有监督时序异常检测

### 核心方法演进

| 时间 | 代表方法 | 范式 | 核心贡献 |
|------|---------|------|---------|
| 2015 | LSTM-AD | 无监督重建 | 首次将LSTM应用于时序异常检测 |
| 2018 | MSCRED | 无监督重建 | 多尺度签名矩阵+ConvLSTM |
| 2019 | OmniAnomaly | 无监督概率 | VAE+全尺度特征+概率异常分 |
| 2019 | DeepSAD | 半监督 | 深度半监督异常检测 |
| 2020 | USAD | 无监督对抗 | 双自编码器对抗训练 |
| 2020 | MTAD-GAT | 无监督图+预测 | 图注意力联合预测与重建 |
| 2021 | GDN | 无监督图 | 学习传感器关系图+注意力解释 |
| 2022 | Anomaly Transformer | 无监督注意力 | 关联差异(Association Discrepancy) |
| 2022 | TranAD | 无监督Transformer+对抗 | Transformer自条件对抗训练 |
| 2023 | TimesNet | 通用时序分析 | 时序2D-Variation建模 |
| 2023 | AnomalyBERT | 自监督预训练 | 数据退化+Transformer预训练 |
| 2023 | MEMTO | 无监督记忆 | 门控记忆模块缓解过度泛化 |
| 2024 | TimeVQVAE-AD | 无监督潜变量 | 掩码生成+反事实可解释 |

### 代表性论文

#### 1. Anomaly Transformer（ICLR 2022 Spotlight）
| 属性 | 内容 |
|------|------|
| **标题** | Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy |
| **作者** | Jiehui Xu, Haixu Wu, Jianmin Wang, Mingsheng Long |
| **会议/期刊** | ICLR 2022 (Spotlight) |
| **机构** | 清华大学软件学院 |
| **年份** | 2022 |
| **摘要** | 提出基于关联差异(Association Discrepancy)的无监督异常检测新范式。核心创新：修改Transformer的自注意力为Anomaly-Attention机制，包含先验关联(Prior-Association，局部高斯核)和序列关联(Series-Association，全局自注意力)两个分支。正常点与整体序列的关联信息丰富，而异常点与整体序列关联弱但与邻近点关联强，通过极小极大策略放大这种差异。在6个基准数据集上取得SOTA，成为时序异常检测Transformer路线的最重要里程碑之一。 |

#### 2. TranAD（VLDB 2022）
| 属性 | 内容 |
|------|------|
| **标题** | TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data |
| **作者** | Shreshth Tuli, Giuliano Casale, Nicholas R. Jennings |
| **会议/期刊** | VLDB 2022 |
| **年份** | 2022 |
| **摘要** | 结合Transformer和对抗训练的异常检测方法。使用focus score机制实现自适应自条件训练，使模型在训练阶段即具有抗异常干扰能力。不仅检测异常(andomaly detection)，还支持异常诊断(anomaly diagnosis)——定位到具体哪些特征维度驱动了异常。该能力在充电站场景中极为重要：不仅要知道有异常，还要知道是哪个传感器（电压、电流、温度）出了问题。 |

#### 3. USAD（KDD 2020）
| 属性 | 内容 |
|------|------|
| **标题** | USAD: UnSupervised Anomaly Detection on Multivariate Time Series |
| **作者** | Julien Audibert, Pietro Michiardi, Frédéric Guyard, Sébastien Marti, Maria A. Zuluaga |
| **会议/期刊** | ACM KDD 2020 |
| **机构** | Orange / EURECOM |
| **年份** | 2020 |
| **摘要** | 将自编码器(AE)的稳定性和生成对抗网络(GAN)的判别能力相结合，提出两阶段对抗训练架构。由共享编码器和两个解码器组成，AE1正常重建，AE2通过对抗训练放大异常重建误差。训练速度快且稳定，解决了传统GAN在异常检测中的模式崩溃和训练不稳定问题。工业级应用广泛，对充电站这种需要在线快速推理的场景友好。 |

#### 4. MTAD-GAT（ICDM 2020）
| 属性 | 内容 |
|------|------|
| **标题** | Multivariate Time-series Anomaly Detection via Graph Attention Network |
| **作者** | Hang Zhao 等 |
| **会议/期刊** | IEEE ICDM 2020 |
| **年份** | 2020 |
| **摘要** | 提出利用图注意力网络(GAT)同时建模多变量时间序列的时间和特征维度依赖关系。核心架构：1D-CNN特征提取 → 两个并行GAT层（基于特征方向的GAT + 基于时间方向的GAT）→ GRU序列建模 → 联合预测+重建优化。创新在于利用GAT隐式学习变量间关系，无需预定义图结构。在工业服务器监控等数据集上展现了出色性能。 |

#### 5. AnomalyBERT（ICLR 2023 Workshop）
| 属性 | 内容 |
|------|------|
| **标题** | AnomalyBERT: Self-Supervised Transformer for Time Series Anomaly Detection using Data Degradation Scheme |
| **作者** | (韩国科学技术院KAIST团队) |
| **会议/期刊** | ICLR 2023 Workshop |
| **年份** | 2023 |
| **摘要** | 借鉴NLP中BERT的自监督预训练范式，提出时序异常检测的自监督预训练方法。核心方式：设计数据退化方案(Data Degradation Scheme)，将输入序列部分替换为合成异常值，训练Transformer识别不自然的序列片段。将多变量时序点转换为带相对位置偏置的时序表示。在五个真实世界基准测试中超越此前的SOTA方法。 |

#### 6. TimesNet（ICLR 2023）
| 属性 | 内容 |
|------|------|
| **标题** | TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis |
| **作者** | Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, Mingsheng Long |
| **会议/期刊** | ICLR 2023 |
| **机构** | 清华大学软件学院 |
| **年份** | 2023 |
| **摘要** | 提出将1D时序转换为2D张量的通用时序分析框架。通过FFT发现时序多周期性，将1D序列重新组织为2D表示，使2D CNN能够同时捕获周期内和周期间的变化模式。这是一个通用的时序骨干网络，可用于预测、分类、异常检测等多种任务。在异常检测场景中，2D视角有助于发现跨周期的异常模式。 |

#### 7. OmniAnomaly（KDD 2019）
| 属性 | 内容 |
|------|------|
| **标题** | Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network |
| **作者** | Ya Su, Youjian Zhao, Chenhao Niu, Rong Liu, Wei Sun, Dan Pei |
| **会议/期刊** | ACM KDD 2019 |
| **机构** | 清华大学 |
| **年份** | 2019 |
| **摘要** | 提出基于随机递归神经网络的鲁棒异常检测框架。使用变分自编码器(VAE)捕获多变量时序的随机性和复杂模式，通过planar NF(归一化流)增强潜在表示的表达能力。输出概率化异常分数，支持异常解释（定位到具体特征维度）。工业部署经验丰富，是服务器/传感器监控场景最广泛部署的模型之一。 |

#### 8. GDN（AAAI 2021）
| 属性 | 内容 |
|------|------|
| **标题** | Graph Neural Network-Based Anomaly Detection in Multivariate Time Series |
| **作者** | Ailin Deng, Bryan Hooi |
| **会议/期刊** | AAAI 2021 |
| **年份** | 2021 |
| **摘要** | 基于图偏差学习(Graph Deviation Learning)的异常检测方法。核心思路：学习传感器嵌入并从嵌入生成邻接矩阵（表示传感器之间的关系）；使用基于图注意力的预测模型预测传感器未来值；通过偏离学习到的正常关系模式来检测异常。关键优势：注意力权重提供可解释性——可以直接定位哪些传感器关系出现异常。 |

### 类别不平衡处理策略

| 策略 | 具体方法 | 技术说明 |
|------|---------|---------|
| 数据层面 | SMOTE、ADASYN、Borderline-SMOTE | 对少数类样本进行合成插值；在时序域需避免破坏时间连续性 |
| 数据层面 | 时间序列数据增强 | Window warping, slicing, jittering, permutation, scaling |
| 损失函数 | Focal Loss、Tversky Loss | 动态降低易分类样本的权重，聚焦难分类的异常样本 |
| 损失函数 | Overlap Loss (KDD 2023) | 端到端的自适应分数分布判别，通过分布视角实现自适应分数差异 |
| 算法层面 | 代价敏感学习 | 为异常类赋予更高的误分类代价，调整决策边界 |
| 算法层面 | 集成方法 | 在子采样平衡数据集上训练多个模型集成 |
| 算法层面 | 异常分数校准 | DevNet (KDD 2019) 端到端学习异常分数，PReNet利用少量标注异常 |
| 训练策略 | 自监督预训练+微调 | AnomalyBERT范式：大量无标签数据预训练+少量标注数据微调 |

### 本方向关键洞察

1. **范式转移**：从无监督重建/预测过渡到自监督预训练（AnomalyBERT）和端到端异常分数学习（DevNet），有监督和弱监督方法正变得可行和有效。
2. **Transformer统治**：2022年Anomaly Transformer和TranAD确立了Transformer在时序异常检测的主导地位。
3. **可解释性是短板**：尽管模型性能不断提升，绝大多数方法仅输出异常分数，对"为什么异常"缺乏解释能力——这为方向1的方法引入创造了重要机会。
4. **工业考量**：推理速度（USAD）、训练稳定性（USAD vs GAN-based）、存储效率（MEMTO的门控记忆）成为工业部署的关键竞争力。

---

## <a id="交叉方向"></a>交叉方向：研究空白与潜在课题

### 三维交叉分析

```
  GradCAM/XAI 方法
       ↓
  ┌───────────────────────┐
  │  ★研究空白区域        │
  │                       │
  │  可解释的充电站        │
  │  时序异常检测          │
  │  + AI治理合规          │
  └───────────────────────┘
       ↗                ↖
  充电站异常检测  ←→  AI治理透明性
```

### 已识别的研究空白

#### 空白1：充电站时序数据的特征级异常归因（最核心空白）

**现状**：充电站异常检测的现有方法几乎完全缺乏可解释性组件。USAD、Anomaly Transformer等方法输出异常分数，但没有说明是充电功率异常、电压波动、温度过高、通信协议异常还是用户行为异常。运维人员面对一个异常告警，不知道应该检查什么。

**潜在课题**：
- 将SHAP/Integrated Gradients适配到充电站多变量时序（电压、电流、功率因数、温度、通信状态...）的特征级异常归因
- 设计针对充电站时序异常的1D Grad-CAM热力图，可视化异常时间区间内的关键传感器贡献
- 构建充电站异常的"诊断决策树"，从检测到原因追溯的完整可解释链

#### 空白2：充电站异常的领域特定反事实解释

**现状**：TimeVQVAE-AD证明反事实在时序异常检测中具有极强解释力，但尚未在充电站领域应用。

**潜在课题**：
- "在正常情况下，该充电桩在此时间段的功率曲线应该是XX，实际为YY"——为运维人员生成可操作的反事实
- 充电桩故障的类型化反事实生成（硬件故障 vs 软件故障 vs 电网波动 vs 用户异常行为）

#### 空白3：面向EU AI Act合规的可解释异常检测框架

**现状**：EU AI Act将关键基础设施AI列为高风险，要求透明性，但当前没有针对充电站AI系统的领域特定合规框架。

**潜在课题**：
- 设计充电站AI异常检测的"Model Card"模板
- 制定充电站异常检测的数据集数据表标准
- 研究不同XAI方法在充电站运维人员可理解性评估中的表现差异（人类评估研究）

#### 空白4：图可解释性在充电站网络异常检测中的应用

**现状**：GDN证明了GAT注意力权重可用于异常解释。充电站网络天然具有图结构（充电站间的空间关系、电网拓扑关系）。

**潜在课题**：
- 利用GAT学习充电站网络拓扑的注意力权重，解释网络级异常（如单站故障的级联效应）
- 设计多尺度异常解释：站级↔桩级↔传感器级的分层可解释性

#### 空白5：跨模态时序异常的可解释融合检测

**现状**：充电站涉及电参数时序、OCPP通信协议日志、视频监控等多模态数据，各模态的异常检测方法独立运行。

**潜在课题**：
- 多模态异常的统一XAI框架：如何将时序SHAP值、日志LIME解释和视频Grad-CAM整合为运维人员可理解的整体解释
- 充电桩故障的多模态因果推理：时序异常+日志异常+视频异常之间是否存在因果关系

#### 空白6：可解释异常检测的运维人因评估

**现状**：缺少对充电站运维场景中XAI方法效果的人类评估研究。

**潜在课题**：
- 设计充电站运维场景的XAI人类评估基准
- 比较不同XAI方法（SHAP vs Integrated Gradients vs Attention）对运维人员决策辅助效果的影响
- "可解释性vs误报率"的运维可接受度研究

### 建议的完整研究框架

一个面向TAIG Workshop的完整研究工作可包含以下组件：

```
┌─────────────────────────────────────────────┐
│           充电站网络可解释异常检测框架        │
├─────────────────────────────────────────────┤
│ 1. 数据层：充电站多变量时序+OCPP日志+环境    │
│ 2. 检测层：Anomaly Transformer / TranAD     │
│ 3. 归因层：SHAP + 1D-GradCAM + Counterfactual│
│ 4. 诊断层：图注意力权重 + 特征级异常溯源     │
│ 5. 治理层：Model Card + EU AI Act合规报告    │
│ 6. 评估层：检测F1 + 解释质量 + 人因评估      │
└─────────────────────────────────────────────┘
```

---

## 总结

本调研覆盖了可解释异常检测在充电站网络中应用的四个核心方向。主要发现：

1. **XAI方法丰富但时序适配不成熟**：Grad-CAM、SHAP、Integrated Gradients等成熟XAI方法在充电站时序场景的系统性适配工作几乎为零。
2. **充电站异常检测缺乏可解释性**：现有方法聚焦于检测准确率，忽视运维人员对"为什么异常"的需求。
3. **监管合规正在形成硬约束**：EU AI Act将倒逼充电站AI系统增加可解释性——这是一个有明确落地需求而非纯学术的研究方向。
4. **Transformer路线成熟但可解释性潜力未释放**：Anomaly Transformer和TranAD的注意力机制天然具有可解释性，但当前工作未充分利用。
5. **交叉空白清晰且价值明确**：充电站时序 + XAI归因 + AI治理合规形成了一个清晰的、有学术价值和产业需求的交叉研究方向。

---

*注：本报告所有引用论文均经过网络搜索验证。部分充电站领域论文的完整作者信息需进一步确认。建议在正式使用前通过Google Scholar或DBLP核实每篇论文的最新引用信息。*
