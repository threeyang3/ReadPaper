# SDFR 后续研究路线：仿真基线、研究切入点与消融实验（2026-11-19）

> **文档性质**：研究选题建议与尚未实施的实验计划；不是原始 SDFR 论文已经完成的实验或结论。依据作者公开代码及 2026-11 核对的 Isaac Sim 官方文档，并参考 RNNPose、UA-Pose、SDFit 等相关工作。**当前尚未搭建仿真、训练模型或实际测量性能。**

## 总体建议

先用作者 PyTorch3D 流程复现 **SDFR / ICP / FRICP / SymmICP** 同条件基线，再构建 Isaac Sim 可控 RGB-D 数据集，最后推进 **深度质量与可见性感知的鲁棒 SDF 位姿优化**。切忌直接把“Huber 替换 L1”当作足够新颖的论文贡献；应在深度/掩码不确定性建模、可见性、可辨识性、收敛稳健性和多场景泛化上构造实质性问题与贡献。

## 1. 从 SDFR 源码看到的可改进处

- [sdfr.py](https://github.com/pangbo1997/SDFR/blob/master/sdfr.py) 的 `SDFR_adam` 以 `torch.abs(sdf_value_scene).mean()` 为核心损失，点间默认同权。
- 旋转用 `so3_log_map` / `so3_exp_map`，平移用 `trg_pts.mean(dim=1)` 初始化；Adam 分别优化，固定 100 次循环，无基于收敛的早停。
- 可选的最远点采样是 150 点；噪声模式下 `run_sdfr.py` 关闭 FPS。应评估极少有效点、混入背景点时的取样偏差。
- 数据生成 `prepare_dataset.py` 使用渲染深度、受控随机位姿误差；`corrupt` 以 RandomErasing、法线视角删点模拟缺失，非光学反射传感器物理模拟。
- 重要工程检查：`run_sdfr.py` CLI 提供 `diverse/corrupt` 选项，但其 `eval_SDFR` 遍历条件目前仅显式涵盖 `standard/noise/scale`；扩展测试前须补上并写回归测试。
- 数据生成脚本的 `--root-dir` 当前有与局部硬编码变量不同步的风险；检查并消除。检查 CAD 模型以 mm 导入、渲染按 m 输出、在线模型点乘 `obj_scale*1000`，以及 `T_{wc}` 与 `T_{cw}`。

## 2. 候选课题及排序

| 方向 | 科学问题 | 最小验证版本 | 难度 | 预期贡献边界 |
|---|---|---|---|---|
| A. 深度和可见性感知的鲁棒 SDF | 点云包含错分割、遮挡与深度缺陷时，如何估计每个测量的可靠性？ | L1→Huber/Cauchy，后加边界/深度/可见性权重 | 中 | 单纯替换损失函数通常不足成论文，需可校准不确定性与实证 |
| B. RGB 轮廓+SDF 融合 | 深度缺失时是否能够继续使用影像轮廓来约束位姿？ | CAD 渲染轮廓 IoU、SDF 残差加权 | 中高 | 已有 render-compare 工作；应形成有区别的困难场景 |
| C. 多视角一致性 | 多个部分视图是否能提升可观测性，如何处理不同相机噪声？ | 已知 `T_{wc}` 合并目标点云/联合优化 | 中 | 堆叠点云属于基础对照，贡献需更深入 |
| D. 低延迟自适应求解器 | 100 步 Adam 是否可以在保持成功率时明显减负？ | early stop / coarse-to-fine / adaptive sampling | 低中 | 容易获得工程收益，但论文新颖性有限 |
| E. 主动视角选择 | 如何基于位姿歧义选择下一位置，兼顾相机移动代价？ | Jacobian 信息量、遮挡可见性与机器人可达性 | 高 | 有关文献已有 NBV，需明确新颖点 |

## 3. 推荐主线 A：鲁棒可见性/不确定性感知 SDF

原始（示意）：
```math
L_0(T) = \frac1N\sum_i | f_{\theta}(T^{-1} p_i) |.
```

初步方案：
```math
L_{robust}(T)= \frac{\sum_i w_i\,\rho(f_{\theta}(T^{-1}p_i))}{\sum_i w_i+\epsilon}.
```

- `rho` 可从 Huber、Cauchy、Tukey 与截断距离中选择；始终对照原始 L1，避免声称其缺乏基本鲁棒性。
- `w_i` 候选来源：深度噪声模型 `sigma_z(z,\theta_i)`、像素到 mask 边界距离、法线入射角、几何一致性、多视角重投影一致性、可见性测试；逐一验证，不把 GT 可靠性信息用在实际算法输入。
- **模型不确定性**与**观测不确定性**不同。前者来自神经 SDF 拟合近似误差，后者来自传感器/分割。科学贡献可考虑联合刻画二者，并验证输出置信度是否校准。
- 显式可见性/深度自由空间约束需要小心：被其他物体遮挡的射线不能简单当作目标空表面约束。
- 目标是假设在固定模型 CAD、相同帧数和优化预算下，困难深度与错分割条件下的 P95 位姿误差、失败率下降，而干净场景基本不恶化。

## 4. 最小实验（先 PyTorch3D，再 Isaac Sim）

### Stage 0 — 可信原始复现

1. 单个/少数 BOP CAD，官方标准数据生成；记录文件与依赖版本、随机数种子、尺度约定。
2. 复现原始 `SDFR_adam` 与点到点 ICP、点到面 ICP、FRICP、SymmICP；锁定同一批 point cloud、初值、算力设备。
3. 不只统计总体成功率，还记录初始/最终旋转误差、平移误差、ADD(-S)、SDF loss、迭代耗时、成功率按初始化误差分桶。
4. 用可视化覆盖 SDF 预测的 `f=0` 表面和观测点云，验证相机/CAD 坐标、单位和优化方向正确。
5. 确认离线 SDF 几何误差低于在线位姿误差，否则归因错误。

### Stage 1 — PyTorch3D 的消融验证

同一场景、随机种子、初始位姿下比较：

- B0 = 原版 L1；
- B1 = 换 Huber/Cauchy/Tukey（选择鲁棒函数）；
- B2 = B1 + 深度/遮挡相关的观测权重；
- B3 = B2 + 可见性过滤（非真值 mask 泄露）；
- B4 = B3 + 迭代或采样优化。

建议先做 **单变量扫描而非指数级全因子排列**：

| 控制变量 | 建议验证值（新实验建议，非原文） |
|---|---|
| 初始旋转误差 | 0–5°, 5–15°, 15–30°, 30–60°, 60–90° |
| 初始平移误差 | 0–1cm, 1–3cm, 3–5cm, 5–10cm |
| 可见点比例 | 100%, 80%, 60%, 40%, 20% |
| 离群点比例 | 0%, 10%, 30%, 50% |
| 有效采样点数 | 64 / 150 / 512 / 2048 |
| SDF 模型质量 | 不同训练步数 / SDF 重建误差 |
| 数据随机性 | 多种随机种子；独立样本与置信区间 |

做选定的二维交互实验，如“可见率×离群率”、“初始旋转×深度质量”，不需穷举全矩阵。

### Stage 2 — Isaac Sim 6.2 RGB-D 测试台

**首先确认运行硬件**。截至 2026-11 的 [Isaac Sim 6.2.0 官方要求](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/requirements.html)：Windows 11 / Ubuntu 22.04/24.04，至少 32GB RAM，表中最小列为 RTX 4080、16GB VRAM；Apple Silicon 不可本机运行，H100/A100 无 RT Cores 不能作为 Isaac Sim RTX GPU。未知 NVIDIA RTX 显卡型号前，不建议先安装大型仿真包。

在兼容 RTX 工作站上使用 [Replicator](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/replicator_tutorials/tutorial_replicator_object_based_sdg.html) 建立场景：

- USD/CAD 已知刚体 + 台面 + 一两个遮挡物；
- 单固定相机开始，再增加腕部相机和 Panda 夹爪；
- 导出对齐的 RGB、`distance_to_image_plane` 类型深度、object instance mask、K、`T_{wc}`、`T_{wo}^{GT}`、object ID、随机种子与噪声配置；
- 使用 `distance_to_image_plane` 为基于 `z` 的 pinhole 反投影输入；不要把 `distance_to_camera` 射线距离直接当同一深度值。
- 调整材料、光照、背景、遮挡比例、深度缺失和分割质量；**传感器噪声要区分物理仿真机制与手工注入模型**。
- 算法环境与 Isaac Sim 独立，通过 `.npz/.h5` 数据契约联动：避免强行把原作者 Python3.8+旧依赖与 Isaac Python/RTX 环境整合在同一虚拟环境。

建议数据结构：
```
datasets/isaac_sdfr/
  scene_000001/
    rgb.png
    depth_z.npy
    mask_instance.png
    sample.npz    # K, T_wc, T_wo_gt, object_id, seed, config
```

训练/测试应按 **物体、场景布局与视角** 设定明确的划分，检查同一 CAD 的 SDF 训练不构成位姿标签泄漏。若评测特定已知物体 refinement，可允许训练 SDF 使用该物体 CAD，但不允许使用测试点云/GT 配准补充优化器信息。

### Stage 3 — 机器人与 Sim-to-Real

只有在静态场景中稳定复现后，再增加 Panda、抓取遮挡和运动。考察腕相机/固定相机、主动感知、真实 D435 的有效深度与分割质量，最终看 **毫米级位姿误差、抓取成功率、失败恢复时间、端到端延迟**。基础优化器在生成点云阶段不应依赖仿真 GT mask。

## 5. 量化指标与论文式对照

- 位姿：对称性感知 ADD(-S)、5cm/5deg、1cm/5deg、5mm/5deg；报告旋转/平移误差分位数（中位数、P90、P95）和总体失败率。
- 鲁棒性：每个污染强度下的成功率曲线及 AUC；清洁环境的性能保持。
- 性能：SDF 网络加载/训练、点云预处理、每次在线优化 P50/P95、GPU 内存、FPS；相同 GPU 计时并先 warm-up。
- 不确定性：若输出测量置信度，报告可靠性曲线、ECE 或 NLL，避免只以视觉效果说明“已建模”。
- 受控变量：同初值、相同 CAD、相同数据、相同随机种子、相同优化预算；反事实/Oracle mask 仅作上限参考，不参加实际算法输入。

## 6. 相关研究（避免重复发明）

- [SDFR 原版](https://github.com/pangbo1997/SDFR)：学习已知 CAD SDF，按绝对 SDF 残差优化刚体位姿。
- [RNNPose](https://pubmed.ncbi.nlm.nih.gov/38289852/)：多轮对应估计与鲁棒位姿优化、下调错误对应权重，说明“鲁棒权重”不是独创概念。
- [UA-Pose（CVPR 2025）](https://openaccess.thecvf.com/content/CVPR2025/html/Li_UA-Pose_Uncertainty-Aware_6D_Object_Pose_Estimation_and_Online_Object_Completion_CVPR_2025_paper.html)：不完整模型的可见性与不确定性建模。
- [SDFit（ICCV 2025）](https://anticdimi.github.io/sdfit/)：SDF 与 mask、normal、depth 的 render-and-compare 融合以及形状/位姿共同优化。不能仅把 RGB/SDF 混合损失宣称为首次提出。
- [Isaac Sim 官方物体合成数据](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/replicator_tutorials/tutorial_replicator_object_based_sdg.html)、[相机传感器](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/sensors/isaacsim_sensors_camera.html)。

## 7. 确定性的首轮交付标准

1. 原版 SDFR 在 1–3 个 CAD 物体上对同一输入可重复；输出位姿误差与耗时日志。
2. 2–3 个对照 ICP/FRICP 基线；评测程序和可视化一致、随机种子锁定。
3. 在标准、噪声、遮挡三个场景中，B0/B1/B2 的消融有复现记录；失败案例提供点云/SDF 可视化。
4. 完成独立 Isaac RGB-D/K/GT/Mask 数据导出后，可不修改 SDFR 评测核心即可加载 Isaac 数据。
5. 再据失败分布选择论文主线，**不预先承诺提升幅度或必然发表**。

## 8. 状态

- 2026-11-19：已提出研究路线并核对代码/官方文档。
- **未开始**仿真环境安装、代码复现、实验运行或性能验证。

## 9. 2026-11-19 先行工作审查：方向 A 并非新颖的研究主张

**触发问题**：用户指出“鲁棒 SDF + 不确定性 / 可见性加权”似乎已在 ReadPaper 中读过；重新查验仓库中已归档的 GFLA、Active 6D Pose、FoundationPose 精读笔记，并查阅外部 RNNPose、UA-Pose、SDFit 与 BayesSDF 资料。

### 9.1 仓库已有的直接重叠思想

| 已归档论文 | 已有实现 / 思想 | 与拟议 A 的关系 | 不能混同的环节 |
|---|---|---|---|
| [GFLA（T-RO 2026）](../../../grasping/doi-10.1109-tro.2026.3715994/notes/explainer.zh-CN.md) | VASS 可见性感知软监督：按可见/未见区域和表面距离给 **SDF 的体素级训练损失**加权；用户曾详细追问创新点 2 | **概念重合度最高**：弱化不可靠几何信息影响 | GFLA 训练场景几何 SDF，SDFR 在**推理时**固定已知 CAD SDF 并更新 SE(3) |
| [Active 6D Pose（2025/26）](../../../active-perception/arxiv-2503.03726/notes/explainer.zh-CN.md) | 使用网络预测的二维中心协方差 \(\Sigma_{u,k}\)，最小化 \(\sum r_k^\top\Sigma_{u,k}^{-1}r_k\) | **优化层面很接近**：观测质量差则减小其位姿优化权重 | 加权的是多帧二维中心重投影残差，而非点到三维 SDF 的残差 |
| [FoundationPose（CVPR 2024）](../../arxiv-2312.08344/notes/explainer.zh-CN.md) | Render-and-Compare Refiner/Scorer；其笔记第 10–11 节还整理 UA-Pose 和不完整模型/置信度研究 | 说明位姿估计的可信度、深度质量、模型不确定性早已是活跃问题 | FoundationPose 主 refiner 不是 SDFR 式在线逐点 SDF 残差优化；排序分数≠校准概率 |

**结论**：此前将“鲁棒损失 + 每点可靠性加权”评为最有学术创新潜力，过度乐观。它仍是值得实现的工程基线与控制变量，但不能未经新颖性调查就宣称研究创新。

### 9.2 仓库外必须对比的已发表 / 公开工作

- [RNNPose（TPAMI 2024）](https://doi.org/10.1109/TPAMI.2024.3360181)：利用图像与 CAD 描述子的对应一致性检查，对不可靠的对应关系降权并在可微 LM 中优化位姿。**直接覆盖“坏测量降权、鲁棒位姿优化”范式**，只是残差来源不是 SDF。
- [UA-Pose（CVPR 2025）](https://openaccess.thecvf.com/content/CVPR2025/html/Li_UA-Pose_Uncertainty-Aware_6D_Object_Pose_Estimation_and_Online_Object_Completion_CVPR_2025_paper.html)：区分已观测 / 未观测参考几何，估计不确定性并指导在线补全；不是同一个 SDFR 任务，但“模型不确定性”已有先例。
- [SDFit（ICCV 2025）](https://openaccess.thecvf.com/content/ICCV2025/html/Antic_SDFit_3D_Object_Pose_and_Shape_by_Fitting_a_Morphable_ICCV_2025_paper.html)：可变形 SDF + mask/normal/depth 联合渲染比较，迭代优化形状和位姿，说明简单叠加图像约束也不独创。
- [BayesSDF（2025 预印本）](https://arxiv.org/abs/2507.06269)：显式估计神经 SDF 的表面/模型不确定性，提醒“模型几何不确定性”也不是空白；其主任务不是 SDFR 刚体 pose refinement。

以上并不意味着每篇都解决同一个实验问题，而是已有方法足以要求更严格的差异和基线。没有确认“某某首次”主张。

### 9.3 修订后的研究决策

- **保留 A 作为工程基础与消融**：B0 原始 SDFR 的 L1、B1 标准 robust M-estimator、B2 可见性/掩码置信加权、B3 多视角/时序一致性。这些默认都是 prior-art-inspired baseline，不作为独立贡献。
- **先构造原法失败模式，再提出差异化科学问题**。值得测试但不预设新颖性的假设包括：(i) 同时传播观测协方差和神经 SDF 几何不确定性至位姿残差；(ii) 可见性、遮挡边界与自由空间约束下的错误置信识别；(iii) 在固定/腕部相机自遮挡中检测位姿不可辨识并拒绝不安全动作。每项都需要对照 UA-Pose / RNNPose / 原始 SDFR，并进一步做 2026 年先行工作检索。
- **避免概念迁移误当贡献**：将 GFLA 的 VASS 改成 SDFR 的在线点权重，仅是训练/推理阶段的迁移；将 Active 6D 的协方差权重改成 SDF 残差权重，也不能自动认为新颖。若无具有区别的模型、理论约束和严谨消融，属于已有方法组合。
- **研究排序更新**：先复现 + 失败诊断（确定性工作）；其次验证“观测噪声 vs SDF 模型误差 vs 姿态不可观测性”的可区分性与评价指标；最后才确认是否有真正可发表的创新角度。

**备注**：方向 A 最初记录在本文件第 2–3 节，现在保留作为历史方案，不覆盖；以后谈创新优先参考本第 9 节的修订结论。

## 10. 2026-11-19 新一轮系统文献调研：从通用鲁棒性转向可检验创新

> **本节更新此前第 2、3、9 节的推荐优先级**，并不删除历史方案。调研来源包括作者 SDFR 源码、ReadPaper 的 GFLA/Active 6D Pose/FoundationPose 精读，以及截至 2026-11-19 的 CVPR/ICCV/WACV/IROS/ICRA 2025–2026 和预印本。尚未运行实验、不能保证新颖性或性能提升。当前列出的“候选贡献”属于研究假设，需后续进一步查新与实测。

### 10.1 相关工作与直接竞争

| 先行研究 | 已有能力 / 研究问题 | 对 SDFR 选题的限制 |
|---|---|---|
| [GFLA/VASS (T-RO 2026)](../../../grasping/doi-10.1109-tro.2026.3715994/notes/explainer.zh-CN.md) | SDF 训练阶段可见性软监督 | “按可见性加权”不是新概念；但作用阶段不同 |
| [Active 6D Pose](../../../active-perception/arxiv-2503.03726/notes/explainer.zh-CN.md) | 观测协方差加权重投影、多峰位姿和 NBV | 简单 SDF 残差加权、多视角/NBV 不能当首创 |
| [RNNPose (TPAMI 2024)](https://pubmed.ncbi.nlm.nih.gov/38289852/) | 在位姿优化中对错误对应降权 | 鲁棒加权的位姿精修有先例 |
| [WALDO (WACV 2026)](https://openaccess.thecvf.com/content/WACV2026/papers/Pakdamansavoji_WALDO_Where_Unseen_Model-based_6D_Pose_Estimation_Meets_Occlusion_WACV_2026_paper.pdf) | 遮挡下可见区域动态采样、多假设、迭代 refinement | 这些模块不能当作独创 |
| [SDFit (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Antic_SDFit_3D_Object_Pose_and_Shape_by_Fitting_a_Morphable_ICCV_2025_paper.html) | 可变形 SDF + mask/depth/normal 等渲染证据优化形状与位姿 | SDF+RGB / 模型在线适配并非空白 |
| [BOP-Distrib (WACV 2026)](https://cea-list.github.io/BOP-Distrib/) | 每图像可见性诱发的多模态姿态真值与分布评价 | “发现/输出多姿态”不是独创；该评价工具可作为基线 |
| [WALDO/ActivePose (2025)](https://arxiv.org/abs/2509.11364) | 基于姿态歧义的主动相机视角与跟踪 | NBV + 模糊消歧有直接先例 |
| [GPD-AP (ICRA 2026)](https://doi.org/10.1109/ICRA57385.2026.11695783) | 抓取姿态驱动的主动感知与困难遮挡测试 | “抓取任务导向视角选择”已有工作 |
| [Deterministic Object Pose Confidence Region Estimation (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Deterministic_Object_Pose_Confidence_Region_Estimation_ICCV_2025_paper.html) | 共形校准关键点区域，再传播至 6D 置信区域 | 仅增加位姿置信区域不足首创 |
| [Pose accuracy is not grasp reliability (2026)](https://arxiv.org/abs/2609.13235) | 位姿残差后验与模拟抓取风险 | “pose 不等于 grasp risk”已有直接论证 |
| [Gradient-SDF (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Sommer_Gradient-SDF_A_Semi-Implicit_Surface_Representation_for_3D_Reconstruction_CVPR_2022_paper.html) | 显式存储梯度并用于 SDF tracking 和光度 BA | “SDF 加梯度”不是首次 |
| [BayesSDF (ICCV 2025 Workshop)](https://arxiv.org/abs/2507.06269) | 神经 SDF 几何不确定性与 Hessian/Laplace | “SDF 本身有不确定性”不是首次 |
| [DualPP (2026)](https://doi.org/10.1016/j.cad.2026.104113) | 自监督 SDF 值与梯度精度改进 | 改进 SDF 梯度也已有先例，必须聚焦下游 pose 目标 |
| [CRISP (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Shi_CRISP_Object_Pose_and_Shape_Estimation_with_Test-Time_Adaptation_CVPR_2025_paper.pdf) | 测试时形状/姿态联合纠正 | CAD 模型偏差的形状适配已有先行工作 |
| [CausalPose (2026)](https://doi.org/10.1016/j.patcog.2026.113905) / [KineFuse (IROS 2026)](https://rasevents.org/uploads/documents/pdfviewer/f2/6e/232157-4134.pdf) | 视觉–触觉可靠性融合、手内遮挡姿态跟踪 | 简单加入触觉+动态权重也非新 |
| [SyncHeC (2026 preprint)](https://arxiv.org/abs/2611.02999) | 空间外参和时间偏差的连续联合标定 | “矫正腕相机标定与延迟”不能直接宣称无先例 |

### 10.2 研究候选排序（收敛到 3 项）

**候选 1｜面向下游位姿优化的神经 SDF 梯度质量与可控更新（优先做第一项探索）**

源代码事实：[lib/trainer.py](https://github.com/pangbo1997/SDFR/blob/master/lib/trainer.py) 的 `step_geometry` 主要对每级 SDF 值使用 `((pred-gts)**2).sum()` 训练；[sdfr.py](https://github.com/pangbo1997/SDFR/blob/master/sdfr.py) 在线使用 `mean(abs(sdf))` 对 `SE(3)` 位姿求导更新。即训练主要约束数值，推理的改进方向依赖 `∇f_theta`；这构成值得量化的目标不一致，但**目前未证明它造成了显著性能瓶颈**。

位姿局部雅可比：
```math
J_i = \frac{\partial f_\theta(T^{-1}p_i)}{\partial\xi}
= \nabla f_\theta(x_i)^{\top} \frac{\partial x_i}{\partial\xi}.
```

研究假设：SDF 数值 MSE 小，不代表局部梯度方向、局部优化条件数或迭代轨迹足够适合姿态估计。可做：
- exact mesh SDF/法向（Oracle） vs 原版 `OctreeSDF`；
- 数值误差、梯度方向误差、非表面区域有效梯度范围、SDF 损失下降曲线、ADD-S、位姿收敛盆地；
- grad/eikonal supervised SDF + 带梯度可信半径的位姿更新；与 Gradient-SDF/DualPP 对比；
- 若 exact mesh 与 learned SDF 的位姿精度和收敛盆地几乎一致，**放弃这条主线**。

创新差异需要聚焦 **downstream pose loss 与稳定 SE(3) 修正**，不能只提“新的梯度监督”。

**候选 2｜部分可见场景下的可辨识性、真实位姿风险与拒绝/补充观测决策（更偏具身）**

研究切口不是简单“置信度高低”，而是区分：
1) 数据是真的不支持唯一姿态（几何对称 / 遮挡）；
2) SDF 梯度几何误差导致解偏移；
3) 有信息却因坏初值而卡在局部极值。

可先对每次 SDFR 优化计算局部信息矩阵 `H ≈ J^T W J` 的谱及多个初始旋转解的分歧，结合 mask/depth consistency，而不是只用 SDF loss 判定成功；在可观测性不足时输出保守的候选解/集合或 abstain，交由下一次观测处理。必须对照 [BOP-Distrib](https://cea-list.github.io/BOP-Distrib/)、[ICCV 2025 6D confidence regions](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Deterministic_Object_Pose_Confidence_Region_Estimation_ICCV_2025_paper.html) 与 [pose posterior grasp risk (2026)](https://arxiv.org/abs/2609.13235)。单一局部 Hessian 对多峰歧义无能为力；应检验多初值与对称群。

若用户线缆穿孔相关，任务相关指标应从总体 ADD 转向 **孔轴/钩面/穿线路径关键位置的容差**（只选择环境刚体为 SDF 目标，不把柔性线缆当刚体）；实验需要防止“用了 GT 姿态做安全判别”的泄漏。

**候选 3｜时间预算受限的多视角与机器人自遮挡下的可信 6D 精修（系统性扩展）**

固定相机+腕相机共用 `T_{world, camera}`；同步、手眼标定、抓手遮挡与动作引起的深度失效会耦合。候选问题：是否能用**当前 SDF 几何的方向性信息缺口**决定继续 refine、等待新帧、切换相机、重新观察还是取消接触动作，并把运动耗时与任务容差计入规划？

此方向与 ActivePose/GPD-AP/Active 6D/SyncHeC 等高度相关；只有证明在同输入、同相机预算、同总时延及强对照条件下有新的机制与可观收益，才可能形成新的论文贡献。开始阶段只作为**验证平台扩展**而不是直接作为主创新。

### 10.3 先做一个能否决题目的最小诊断实验

- 2–3 个 CAD：非对称、有接触细节/小孔、近似对称。
- **同一真值点云/初始扰动**下，三种 SDF 表示：精确 CAD 距离查询（作为性能上限）、官方 OctreeSDF、增加 Eikonal/法向梯度监督的 SDF。
- 同一优化器与点数预算：对比 `|f-d_CAD|`、梯度角误差、下降方向余弦、姿态估计 P50/P95、成功率、初始化盆地、耗时。
- 错误分析分层：无遮挡/部分遮挡、点云污染、真实 CAD 偏差、几何近对称。
- **决策门槛**：若“更优 SDF 梯度”并不明显改善 pose 性能，转入候选 2 的不可观测性与风险；若确有强因果关系，再设计 pose-aware training 或 adaptive trust-region。不要先安装 Isaac Sim 才开始测。
- 若发现纯深度真值都不足以确定正确解，必须改进观测或输出不确定性，**不能要求优化器从无信息中恢复真值**。

### 10.4 论文贡献形成条件

1. 比过去方法真正多回答一个问题（不是简单模块组合）。
2. 不只是平均 ADD 小，还应显示统计可靠的失败案例改善、收敛盆地及计算成本。
3. 与 SDFR、Exact SDF、Gradient-SDF/DualPP、FoundationPose、WALDO 等强而公平的基线对照，按任务约束选择子集。
4. 在纯合成和真实/半真实观测中都验证，明确可用真值的阶段；方法不能依赖测试集 GT 预测信心。
5. 不声称已验证原创性：上述只是较完整查新后的**候选假设**，严格 novelty 还要结合最新版综述、论文全文及精确引文网核查。

### 10.5 本轮明确建议

**先实施“候选 1 的最小诊断实验”，而不是直接把候选 1 写成新算法。** 它最接近 SDFR 的研究对象，能用已有 CAD/PyTorch3D 做快速假设筛选。候选 2 作为具身操作方向备选；候选 3 作为后期 Isaac Sim 系统验证路线。**截至当前只有调研与设计，没有运行任何仿真、训练或实验。**
