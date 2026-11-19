# Codex 执行任务书：RTX 5090 主机复现 FoundationPose × Isaac Sim

> 可直接粘贴给能够访问本地主机终端的 Codex/编程 Agent。编写日期：2026-11-19。目标论文：FoundationPose (CVPR 2024, arXiv:2312.08344)。这是**执行任务**，不是只写方案。

## 0. 总体角色与目标

你是具备 Linux、CUDA、PyTorch、Isaac Sim、OpenUSD、机器人视觉和 6D Pose Estimation 经验的研究工程师。请在我当前授权的**本地 RTX 5090 主机**上建立完整、可复现、可评估的 FoundationPose 仿真实验工程，实际执行、排错、保存原始日志与可视化结果，直至达到下文验收标准，或提供可复核的环境/权限阻塞证据。

论文：https://arxiv.org/abs/2312.08344
源码：https://github.com/NVlabs/FoundationPose
Isaac Sim 6.2：https://docs.isaacsim.omniverse.nvidia.com/6.2.0/
FoundationPose README：https://github.com/NVlabs/FoundationPose/blob/main/readme.md

**研究范围三层**：
A（必须）：使用官方预训练权重，在 Isaac Sim 生成的已知 CAD 刚体 RGB-D/Mask 上完成 model-based 初始位姿估计、连续跟踪和定量验证；
B（必须，分规模）：用 Isaac Sim 构建可配置的多物体物理堆叠、随机光照、材质、视角等合成数据管线，并对遮挡/深度噪声/相机运动进行消融；
C（可选扩展）：Franka Panda 腕相机模拟、基于已知相机运动的跟踪先验补偿、丢失检测与重注册；model-free 神经场重建以及训练/微调作为**独立可选模块**，不能阻塞 A/B。

**不要混淆**：论文使用 Isaac Sim 生成约 60 万个场景、120 万张图像训练模型，主要在真实 LINEMOD/YCB-Video/T-LESS/YCBInEOAT 等数据上评测；论文**没有**报告完整机械臂 Isaac Sim 闭环操作实验。不能把我们新搭的仿真实验冒称为作者官方端到端复现。若官方 Isaac Sim 场景生成脚本并未公开，明确标为“依据论文配置重建的仿真生成器”，不是“一键复现原场景”。

## 1. 执行边界及安全原则

1. 先检查主机，自动探测 OS、GPU、驱动、CUDA、PyTorch、内存、磁盘、现有 Isaac Sim/Docker/Conda，**不要猜测**。默认优先 Ubuntu 24.04 原生；若实际是 Windows/WSL2 或 Ubuntu 22.04，评估并采用受支持的路线，不允许默默重装操作系统。
2. 不得破坏已有科研环境。不得擅自升级系统驱动、内核、替换全局 CUDA、删除目录、修改他人项目、关闭 Secure Boot、清空 Docker images；涉及 sudo、重启、大于 30GB 的下载等，先给出明确变更影响并请求一次必要授权。普通用户级安装、目录内构建和小规模必要下载可直接执行。
3. 不使用 GitHub Actions/CI 消耗额度；所有冒烟、单元、benchmark、报告生成在本机执行。不得把权重、巨型数据集、秘钥或机器标识上传到公开仓库。
4. 记录任何实际运行失败的报错、命令、退出码、日志位置；禁止用虚构截图、模拟输出、测试桩成功代替真实 CUDA/Isaac Sim 推理。
5. 不设未经测试的 FPS/精度成功标准；对论文数字仅引用作者原始条件。在真实机器人上**不要执行实际机械臂运动**，此任务只做仿真。
6. 每阶段完成后更新本地 PROGRESS.md（已完成、验证证据、未解决问题），并推进下一阶段；遇到可定位错误主动修复，不要停留在“建议怎么做”的分析。

## 2. P0：硬件、环境与依赖审计

运行并记录：uname -a、lsb_release -a（若存在）、nvidia-smi、nvidia-smi -q、nvcc --version（若存在）、free -h、df -h、python --version、conda --version、docker --version、vulkaninfo --summary（若存在）。记录 RTX 5090 显存、驱动、CPU、RAM、盘符空间、现有 Conda/Docker、Isaac Sim 位置。

特别注意 RTX 5090 是 Blackwell sm_120，旧 PyTorch+cu124 可能报 no kernel image is available。使用支持 sm_120 的**稳定 PyTorch CUDA 12.8 或更高的已验证构建**；不能只看 torch.cuda.is_available()，还须打印 torch.version.cuda、torch.cuda.get_device_capability()、torch.cuda.get_arch_list()，并**真实执行一个 CUDA kernel**。对依赖编译使用与 PyTorch 兼容的 nvcc，若需要设置 TORCH_CUDA_ARCH_LIST，使用目标架构 12.0，并验证生成代码能执行。

优先 Isaac Sim 6.2.0 官方发行版或已装且兼容的受支持版本，明确记录版本；按照对应版本文档使用 API，不拼接旧 4.x/5.x 示例。在运行前使用官方兼容性检查器检查 NVIDIA driver/Vulkan/RT Core。Isaac Sim 6.2.0 官方系统要求页面列出 Linux 测试驱动 595.58.03；以当前版本官方文档为准，不把此示例数字机械当成所有机器的唯一合法版本。

建立 docs/environment-audit.md（命令、结果、支持判定、安装计划、组件版本和 Git SHA）。环境审计失败不能进入仿真采集。

## 3. P1：工程和环境隔离

建议工作区 ~/research/foundationpose-isaacsim5090（已有目录则复用且不得覆盖）。形成清晰可执行结构：

    foundationpose-isaacsim5090/
      README.md
      PROGRESS.md
      configs/
      environments/
      third_party/FoundationPose/
      simulation/
      inference/
      evaluation/
      scripts/
      tests/
      docs/
      data/       # 忽略 Git，只保留小样例和 schema
      results/    # 忽略大型输出，允许小型示意
      logs/
      reports/

FoundationPose 官方源码 clone 到 third_party，记录固定 commit SHA，所有兼容补丁**在独立分支/补丁文件**维护，勿直接手改后失去差异。检查当前 main 的 readme.md、environment.yml、requirements.txt、build_all_conda.sh、estimater.py、run_demo.py、predict_pose_refine.py、predict_score.py。

**硬性环境隔离**：
- Isaac Sim 进程使用独立的官方内置 Python（6.2 系通常 Python 3.12）或专用环境、官方 SimulationApp/Replicator；
- FoundationPose 使用其自己的 Conda/Python 环境（项目 environment.yml 现为 Python 3.11），配独立的 PyTorch/CUDA 原生扩展。
- 不在 FoundationPose 环境直接 pip 安装 Isaac Sim，也不在 Isaac Sim 内置环境里乱装 PyTorch3D/BundleSDF。
- 初期只通过磁盘数据交换（完整帧数据 + manifest），先**顺序**运行 Isaac Sim → 退出释放 GPU → FoundationPose 推理/评估，避免单卡 32GB 显存冲突。后期才评估双进程流式管线。

安装时优先使用与 5090 兼容的官方文档、可复现的正式版本；查验 RTX50 兼容性 issue/PR（如 NVlabs/FoundationPose#369）但**不要盲目 cherry-pick 未合并 PR**，只提取并测试必要兼容修改。安装和本地编译：PyTorch（CUDA 12.8+ 且含 sm_120）、PyTorch3D、nvdiffrast、Kornia、open3d/trimesh 及项目 mycpp。若 model-free 未启用，可先跳过 BundleSDF/mycuda、Kaolin，优先跑通 model-based。

至少通过：torch GPU kernel、nvdiffrast 最小 GPU rasterize、PyTorch3D import 与必要 op、mycpp import、自定义扩展 smoke tests。失败时依据第一个真实 traceback 定位，不允许简单通过捕获异常假装可用。

如果环境配置阻塞，给出隔离的 Docker 替代方案和诊断，保存 compatibility.md；未经许可不得重装系统/驱动。

## 4. P2：官方 Demo 最小闭环

下载官方 README 所需**两组权重**（refiner：2023-10-28-18-33-37；scorer：2024-01-11-20-02-45）及最小 demo_data，验证目录、权重文件大小/哈希可记录；外部来源需要用户授权或认证时给出准确地址和阻塞，不得伪造权重。

先执行官方 run_demo.py，第一帧 register、后续 track_one；保留原 demo 可视化、逐帧 4×4 相机系物体位姿、首帧及稳态推理耗时、GPU 最高显存。

禁止指向包含重要文件的 debug_dir，因为官方 run_demo.py 会清理该目录内容。对官方代码的必要修改记录为 patch，不删除其原功能。

正式建立 inference/adapters.py，以规范化数据类型、颜色空间、深度单位、Mask bool、K(3×3) 以及 4×4 Pose。只有官方 Demo 跑通且输出可验证，不把无法运行的代理/空值作为成功，才转 P3。

## 5. P3：Isaac Sim 单物体数据集与坐标正确性

在 Isaac Sim 里优先导入**一个已知尺寸、已知原始模型坐标系的刚体 mesh**（使用官方可用且许可合适的简单对象或自建几何），场景有平面、环境光和至少一个 RGB-D 相机。第一版 resolution=640×480，headless、确定性 seeds、少量帧；使用 GT instance mask。不要先加深度噪声、动态遮挡与多物体。

创建 simulation/generate_dataset.py 和由 Isaac Sim 官方 Python 环境启动的脚本，能够命令行指定 seed、scene_count、frame_count、resolution、sensor_type、output_dir 和 headless。首次实现 1 场景 × 10 帧的烟雾测试，然后 10 场景 × 20 帧。

每帧写出：
- color.png：RGB、uint8，明确是否 alpha/BGR；
- depth.npy：float32，**单位米，采用 camera-Z depth**（Isaac Sim 的 distance_to_image_plane/depth），不能与 distance_to_camera 欧氏射线距离混为一谈；
- mask.png：目标实例/prim 对应的二值 Mask；若同类多物体，应按 instance id 区分；
- camera.json：K、图像宽高、OpenCV optical camera frame 的 T_world_camera、时间戳；
- object.json：目标 GT T_world_object、mesh 名称、尺度/坐标约定与对称性；
- manifest.jsonl：相对路径、scene_id、frame_id、object_id、seed、数据来源、所有坐标变换以及 train/test 切分。

**强制约定**：T_AB 表示将 B 坐标系点转换到 A 坐标系；所有平移单位 m。世界真值位姿应满足：
T_camera_object_GT = inverse(T_world_camera) × T_world_object。
明确 USD 相机/世界轴、OpenGL 与 OpenCV optical frame 的转换；使用当前版本官方 API，不可直接把 USD transform 当 OpenCV 外参。

必须自动测试：
1. K、RGB/Depth/Mask 分辨率完全一致，Depth 非空且有有效像素；Mask 面积非零，实例身份正确；
2. 3D 物体中心经 T_camera_object_GT+K 投影回 RGB 中，在 Mask/物体附近；
3. GT pose 下渲染的 mesh 与图像 Mask 的轮廓有较高重合度；单位、尺度、mesh 原点没有错；
4. 相机移动但世界物体静止时，变换前后得到的 T_world_object 一致；
5. 样本采集后重读保持数据及坐标完全一致。

输出测试可视化：RGB + GT Mask、depth 伪彩色、GT 轴/盒子投影、GT mesh overlay（不同来源图像注明），保存校验报告。

## 6. P4：真实 FoundationPose 推理与 benchmark

基于官方类，而非重新实现一个简化版 pose predictor。读入 Isaac 数据的官方格式适配器，调用：
- 首帧 register(K, rgb, depth, ob_mask, iteration=5)；
- 后续帧 track_one(rgb, depth, K, iteration=2)；
- 正确维护上一帧状态，多个物体必须单实例独立状态，不共享一个 pose_last 导致串扰；
- 预测为 T_camera_object，转换世界坐标时用 T_world_object_pred=T_world_camera×T_camera_object_pred。

注意 FoundationPose 内部的 **centered mesh** 坐标变换；如果扩展了外部相机运动补偿，不可直接把 uncentered 原模型的 T_camera_object 塞进 self.pose_last，必须明确 adapter 使用的变换并写单元测试。

评估：
- translation error，单位 mm；
- rotation geodesic error，单位 deg；显式考虑几何对称性；
- ADD / ADD-S，mesh 点数及采样 seed 固定，物体直径 d 定义明确；
- 成功率 ADD(-S)<0.1d，另外记录 5mm/10° 等严格指标，但**新仿真数据不可冒充论文数据集精度直接比较**；
- tracking drift、失效次数、重新定位成功率（若已实现）；
- 首帧 register 延时、单帧 tracking 延时/FPS：分开统计数据读取、GPU 推理、渲染、总流程耗时；CUDA timing 使用 synchronize、预热，单独统计。
- 附加资源监控：GPU peak VRAM、CPU RAM、磁盘、是否发生 OOM。

每一场景输出 CSV/JSONL 和覆盖图（pred 轴、GT 轴、RGB、估计 mesh overlay），另给 summarize.py 输出均值/中位数/p90/p95、失败样例及短视频；不得只挑成功样例展示。

先 10 场景 × 20 帧、固定 seed 列表，跑通后再扩大规模。预期目标可以是简单 GT Mask + clean depth 场景误差足够小（例如 median ADD-S <0.1d）；若不达标，定位失败原因并记录，不得删掉失败样本或虚构成功率。

## 7. P5：论文风格的 Isaac Sim 合成场景与鲁棒性实验

先实现 5–10 个物体轻量场景，验证正确，再逐步扩至论文风格设定；物体资源只有授权素材才能下载，记录每个模型来源、license、单位与有效物理碰撞网格。

根据论文与补充材料进行**可配置的参数化重建**：
- 物体池 GSO/Objaverse（可先以少量替代 CAD 测试，不必立刻下载完整大库）；
- 每场景 70–90 物体为**论文配置上限目标**，先缩小以适配 5090、避免 OOM；
- 尺寸随机 5–30 cm、承载平台 1–1.5 m、重力掉落/碰撞/堆叠直至稳定；
- 每场景 2 台随机相机（平台上方半球，距离 0.2–3.0 m）、0–5 光源、不同 HDR、材质/金属度/粗糙度/光照方向随机；
- 多视角投影，instance masks、RGB、Depth、相机 GT、物体 GT、bbox；
- 渲染模式与质量可配置，清楚记录真实采集使用的相机数和帧数；
- 不强制复现论文 LLM+TexFusion 扩散纹理增强（官方也未公开相应数据/权重）；如未实现，报告明确标注缺失项。

对每类固定 seed 做分组对照，至少：
A 清洁单物体 + GT Mask；
B 几何遮挡分级（例如按可见像素比例区间汇总）；
C 深度 missing pixels / Gaussian 与随距离变化的噪声；
D GT Mask vs 可重复模拟的 mask erosion/dilation/漏检，以及可选外部分割器；
E 固定相机/移动物体 vs 移动相机/静止物体；
F 光照/材质随机化、模型纹理差异。

Depth 的噪声模型若只是**近似 D435**，要明确标注 simulated/noise-model，不可谎称等价于真实 D435 传感器。分组只改变指定变量，保持其他 seeds、相机、物体设置一致；并把失败/异常数据保留在报告中。

**关键**：论文的 600K 场景/1.2M 图像是原作者的全规模训练集，不是单台 5090 初始验收要求。先验证方法正确、复现性和数据质量，再提供可选的离线批量生成任务与磁盘预算评估。

## 8. P6：可选 Franka Panda + 腕部相机（仅仿真）

在 Isaac Sim 添加 Franka Panda 和腕部 RGB-D 相机，确保相机外参、内参与分辨率可重复导出。世界静止物体、移动腕相机场景中，原始 FoundationPose track_one 只用前帧单姿态作为初值，可能因运动而退化。

实现应用层 motion compensation prior：
T_camera(t)_object_prior =
inverse(T_world_camera(t)) × T_world_camera(t-1) × T_camera(t-1)_object。
注意正确处理 estimator 的 centered mesh 约定；添加输入输出变换单元测试以及 GT 先验实验。另做阈值或质量指标触发重新 register；不要把 Scorer 输出未经校准就叫做“成功概率”。测试相机小运动/大运动、遮挡和恢复，比较补偿前后失效率与耗时。无需真实机械臂硬件。

如果单 GPU 同时运行 Isaac 渲染和 FoundationPose 会 OOM，先使用分阶段轨迹采样/文件回放模拟闭环验证，并注明实时在线部分尚未达成。

## 9. P7：可选训练和 model-free，不允许替代主线

只有 A/B 全面完成后才评估：
- Model-free：对带 K、Mask、RGB-D 和参考相机位姿的小型参考集调用 BundleSDF neural field reconstruct，mesh 提取后再走同样 register/track；不等于零训练成本。
- 训练/微调：检查官方训练代码和权重许可，以单卡 5090 可承受的小数据子集做训练 loss、参数更新和验证；与预训练 zero-shot 模型分开报告。不得将“跑完若干小样本”称作完整复现论文百万级训练；大型实验先估算耗时、显存与盘占用，再征求授权。

## 10. 交付文件和可复现操作

必须生成并维护：

- README.md：一键启动顺序、环境、数据来源、命令、已完成范围、已知限制；
- docs/environment-audit.md：真实机器信息、版本表、GPU 架构测试结果、外部仓库 SHA；
- docs/architecture.md：两套环境隔离、schema、pose convention、数据流；
- docs/compatibility.md：5090 / PyTorch3D / NVDiffRast / CUDA 编译问题及补丁说明；
- docs/reproduction-protocol.md：场景参数、固定 seeds、数据质量、评估指标、与论文的差异；
- scripts/setup_foundationpose.sh、scripts/check_env.sh、scripts/run_smoke.sh、scripts/generate_data.sh、scripts/run_benchmark.sh、scripts/make_report.sh（名称可统一且必须真实可运行）；
- configs/*.yaml：GPU、resolution、随机种子、光照、遮挡、物体数量、传感器误差、refine iterations、输出目录；
- tests/：最少包含相机投影往返、pose inversion/compose、颜色/depth/Mask 维度与单位、mesh scale、predict/GT 对齐、同种子数据可重复性；
- reports/foundationpose_isaacsim_reproduction.md：**真实环境、已执行命令、结果表、耗时/显存、实际截图链接、失败样例、相对论文的差异及下一步**；
- logs/：构建日志、Isaac 日志、GPU 推理日志；
- results/：每个 scene/frame 的预测位姿、GT 位姿、误差、CSV/JSONL、可视化图/视频；
- PROGRESS.md：阶段状态、阻塞、证据与 commit SHA。尽量只把源码/小体积示例纳入 Git，数据/权重/大型日志忽略。

必须给出最少两条从干净环境可运行的命令：一次冒烟测试（生成小场景 → 官方 FoundationPose 初始化/跟踪 → 评估与可视化），一次可复现 benchmark（多个 seeds、输出完整报告）。若因环境隔离要分两个 shell 命令或包装脚本，写出准确调用，不得伪造“真正一键”。

## 11. 验收标准

**Level 1 可验收**：
1. RTX 5090 上 PyTorch CUDA sm_120 kernel、NVDiffRast、PyTorch3D、mycpp 真实通过；
2. 原始 FoundationPose Demo 使用官方权重输出非伪造的姿态和可视化；
3. Isaac Sim 至少生成 1 场景 × 10 帧正确 RGB、Depth、GT Mask、K、GT pose；
4. 坐标变换、深度定义、mesh 尺度和 GT overlay 测试全通过；
5. 从仿真生成的帧真实调用 register + track_one，并保存全部预测、评价和原始日志。

**Level 2 可验收**：
1. 至少 10 场景 × 20 帧完成可复现评估（按资源允许可分批），记录多个随机种子；
2. 遮挡、深度噪声、Mask 扰动、相机运动至少四项实验各自有对照和数值；
3. 能输出总体/分组指标、可视化、失败案例、GPU 显存/延时、版本信息；
4. 对照论文清楚区分相同输入任务与非同一 benchmark，不编造论文复现精度。

**Level 3 是扩展而非主线阻塞条件**：Franka 仿真腕相机、运动补偿、重注册与可达性/运行成本评估。

遇到硬件、许可、外部权重、兼容性或授权阻塞，不要宣告成功；在 PROGRESS.md 写明最后成功阶段、复现命令、完整报错、已试过的修复、能继续执行的最小任务。能做的继续做，直到实际完成或遇到明确外部阻塞。

## 12. Agent 工作方式

从 P0 立即开始检查本机、编写代码与运行测试，而不是只回复理论建议。自行合理选择可兼容的稳定版本，严禁把未经验证的 API 名称当作已运行成功。按阶段小提交到本地 Git 仓库；没有得到远程仓库地址/推送授权，不执行远程 push。尽量本机测试，避免任何 CI 额度。随时更新文档，把每一步实现与论文原始内容区分。最后用中文提交**复现状态表 + 环境版本 + 关键测试命令 + 核心定量结果 + 证据路径 + 已知未完成项**。