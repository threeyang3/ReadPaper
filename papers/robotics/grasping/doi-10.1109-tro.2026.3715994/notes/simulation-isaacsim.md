# GFLA × Isaac Sim：Franka Panda 抓取仿真复现与遮挡评测方案

> **规划版本**：v1.0，2026-11-13。**状态**：研究与实现方案，不代表已经安装、运行 Isaac Sim 或复现了论文结果。
>
> 本方案依据 [GFLA 官方代码](https://github.com/Nx1021/GFLA-release)、[IEEE 论文摘要](https://doi.org/10.1109/TRO.2026.3715994)、[Isaac Sim 6.2](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/)、[Isaac Lab 3.0](https://docs.nvidia.com/isaaclab/3.0/isaac-lab/) 官方资料设计。原文没有经过完整 PDF 逐式核验。此文为**新增仿真研究方案，不是作者已经完成的实验**。

## 0. 研究目标与边界

研究目标：基于 GFLA 已发布的预训练模型，在 **Isaac Sim 6.2 + Isaac Lab 3.0 + Franka Panda 双指夹爪 + 腕部 RGB-D 相机** 环境中重现抓取推理与真实物理仿真执行；在不引入训练真值信息的前提下评价：

1. GFLA 在 Isaac Sim 合成 RGB-D 上能否输出几何合理的 GraspNet-format 6D grasp；
2. 经坐标变换与可达性/碰撞过滤后，Franka 是否能实际抓起并稳定持有物体；
3. 物体自遮挡、多物体遮挡、腕部夹爪自遮挡、相机角度、深度噪声与摩擦系数如何影响成功率；
4. 可选地，额外观察/重新规划能否改善强遮挡场景（**作为新增研究，不是原论文能力**）。

**不属于首轮范围**：GFLA 从头训练、柔性线缆动力学、真实机器人部署、实时 RL、大批量照片级渲染、多相机融合及视频闭环视觉伺服。

## 1. 版本与环境选型（以明确兼容性为最高优先级）

| 子系统 | 锁定建议 | 备注 |
| --- | --- | --- |
| Isaac Sim | 6.2.0（或相应发行包版本） | 物理：Isaac Sim PhysX；渲染：Isaac Sim RTX |
| Isaac Lab | 3.0.0 正式版 | 采用 Direct 型任务/独立工程；先单环境再批量 |
| 仿真 Python | 3.12 | 官方匹配 PyTorch 2.11 / CUDA 13 相关构建；按照官方锁文件与 overrides 安装 |
| GFLA Python | 作者测试的 3.11 | Torch 2.3.1+cu121、torchvision 0.18.1、PyTorch3D 0.7.7 |
| GFLA 权重 | 作者 release v1.0.0 RealSense checkpoint | 同时明确配对 FastSAM 规格、config 及提交 SHA |
| 任务执行 | Isaac Sim 新版 Franka Pick Place / cuMotion | 控制从简单分段状态机开始 |
| 相机 | RTX pinhole RGB + Z-depth | 与 D435 相似的内参/深度特征，先理想后引入噪声 |

- **两套独立进程与 Python 依赖**：不在 Isaac Sim Python 3.12 环境强装作者 GFLA torch 2.3；通过本机 ZeroMQ、共享内存或独立服务连接，首轮以简单可靠的 NumPy/序列化通道为主。
- **显卡**：Isaac Sim RTX 要求 NVIDIA RTX GPU，官方 6.2 最低参考为 RTX 4080、16 GB VRAM、32 GB RAM；H100/A100 不具有 RTX 核心，不能独立完成 Isaac Sim RTX 渲染。若 GPU 显存不足，仿真/推理部署至两台机器。
- **许可**：作者 GFLA-release 仓库目前未确认有根目录 LICENSE；仿真工程建议仅提供本方新写的桥接和实验代码，按上游条款使用其原始代码/权重链接，不未经授权搬运并再发布整个上游工程。
- **官方变更**：Isaac Lab 3.0 使用 XYZW 四元数、ProxyArray；避免把旧版 Lab 2.x 的 WXYZ 和 .data 张量代码不加转换地复制过来。

## 2. 系统框图

```text
Isaac Sim / Isaac Lab (PhysX + RTX)
  USD scene: Franka + realistic fingers + table + rigid objects
    ├── wrist RGB-D: RGB uint8, optical-axis depth float32 (meters)
    ├── K + T_world_camera (+ camera frame convention)
    ├── GT instance/depth/pose/contact for evaluation only [never sent to GFLA]
    └── joint state & physics step
             │ request: one RGB-D observation + K + camera→table transform
             ▼
 GFLA service (Python 3.11 / Torch 2.3.1 / PyTorch3D 0.7.7)
    CPDN → 2D contact map → SCN local SDF → analytical grasp filtering
             │ response: N×17 GraspNet-format candidates
             ▼
 Grasp adapter
    T_world_grasp, T_world_end_effector, max jaw width, IK, pre-grasp,
    robot collision checks, candidate ranking and fallback
             │
             ▼
 Franka state machine: home → settle → observe → infer → pregrasp
   → approach → close → lift → hold → score → reset
             │
             ▼
 Results: physical success, slip, contact, collision, occlusion, latency,
          visualizations, logs, camera intrinsic/extrinsic and exact seeds
```

## 3. GFLA 可复用接口与必须核对的源代码行为

官方 `test.py` 是 GraspNet 数据集评测脚本，不是直接读取独立图像的通用推理 CLI。关键底层入口位于 [models/CFG_main.py](https://github.com/Nx1021/GFLA-release/blob/main/models/CFG_main.py)：

- `CFGNet.forward(raw_rgb, depth, intr, Ts_Cn_in_T, usr_input_mark=None)`；
- `CFG_Predictor.run_model` 包含抓取生成、恢复到相机坐标系、宽度调整、NMS 等后处理；
- `_run_model` 支持 **4 项无 GT 输入** `(rgb, depth, intr, Ts_Cn_in_T)`，因此可以抽出一帧仿真图像推理入口；
- `cfg/trainer_config.yaml` 的 `restore_to_Cam: True` 使最终抓取恢复到相机坐标系；
- `cfg/trainer_config.yaml` 的 `WIDTH_ADD: 0.016` 已在作者结果输出之前加宽，外部适配器**不能再盲目叠加同样的 16 mm 补偿**；
- `cfg/config.yaml` 的 `max_3d_width: 100` 是作者候选范围，但 Franka Hand 行程约 80 mm，必须过滤不兼容宽度；
- `CFGNet.forward` 在局部场景选择时把 `centers_3d[..., 2]` 设为基于 `volume_size` 的桌面高度，故需保证 `Ts_Cn_in_T` 将相机观测点正确映射到 **Z 向上、桌面近 z=0 的工作台坐标系**；
- `cfg/config.yaml` 的 `modify_depth_by_table: False` 默认不做该额外深度裁剪；默认不代表可以省略 `Ts_Cn_in_T`。

**接口协议建议**：`InferRequest{rgb: uint8[H,W,3], depth_m: float32[H,W], K:float64[3,3], T_table_camera:float64[4,4], frame_id, stamp, seed, metadata}`；`InferResponse{grasps:float32[N,17], grasp_frame:'camera_optical', timings, diagnostics, model_version}`。实际 `Ts_Cn_in_T` 方向要按已核查的代码转换链写单元测试；协议字段的定义必须单一明确。需要记录其原始作者参数，不修改训练代码与 checkpoint 进行所谓“调参复现”。

**GraspNet N×17 布局**：`[score, width, height, depth, R(9), t(3), object_id]`。GraspNet 约定抓取局部坐标 **X 为进给/接近方向，Y 为两爪开合方向**，与 Panda tool frame 一般不同，必须定义固定 `T_grasp_ee`。

**坐标变换**：定义 `T_AB` 将 B 系坐标映射到 A 系。则（若 `T_CG` 为 GFLA 输出的抓取到相机变换）
`T_WE = T_WC @ T_CG @ T_GE`；其中 `T_GE` 为抓取坐标到期望 end-effector 控制坐标的固定变换。通过三轴可视化和已知的对象姿态做测试，避免把抓取本地 +X 接近方向与相机光轴混淆。

**深度与图像**：Isaac Lab `depth` 是沿光轴的 Z 距离（米），不是 range-to-camera；在 GFLA 适配器里必须检查是否重复进行毫米→米转换。保证 RGB 三通道色序与 FastSAM 期望匹配。目标坐标旋转由 SDK 提供的 camera optical convention 统一到 ROS/OpenCV `Z` 前、`X` 右、`Y` 下；Lab 3 所使用的四元数是 `xyzw`。

## 4. 场景和相机设计

### 4.1 首轮场景

- 基础：桌面、一个非透明刚体（先简单立方体/圆柱，再用具体网格）、Franka Panda、标准平行夹爪；
- 物理：刚体碰撞形状正确、质量和惯量合理、摩擦系数可控、重力、关节驱动和抓握力约束；不要在抓取时直接把物体 attach 到夹爪使其“假成功”；
- 复现实验时优先中等尺度、形状不复杂的抓取目标，再扩展至不同尺寸的日用品、YCB/GraspNet 类网格和从训练物体类别中排除的 novel-like 对象；
- Isaac Sim 的 USD 场景 asset 和物理 collision mesh 分开管理；碰撞网格与可视网格差异需记录。

### 4.2 相机

- Camera-A：场景外固定相机，仅作为观测对照，不参与 GFLA 初版的眼在手上结论。
- Camera-B：固定在 Panda 腕部/工具端法兰的 eye-in-hand RGB-D，相对安装变换 `T_ee_camera` 可配置。
- 相机内参 `K`、分辨率、光学坐标约定、近平面/远平面固定记录；先用 pinhole、理想 Z-depth，后引入仿 D435 深度误差。
- 首次验证可用 640×480 降低成本，但**正式与发布 RealSense 训练权重比较必须确认模型训练输入分辨率、相机内参与预处理一致**；当必须变更分辨率时对 `fx,fy,cx,cy` 相应缩放。
- 其他通道如 instance segmentation、GT surface normals、物体真值位置只用于离线分析和 Oracle 对照，不能提供给正常 GFLA 服务。

### 4.3 遮挡实验设计

1. **O0** 无遮挡：相机看到主要目标，作为基本性能基线。
2. **O1** 物体自遮挡：旋转物体或相机，显著改变可见表面比例。
3. **O2** 多物体遮挡：稀疏摆放 → packed → piled，控制密度、邻居个数。
4. **O3** 腕部/夹爪自遮挡：固定相机实际装在腕部，改变腕姿态、相机安装偏移、预抓姿态和夹爪开合，让手指/手腕真正进入图像并挡住目标。
5. **O4** 低角度观测：相同物体/相同随机种子，对比相机倾角；应区分低视角引发的物体互挡和机器人自挡。
6. **O5** 噪声和反光：深度丢失/量化、噪声、RGB 曝光/材质变化。

**遮挡率定义建议**：在冻结同一相机位姿与全部物体位姿后，比较带遮挡与去除指定遮挡体时目标单独可见像素数量，`r = 1 - A_visible / A_reference`。分别统计 `r_object`（其他物体造成）和 `r_robot`（夹爪/机械臂造成），不能仅用一个总遮挡比混淆原因。可用 instance ID 与深度比较、标注射线遮挡关系来交叉核对。

## 5. 分阶段最小化风险的执行顺序

### P0：预备验证（不接入仿真）

- 在独立 GFLA 环境使用作者 RealSense checkpoint，对官方 GraspNet 一张已知样本完成推理；
- 校验模型路径/分割权重、PyTorch3D/本地 NMS、输出非空、`N×17`、无 NaN/Inf，并与作者 release 的标准测试样本做对照；
- 锁定官方仓库 commit、权重 SHA256、FastSAM 大小、Python/package 版本；
- 单元测试抓取格式、宽度单位、旋转矩阵正交性和 score 范围。

**通关标准**：对作者数据样本能稳定获得格式正确的输出；若失败先解决模型环境，避免误判 Isaac Sim。

### P1：Isaac Sim 极简物理与相机测试（暂不调用 GFLA）

- 运行 NVIDIA 自带的 **Franka Pick Place standalone** 示例，确认真实 PhysX 的接近、夹紧、提升和释放；
- 加一台固定 RTX camera，输出 RGB+Z-depth+K；在画面中选择已知 3D 点，投影/逆投影残差做单位测试；
- 将相机搬到手腕上，验证手腕运动时 `T_world_camera = T_world_ee @ T_ee_camera` 正确；
- 用脚本依次把不同静态盒体放上桌面，保存 RGB/Depth/K/T/instance mask。

**通关标准**：即便不用 GFLA，机械臂也能使用 **GT/手工给定** 的抓取姿态把简单物体拾起；相机深度和坐标转换精度可重复验证。

### P2：将 GFLA 接入静态 Isaac 相机

- 单独实现 `gfla_service` 的无 GT 四输入模式，避免直接触发 GraspNet 数据集包装；
- 允许先将一帧 Isaac RGB-D 写为本地 `.npz` 并离线推理：用于确诊协议与图像问题；再切换 IPC 请求模式；
- 输出可视化叠加图：接触投影、grasp local frame、局部 SDF、原始点云与碰撞过滤候选；
- 要先正确恢复相机系的 6D 抓取，**不执行机械臂动作**。

**通关标准**：抓取姿态的几何位置、接近方向与物体实际位置相符；不能只看预测置信度，需至少渲染 6D 夹爪模型进行碰撞观察。

### P3：完整抓取执行闭环

- 输入单帧 RGB-D → GFLA Nx17 → 相机坐标转换到世界/robot base → 按 Franka 最大可夹宽度过滤 → IK/可达性、整臂避障 → 预抓 → 接近 → 闭合 → 提升 → 持有；
- 控制器采用新版官方 Franka Pick Place 的 measured-state + timed phase 结构，第一版不需要学习控制；
- 坐标应严格区分 grasp frame 与 Panda TCP，夹爪的 `width`、`depth`、`height` 分别控制对应几何，而不直接把它们当作 TCP 的 XYZ；
- 按先验顺序尝试下一个候选是 **实验协议选择**：首轮建议每个 episode 只执行一个 best feasible grasp，不在单次 trial 内反复尝试以免抬高成功率；
- 运动规划失败记录为 `UNREACHABLE`，与抓取夹持失败分开统计。

**通关标准**：至少通过简单物体的若干成功/失败录像、运动日志、接触轨迹形成可以审查的完整闭环。

### P4：专项遮挡/泛化/鲁棒性评测

- 分组评估 O0–O5，固定随机种子和场景清单，成对比较；
- 首先 10–20 个场景做 smoke，下一阶段每主要实验条件至少 100 个有效独立试次；复杂多物体使用同组场景多随机种子，报告有效样本数量和置信区间；
- 对“物体本身不可见”和“手爪遮挡”分别建立条件，一项只改变一个主要变量；
- 记录 GFLA 失败来源（候选空、姿态错误、分割混淆、SCN 补全/碰撞漏检、宽度不可用）和控制器失败来源（IK、轨迹、夹爪接触、滑移）；
- 暂不主动做 data augmentation / fine-tune，先记录冻结权重跨域基准。随后再单独测试适应与改善措施。

### P5（可选，研究性拓展）：主动观察与再抓取

- 在物体保持不动且抓取尚未执行时，让腕部相机从若干可达候选视角再观察；
- 每个视角单独运行原 GFLA，按无遮挡比/接触置信/预计可执行抓取数量选一个；再尝试基于多帧证据建立更高级的策略；
- **清楚标明为 GFLA + active view 的新方法**，不是论文原始单视角协议。

## 6. 评价指标与统计口径

### 6.1 核心成功定义（拟议）

- 抓取成功：执行一次预定闭合并提升目标，使目标质心高度相对抓取前提高 **≥0.10 m**，并保持 **≥1.0 s**，不借助焊接/attach，不发生脱落。可对无法提升 10cm 的特殊大物体另立预先公开的阈值。
- `SR_total = successful_trials / all_attempted_trials`，未检测到候选、不可达或规划失败也要计入分母；另报告 `SR_executed`（仅实际执行 grasp 的分母），防止混淆。
- 清空率 `CR`：对多物体 episode，在固定上限抓取次数内清空桌面的比例；物体接触后被推出桌面不算正常清空。
- 真实碰撞与误碰、掉落、滑移、接触力、抓取宽度适配失败、相机挡住目标比例、候选保留率、可达率、运行时间。

### 6.2 对照与消融（先基础后复杂）

| 组 | 观测 | 推理 | 用途 |
|---|---|---|---|
| A | Isaac ideal RGB-D | GFLA 原始预训练权重 | 官方方法 sim 域基线 |
| B | Isaac 带标定噪声 RGB-D | GFLA 同权重 | 深度/视觉 sim-to-real 敏感性 |
| C | 与 A 相同 | GFLA + GT 目标掩码（oracle） | 隔离分割/可见性误差，不作为方法成绩 |
| D | 与 A 相同 | GFLA + GT 几何/SDF（oracle） | 上界：几何补全误差，不作为方法成绩 |
| E | 与 A 相同 | 可信简单点云/解析 antipodal baseline | 比较是否真的优于朴素抓取 |
| F（扩展） | A / B 相同 | 已发布 GraspNet 类 6D grasp 方法 | 更强研究基线，需额外权重与许可 |
| G（扩展） | 单视角 vs 二次观察 | GFLA + 视角选择 | 验证自遮挡应对能力 |

注意 C/D 明确是 **oracle ablation**；它们使用仿真真值，只能说明误差来源，不能和 A 当作同等信息条件下的正式竞争结果。

### 6.3 数据记录

每次 trial 最少记录：

`run_id, seed, asset_ids, mass, friction, source_model_sha, sim_version, K, T_world_camera, T_table_camera, gripper_width, camera_mount, occlusion_object, occlusion_robot, num_raw_grasps, num_feasible_grasps, chosen_score, grasp_pose, action_log, IK_status, collision_reason, lift_height, hold_time, slipped, success, inference_ms, render_ms, control_ms, overall_ms`。

每个失败附保存 RGB、Depth、分割真值（仅评测端）、SDF/抓取调试可视化、简短轨迹视频和分类错误码。

总体报告提供 **成功次数/试次、百分比、95% Wilson 区间**；同一场景种子上做 A/B 成对比较。要分开记录 GraspNet **AP**（静态几何评分）与 PhysX **物理执行 SR**（真实仿真夹持结果）。

## 7. 关键工程细节与验收用例

### 7.1 单元测试

1. 单位换算：1m / 1000mm 路径一致、无重复尺度；
2. 投影与逆投影：选可见 XYZ → RGB 像素 → Depth 重建，误差在标定与分辨率规定阈值内；
3. 旋转：`det(R)≈1`、`R^TR≈I`，并且相机旋转 90° 以后规划坐标随之正确变化；
4. Isaac Lab3 四元数 `xyzw` 与作者 GraspNet 3×3 R 显式转换；
5. `Ts_Cn_in_T`：仿真已知一个桌面点 `z_table=0`，变换到表坐标后应接近该平面；
6. Grasp frame：local X 是接近方向、local Y 是夹爪开合方向，视觉化模型手指中心对齐；
7. Gripper width：输出含 `WIDTH_ADD` 额外开度；不可再重复添加，过滤 `>0.08m`，留出碰撞/行程余量；
8. 自遮挡：当夹爪遮住物体时，instance ID 和 z-buffer 判定遮挡归因正确，不能误把工具当作物体表面；
9. 结果无 NaN / Inf，能解析 `N×17`，不因没有任何 grasp 崩溃；
10. 受控种子场景重建时物体姿态、相机成像、物理参数一致，关键结果可重放。

### 7.2 物理仿真

- 推荐从固定物理步长 `1/120 s` 起步，必要时对轻薄、易滑物体减小步长、增加接触求解迭代；**这一数值是工程起始参数，不是论文配置或保证**。
- 记录静/动摩擦、质量、材质、接触偏置、夹爪驱动参数与控制更新频率，防止“调参调出成功率”。
- 用 GT 抓取和人工设置的夹爪接触测试区分模拟器物理问题；仅当 GT 正确抓取可重复成功，才能合理评价 GFLA 的感知成败。
- 使用 Isaac Sim 6.2 的 `isaacsim.sensors.experimental.physics.ContactSensor` 或更完整的接触读取 API；不要依赖已弃用的 6.x 旧传感器扩展。

## 8. 建议的外部实现工程布局

**不要把大量仿真代码、模型权重、数据集硬塞进 ReadPaper**。后续正式开发可建立独立 `GFLA-Isaac` 工程（或现有机器人研究仓库中的独立子模块），ReadPaper 保留论文、计划和研究记录。

```text
gfla-isaac/
├── README.md
├── configs/
│   ├── sim.yaml
│   ├── gfla.yaml
│   ├── camera_fixed.yaml
│   ├── camera_wrist.yaml
│   └── experiments/*.yaml
├── sim/
│   ├── assets/                    # USD + converted YCB meshes
│   ├── franka_scene.py
│   ├── camera_adapter.py
│   ├── panda_gripper.py
│   ├── motion_executor.py
│   └── grasp_task.py
├── gfla_bridge/
│   ├── schema.py
│   ├── client.py
│   ├── server.py                 # run under GFLA Python 3.11
│   ├── model_adapter.py
│   └── grasp_transform.py
├── experiments/
│   ├── conditions.py
│   ├── run_batch.py
│   ├── occlusion_metrics.py
│   └── evaluate.py
├── tests/
│   ├── test_grasp_frame.py
│   ├── test_camera_depth.py
│   ├── test_table_transform.py
│   ├── test_width.py
│   └── test_transport.py
├── reports/                      # tracked text and small figures only
└── outputs/                      # large data/logs/videos locally, gitignored
```

**上游原则**：把 GFLA 上游源码和 release 权重独立存放与版本锁定，优先调用/包裹原函数，不在首轮重构 CPDN/SCN 网络。后续需要补丁时仅在本工程放最小 patch，并附对比测试。CI 仅运行不依赖 RTX 的小单测；GPU/PhysX 真正实验采用本地/服务器按需执行，避免占用不必要的 CI 额度。

## 9. 优先级、停机条件与下一步

| Gate | 不通过就暂停后续集成 |
|---|---|
| G0 | 作者 RealSense 权重与 GraspNet 官方样本单帧推理正常 |
| G1 | Isaac Sim+Franka oracle 抓取可重复执行，RGB-D/K/T 合法 |
| G2 | 同一帧数据在桥接前后得到相同的 GFLA 抓取（误差在浮点容忍范围） |
| G3 | GFLA 抓取工具坐标可视化与可达性检测正确 |
| G4 | PhysX grasp 完整执行，成功/失败均有可归因记录 |
| G5 | 足量独立试次、随机种子、分组遮挡率、置信区间和可复验记录 |

**建议从 G0–G3 作为第一轮交付**：先验证 NVIDIA Franka Pick Place standalone 以及作者发布权重，再造一个普通刚体的单视角抓取。全部门槛通过之后再开展批量 PhysX 抓取和腕部遮挡研究。训练/微调应最后考虑，优先检验 frozen GFLA 的真实跨域表现。

## 10. 参考

- [GFLA 官方代码及发行权重](https://github.com/Nx1021/GFLA-release)
- [GFLA IEEE 原文摘要](https://ieeexplore.ieee.org/abstract/document/11622692)
- [Isaac Sim 6.2 安装要求](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/requirements.html)
- [Isaac Lab 3.0 安装](https://docs.nvidia.com/isaaclab/3.0/isaac-lab/installation)
- [Isaac Lab 3.0 Camera](https://docs.nvidia.com/isaaclab/3.0/concepts/sensors/camera)
- [Isaac Sim 6.2 Franka Pick and Place](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/examples/manipulation_franka_pick_place.html)
- [Isaac Sim 6.2 Contact Sensor](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/sensors/isaacsim_sensors_physics_contact.html)
- [GraspNet 17D format](https://graspnetapi.readthedocs.io/en/latest/grasp_format.html)
