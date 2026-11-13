# ReadPaper · 论文阅读档案

持续维护的论文原文、中文图解、问题讨论与研究启发。**每篇论文只建立一个长期档案**，避免每次对话重复产生文件。

## 阅读目录

| 领域 | 子方向 | 论文 | 首次阅读 | 档案 |
| --- | --- | --- | --- | --- |
| Robotics 机器人学 | Active Perception 主动感知 | [Active 6D Pose Estimation for Textureless Objects using Multi-View RGB Frames](papers/robotics/active-perception/arxiv-2503.03726/README.md) | 2026-11-11 | [精读笔记](papers/robotics/active-perception/arxiv-2503.03726/notes/explainer.zh-CN.md) · [交流记录](papers/robotics/active-perception/arxiv-2503.03726/notes/dialogue.md) |
| Robotics 机器人学 | Pose Estimation / Refinement 位姿估计与细化 | [Fast and Accurate 6-D Object Pose Refinement via Implicit Surface Optimization](papers/robotics/pose-estimation/doi-10.1109-tro.2025.3562484/README.md) | 2026-11-11 | [精读笔记](papers/robotics/pose-estimation/doi-10.1109-tro.2025.3562484/notes/explainer.zh-CN.md) · [交流记录](papers/robotics/pose-estimation/doi-10.1109-tro.2025.3562484/notes/dialogue.md) |
| Robotics 机器人学 | Grasping / Grasp Planning 机器人抓取 | [GFLA: A Grasping Framework With Learning-Based Perception and Analytical Modeling for Single-View Scenes](papers/robotics/grasping/doi-10.1109-tro.2026.3715994/README.md) | 2026-11-13 | [精读笔记](papers/robotics/grasping/doi-10.1109-tro.2026.3715994/notes/explainer.zh-CN.md) · [交流记录](papers/robotics/grasping/doi-10.1109-tro.2026.3715994/notes/dialogue.md) |

完整表格：[CATALOG.md](CATALOG.md)。分类与更新约定：[GUIDE.md](GUIDE.md)。

## 归档结构

~~~text
ReadPaper/
├── README.md                       # 主页与入口
├── CATALOG.md                      # 可检索论文索引
├── GUIDE.md                        # 分类、命名和增量维护规则
├── scripts/fetch_pdfs.py           # 从开放来源获取 PDF 并验证文件头
├── .github/workflows/archive-pdfs.yml
└── papers/
    └── robotics/
        └── active-perception/
            └── arxiv-2503.03726/
                ├── README.md       # 元信息、核心结论、资源入口
                ├── metadata.json   # 可机读的元数据、原文来源、许可
                ├── paper.pdf       # 原版 PDF（由归档工作流生成）
                ├── notes/
                │   ├── explainer.zh-CN.md
                │   └── dialogue.md
                └── assets/         # 自绘图示（SVG）
        └── pose-estimation/
            └── doi-10.1109-tro.2025.3562484/
                ├── README.md      # 书目与版权状态
                ├── metadata.json
                ├── notes/         # 中文精读与讨论
                └── assets/        # 自绘 SDF/ICP 图示
        └── grasping/
            └── doi-10.1109-tro.2026.3715994/
                ├── README.md
                ├── metadata.json
                ├── notes/         # 中文精读与交流记录
                └── assets/        # 自绘方法和接触族图示
~~~

## 设计原则

- **稳定 ID**：以 DOI 或 arXiv ID 建唯一目录，分类用于主归档，标签用于跨领域检索。
- **原文与解读分开**：PDF 保持来源原样；中文笔记单独维护，不把讲解改写进原 PDF。
- **持续讨论、增量更新**：对同一论文的后续提问，优先修订已有精读笔记，在 dialogue.md 记录新结论与讨论时间；不覆盖人工修改。
- **图文相邻**：SVG 图示放在相应论文的 assets 下，相对路径引用，GitHub 与 Obsidian 均可浏览。
- **尊重版权**：各论文独立记录来源版本和再发布许可；不是所有开放获取 PDF 都允许再上传。

> 注：此仓库目前为**公开仓库**。文献原文的版权与许可属于原作者，不因收录而转移。论文归档脚本只处理已标注可再分发许可的文献。

