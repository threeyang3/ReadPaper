# GFLA 阅读记录

## 2026-11-13｜查找、分析用户提供的 Cubox/微信科技文章中引用的论文

**来源**：[CAAI认知系统与信息处理专委会的中文解读](https://mp.weixin.qq.com/s/pKhe56xuswkVt9PI7_Nwcg)；用户提供的是 Cubox 阅读页 HTML。

**核对**：论文确认为 Xiao Ning 等人的 GFLA，IEEE T-RO 42:3141–3159（2026-07-23），DOI 10.1109/TRO.2026.3715994；[IEEE 原始摘要](https://ieeexplore.ieee.org/document/11622692) 与 [官方代码](https://github.com/Nx1021/GFLA-release)。

**讨论核心**：CPDN 用可见接触与另一端接触的二维投影来降维，SCN 推断隐藏几何，接触族在剩余射线参数上展开三维候选，通过法向力闭合、接近角与预测 SDF 碰撞检查产生 6D 抓取。

**阅读提醒**：
1. “至少一个接触点可见”与“背面连续可补全”是启发式先验，不是物理普适定理。
2. 当前源码的二维接触预测头为 6 通道（置信度、独立分数、两个二维位置），而中文页面写 5 通道，需完整论文复核。
3. 绝不能把 GFLA 的场景预测 SDF 与上一篇 SDFR 的已知 CAD 训练 SDF 混同。
4. 4.5 FPS 等时延与部分消融数据目前只经二次解读获得，尚未核查正式论文 PDF。
5. 鉴于仓库为公开仓库，正式 PDF 的转载权未核实，当前只存 DOI/原文链接及原创中文分析、SVG。

**后续可做**：根据 [精读笔记](explainer.zh-CN.md) 开展 SDF/CPDN/DFM/VASS 的逐式讲解、真机复现和与 GraspNet/AnyGrasp 比较。
