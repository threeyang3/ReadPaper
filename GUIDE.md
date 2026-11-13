# ReadPaper 维护规范

## 1. 一篇论文一个目录

- 首选主键 arXiv ID（如 arxiv-2503.03726）；无 arXiv ID 时使用 DOI 的稳定短名。
- 目录 \`papers/<primary-domain>/<topic>/<paper-id>/\`。
- 论文更换版本时，在 metadata.json 更新 PDF version/source，并保留更新日期；不因标题变化重建目录。
- 论文主目录中的 README.md 负责「它是什么、资料在哪里」；notes/explainer.zh-CN.md 负责完整分析；notes/dialogue.md 负责对话增量。

## 2. 讨论如何更新

1. 先找到已有论文 ID，检查最新笔记与人工修改。
2. 直接勘误或补充精读相关章节；区分「论文结论」「解释性示意」「个人思考 / 可能的延伸」。
3. 在 notes/dialogue.md **追加**一条含日期、问题、结论、后续待办的摘要；不要用新文件制造重复历史。
4. 若用户提出新问题（例如公式推导、实验复现、与其他方法对比），在精读笔记中加相应小节，并在交流记录中链接该小节。
5. 修改 CATALOG.md 的标签和阅读状态仅在确有需要时进行。
6. 不修改原文 PDF 内容；新版本应保留版本、许可证及下载来源记录。

> 这里的「后续更新」是指用户在今后的会话中继续阅读该论文、且 GitHub 授权可用时，在该次交流中更新仓库；不表示系统能在没有新会话时自动监听或运行。

## 3. 图示

- 优先使用自绘 SVG、GitHub 原生 Mermaid 和数学公式；所有自绘图须注明是示意还是实验原图。
- 复用论文原图须符合原文许可、注明出处；不可把自行说明图当成论文的真实实验结果。
- Markdown 用相对路径引用图片，兼容 GitHub、Obsidian 和常见 Markdown 阅读器。

## 4. PDF 存储与许可

- metadata.json 明确 source_pdf、license、paper_file。
- 原文是开源可再分发协议时，可用公开仓库工作流获取并提交 PDF；禁用许可或授权不明的文献不得直接纳入公开仓库，应仅保留合法来源链接，或将仓库改为私有并遵守相应使用条款。
- 论文 arXiv v2 使用 CC BY-NC-SA 4.0：署名、非商业使用、相同方式共享，详见其 [arXiv 页面](https://arxiv.org/abs/2503.03726) 与 [许可正文](https://creativecommons.org/licenses/by-nc-sa/4.0/)。
- 文献作者与图示来源的署名信息保存在论文 README.md 中。
- 归档脚本只验证 PDF 文件标志、下载成功和合理文件大小；不声称验证文件签名或内容真实性。

## 5. 每篇论文推荐资料

| 文件 | 用途 |
| --- | --- |
| README.md | 书目信息、三句话总结、导航、版权 |
| metadata.json | 元数据与 PDF 下载指令 |
| paper.pdf | 原始来源 PDF（不改写） |
| notes/explainer.zh-CN.md | 长篇图解、原理与实验 |
| notes/dialogue.md | 继续提问时追加的问答摘要、勘误和研究决策 |
| assets/*.svg | 独立的高清示意图 |


## 6. 跨软件读取与自动更新

- 根目录 `catalog.json` 是公开、只读、确定性生成的目录索引（`python scripts/build_catalog.py --write`）。使用 `paper_uid` 关联 Zotero、PaperFlow 与 Obsidian；客户端不得以文件夹路径作为主键。
- `pdf_state=link-only` 表示版权或下载源未满足公开归档条件，不是流水线失败；`missing` 表示曾声明应归档 PDF 但文件还未取得。两者均不得被标为“全文已分析”。
- 新下载的 PDF 生成 `paper.pdf.provenance.json`，记录来源、版本和哈希；已有无 sidecar 的 PDF 作为 legacy 标识，不会自动覆盖。版本变化需要显式保存新旧版本及重新分析，不能仅改元数据而悄悄沿用旧 PDF。
- 分析范围 `analysis_scope` 与 PDF 可用性是两项独立证据；阅读客户端必须分别展示。修改人类笔记必须用增量编辑，导入程序应只写自己的派生文件。
- `catalog.json` 适合云端服务与阅读器读取；它**不是** PaperFlow 的标准公共 Feed，不能把它直接传给 `paperflow source sync`。正式桥接前必须做身份映射和数据层级适配。
