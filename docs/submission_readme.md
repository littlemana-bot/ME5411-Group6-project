# Group 6 复核包与最后补充项

已生成英文技术报告、四张可编辑 SVG 流程图、PDF 中的阶段图与实验表格，以及包含代码、模型、数据包和运行说明的复核压缩包。

## 报告状态

当前是 **技术草稿**，不是已审核的最终提交稿。题目明确要求小组自己的想法、分析和推理；草稿保留了小组讨论、结论与实际贡献的待填位置。没有生成或冒充成员信息和课程许可。

- PDF：`output/report/ME5411_Group06_Report_Draft.pdf`
- 可编辑文本：`docs/report_draft.md`
- 流程图：`docs/figures/*.svg`，对应 PNG 可用于插图。
- 构建代码：`tools/build_report.py`，使用 ReportLab 和 Pillow；报告构建工具不参与分类。

PDF 使用 11pt Arial 正文和单倍行距，当前共 27 页，低于 50 页上限。题目说典型长度为 30–40 页，不是最低页数；补入小组分析后应再次检查页数和排版。

Markdown 是可编辑文本源稿；直接改 Markdown 不会自动改变 PDF。请将定稿文字同步到构建脚本的对应段落，或在小组采用的编辑器中完成定稿后重新导出。当前环境未提供可用的 Word 渲染器，因此没有交付未验证排版的 DOCX。

## 还需要小组提供

1. 四位成员的姓名、学号；可按 `docs/members.example.json` 建立 `docs/members.json` 后重建封面。
2. PCA/SVM 在 Part 2 讲过的课件页或章节。现有题目仅要求使用课程讲过的方法；外部算法文档不能证明课程覆盖。
3. 小组对增强、滤波、分割、模型差异的原始分析、方法选择理由、结论及实际贡献。提供文字后可进行语言编辑和排版。
4. 确认本地更新版 `Test_Image_2026.bmp` 是题目所指 `charact2.bmp` 的课程指定版本。

完成这些信息后，再生成最终报告和 `Group_06.zip`。目前包名为 `Group_06_review.zip`，并在包内标明尚未完成的项目。

## 运行与复现

从复核包解压进入 `Group_06` 目录：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-tested.txt
Expand-Archive -LiteralPath dataset_2026_updated.zip -DestinationPath dataset_2026_updated -Force
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
.\.venv\Scripts\python.exe segment_characters.py
.\.venv\Scripts\python.exe -m task7.verify_results
.\.venv\Scripts\python.exe -m task7.recognize
```

以上验证保存模型。完整重新训练为 `.\.venv\Scripts\python.exe run_task7.py`，会更新正式模型和指标；应在独立副本中执行，保留对照结果。快速自检为 `run_task7.py --quick`，结果隔离保存到 `output/task7_smoke/`。

报告重建需要额外安装 `reportlab` 和 `Pillow`，并在 Windows 上使用系统 Arial 字体文件：

```powershell
python tools/build_report.py
python tools/package_submission.py
```

压缩包 CRC 和每个成员文件的 SHA-256 均由打包脚本检查。`output/submission/package_integrity.json` 保存最终压缩包哈希。缓存、临时环境及 Git 元数据不会打包。

提交要求：报告与代码放在 `Group_06.zip`，于 2026-11-20 23:59 新加坡／香港时间前上传 Canvas。本轮未执行 Canvas 上传。
