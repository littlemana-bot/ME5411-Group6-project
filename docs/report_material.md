# ME5411 Group Project — 报告素材与待办（Task 1–7）

本文件汇总当前进度、可直接引用的图表与数字，以及报告骨架。报告正文由小组撰写，
数字均来自仓库中的结果文件，可随时对照复查。

---

## 1. 交付物清单

| 要求 | 状态 | 证据位置 |
| --- | --- | --- |
| Task 1 显示原图 + 对比度增强实验 | 完成 | `output/segmentation/01_original.png`、`02_contrast.png` |
| Task 2 5×5 均值滤波 + 不同尺寸比较 | 完成 | `03_mean_3x3.png`、`03_mean_5x5.png`、`03_mean_7x7.png`（显式实现，见 `segment_characters.py: mean_filter`） |
| Task 3 截取含 `HD44780A00` 行的子图 | 完成 | `04_roi.png`，ROI = (10, 185, 980, 350)，右下边界不含 |
| Task 4 二值化 | 完成 | `07_local_contrast.png` → `08_binary.png`（局部对比度阈值 10） |
| Task 5 字符轮廓 | 完成 | `10_contours.png`（含字符孔洞边界） |
| Task 6 切分并标注字符 | 完成 | `09_cleaned.png`、`11_annotated.png`、`12_character_sheet.png`、`13_projection.png`、`characters/`、`segmentation.json` |
| Task 7 数据集 75/25 划分 | 完成 | `output/task7/split.json`（分层，seed 42；测试 1778 张全程不参与调参） |
| Sub-task 7-1 CNN | 完成 | `task7/cnn.py`、`task7/train_cnn.py`、`output/task7/cnn/` |
| Sub-task 7-2 非 CNN 方法 | 完成 | `task7/baseline.py`、`output/task7/baseline/` |
| Sub-task 7-3 两者对比 | 完成 | `task7/compare.py`、`output/task7/comparison/comparison.md` |
| 预处理与超参数实验 | 完成 | `output/task7/cnn/experiments.json`、`output/task7/baseline/metrics.json` 的 `study` 段 |
| 端到端识别芯片字符 | 完成 | `output/task7/recognition/`（三种输入，见 §5） |
| 报告正文（30–40 页，11pt Arial，单倍行距，≤50 页） | **待小组撰写** | 见 §6 骨架 |
| `Group_XX.zip`（报告 + 代码） | **待打包** | 见 §7 |

---

## 2. 数据集事实（可直接写入报告）

- 7 类字符：`0 4 7 8 A D H`，每类 1016 张，共 7112 张，全部 128×128、8 bit 灰度 PNG。
- 深色墨水、浅色背景；墨水像素占比 7%–19%，字体/字重/斜体风格差异很大（见 `output/task7/cnn/preprocessing.png`）。
- 划分：训练 4620（65%）、验证 714（10%）、测试 1778（25%），每类等量，分层随机，seed 42。
- 验证集仅用于选择预处理与超参数；测试集只在最终评估时使用一次。
- **Figure 1 的芯片照片没有参与任何训练**：照片只用于 Task 1–6 的处理演示和 §5 的端到端验证。

## 3. Task 7-1 结果（从零实现的 NumPy CNN）

结构：`conv3×3(16) → ReLU → maxpool2 → conv3×3(32) → ReLU → maxpool2 → flatten → dense128 → ReLU → dropout0.3 → dense7`，
共 267,975 个参数；只用 NumPy 实现 im2col 卷积、池化、全连接、dropout、softmax 交叉熵与 Adam。

**梯度校验**（float64 中心差分，6 个采样点/张量）：最大相对误差 1.35e-6（`fc1.b`），其余张量 ≤1e-6。

预处理与超参数实验（验证集，每项训练 8 epoch）：

| 配置 | 预处理 | 输入 | 通道 | lr | 增强 | 参数量 | 验证准确率 | 训练时间 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| croppad32_na | crop_pad | 32 | 16/32 | 3e-3 | 否 | 267,975 | **0.9958** | 90.6 s |
| croppad32_lr | crop_pad | 32 | 16/32 | 1e-3 | 是 | 267,975 | 0.9916 | 87.7 s |
| crop32 | crop | 32 | 16/32 | 3e-3 | 是 | 267,975 | 0.9902 | 89.5 s |
| croppad32_sm | crop_pad | 32 | 8/16 | 3e-3 | 是 | 133,351 | 0.9902 | 40.7 s |
| croppad32 | crop_pad | 32 | 16/32 | 3e-3 | 是 | 267,975 | 0.9888 | 87.8 s |
| raw32 | none | 32 | 16/32 | 3e-3 | 是 | 267,975 | 0.9860 | 90.2 s |
| croppad24 | crop_pad | 24 | 16/32 | 3e-3 | 是 | 153,287 | 0.9790 | 50.5 s |

最终模型（选定配置 croppad32_na，训练 25 epoch，验证集早停）：

- 测试准确率 **0.9910**，宏平均 F1 **0.9910**（1778 张中 16 张错误）。
- 训练 255.2 s；单张推理 0.850 ms，批推理 0.606 ms/张；模型文件 1,073,874 B。
- 主要错误：`0→D` 2 例、`8→0` 2 例、`A→H` 2 例、`D→0` 2 例，其余为风格化字体的单例错误（见 `errors.png`）。

## 4. Task 7-2 与 7-3 结果（PCA + RBF SVM 对比）

流程：标准化 → PCA 降维 → RBF SVM。调参顺序：预处理 → PCA 维数 → (C, gamma)。

- 预处理（PCA=100, C=10, gamma=scale）：`crop/32` 0.9986、`crop_pad/32` 0.9958、`crop_pad/16` 0.9944、`none/32` 0.9916。
- PCA 维数（crop/32）：100 → 0.9986（方差 0.935），200 → 0.9986，50 → 0.9972，30 → 0.9944。
- (C, gamma) 网格：`C=10, gamma=scale` 与 `C=100, gamma=scale`、`gamma=1e-3` 均达 0.9986；`gamma=0.01` 明显变差（0.969）。

最终模型（crop/32、100 维 PCA、C=10、gamma=scale、714 个支持向量）：

- 测试准确率 **0.9921**，宏平均 F1 **0.9921**（14 张错误）。
- 主要错误：`0→8` 4 例、`8→0` 2 例、`D→A` 2 例，其余为单例错误（见 `output/task7/baseline/errors.png`）。
- 拟合 0.18 s；单张推理 0.300 ms，批推理 0.089 ms/张；模型文件 1,050,368 B。

对比（统一测试集 1778 张）：

| 指标 | CNN (7-1) | PCA + SVM (7-2) |
| --- | --- | --- |
| 准确率 | 0.9910 | **0.9921** |
| 宏平均 F1 | 0.9910 | **0.9921** |
| 训练时间 | 255.2 s | **0.18 s**（约 1400 倍差距） |
| 单张推理 | 0.850 ms | **0.300 ms** |
| 批推理 | 0.606 ms/张 | **0.089 ms/张** |
| 可训练参数 | 267,975 | 72,100 |
| 模型文件 | 1,073,874 B | 1,050,368 B |

每类 F1（0/4/7/8/A/D/H）：CNN 0.9843 / 0.9961 / 1.0000 / 0.9902 / 0.9941 / 0.9862 / 0.9861；
SVM 0.9861 / 0.9961 / 0.9980 / 0.9882 / 0.9902 / 0.9921 / 0.9941。

**结论要点**：两者在干净测试集上实际打平（14 与 16 个错误，差距 0.0011，小于抽样噪声），
差距体现在成本而非判别力；但域偏移下 CNN 明显更稳（见 §5）。

## 5. 端到端识别（Task 6 切出的 10 个字符，期望 `HD44780A00`）

| 输入 | CNN | PCA + SVM |
| --- | --- | --- |
| `normalized`（切分后的干净二值字符） | `HD44780A00` — **10/10** | `H044780A00` — 9/10（1 个 D→0） |
| `gray`（照片原始灰度裁剪） | `HH44780A00` — 9/10 | `HHHHHHHHHD` — 1/10 |
| `gray_enhanced`（对比度拉伸 + 轻度去噪 + Otsu 裁剪框） | `H044780A00` — 9/10 | `HDDDDHDDDD` — 2/10 |

讨论点：训练集是干净的渲染字形，实拍裁剪存在低对比度与传感器噪点，属明显域偏移。
**CNN 从像素直接学习笔画特征，泛化到实拍图时仅损失 1 个字符；PCA + SVM 在同样输入上几乎失效**
（全体预测为 H），说明固定线性投影把噪声放大成了与类别无关的分量；加入对比度增强后 CNN 恢复到
9/10，而 SVM 仍只有 2/10。这正面回答了 7-3 要求的“两种方法差异从何而来”。

## 6. 图表清单（可直接插入报告）

Task 1–6（`output/segmentation/`）：

| 文件 | 建议图注 |
| --- | --- |
| `01_original.png` | Original 16-bit chip image (990×367) |
| `02_contrast.png` | Grayscale with autocontrast applied |
| `03_mean_3x3/5x5/7x7.png` | Mean filtering with 3×3, 5×5 and 7×7 kernels (explicit implementation) |
| `04_roi.png` | Sub-image containing the line `HD44780A00`, ROI (10,185,980,350) |
| `07_local_contrast.png` | Local contrast response (Gaussian σ=3 minus σ=25 background) |
| `08_binary.png` | Thresholded response (threshold 10) |
| `09_cleaned.png` | Components kept after area and height filtering |
| `10_contours.png` | Foreground boundary pixels, including character holes |
| `11_annotated.png` | Detected character boxes with left-to-right indices |
| `12_character_sheet.png` | Segmented characters in reading order |
| `13_projection.png` | Vertical foreground projection with the touching-pair splits marked |

Task 7（`output/task7/`）：

| 文件 | 建议图注 |
| --- | --- |
| `cnn/dataset_samples.png` | Dataset samples, one row per class: typeface variation within each class |
| `cnn/preprocessing.png` | Effect of the three preprocessing modes at 32×32 (none / crop / crop_pad) |
| `cnn/training_curves.png` | CNN training history: accuracy and cross-entropy loss per epoch |
| `cnn/confusion_matrix.png` | CNN confusion matrix on the 25% test split |
| `cnn/errors.png` | CNN misclassified test samples (true → predicted) |
| `baseline/pca_variance.png` | Cumulative explained variance of the PCA projection |
| `baseline/confusion_matrix.png` | PCA + SVM confusion matrix on the same test split |
| `baseline/errors.png` | PCA + SVM misclassified test samples |
| `comparison/accuracy_bars.png` | Test accuracy and macro F1 of both methods |
| `comparison/per_class_f1.png` | Per-class F1 of both methods |
| `comparison/training_time.png` | Training cost of both methods on the same split |
| `comparison/latency.png` | Inference latency of both methods |
| `recognition/label_normalized.png` | End-to-end classification of the segmented characters (clean input) |
| `recognition/label_gray.png` | Same characters from the raw photo crops (domain gap) |
| `recognition/label_gray_enhanced.png` | Same crops after contrast enhancement and denoising |

## 7. 报告骨架建议（11pt Arial、单倍行距、30–40 页）

1. **Title page**（含全部组员姓名与学号）— 1 页
2. **Introduction**：问题、目标、芯片字符识别难点 — 1.5 页
3. **Dataset**：照片 + 7 类数据集统计、划分（§2）— 1.5 页
4. **Tasks 1–6**：每步方法 + 流程图 + 结果截图 + 讨论 — 10–12 页
   - 用 `01`–`13` 各阶段图作为“中间结果截图”，每个任务给出参数与观察结论
5. **Task 7 设置**：划分、评测协议、为什么验证集与测试集分离 — 1 页
6. **Sub-task 7-1**：网络结构图、im2col/Adam 说明、梯度校验、预处理与超参数敏感性 — 4 页
7. **Sub-task 7-2**：PCA + SVM 原理、调参过程与结果表 — 3 页
8. **Sub-task 7-3**：§4 对比表 + 每类 F1 图 + 误差分析 + 成本权衡 — 3 页
9. **端到端与域偏移**：§5 表 + `label_normalized/gray/gray_enhanced.png` — 2 页
10. **Discussion & lessons learned**：自写 CNN 的收益与代价、数据量假设、失败案例 — 3 页
11. **References**：NumPy、Pillow、scikit-learn 版本与文档链接 — 0.5 页
12. **Appendix**：运行步骤（可复制 README §环境与运行）、关键代码清单 — 3–5 页

需要补充绘制的图：Task 1–6 的处理流程图、CNN 结构图、PCA + SVM 流程图（可用 draw.io / PowerPoint）。

## 8. 打包与复现

```powershell
python -m pip install -r requirements.txt
python segment_characters.py     # Task 1–6，输出 output/segmentation/
python run_task7.py              # Task 7 全流程（完整实验约 14 分钟）
python -m task7.train_cnn --figures-only   # 仅重绘 Task 7-1 的图，不重新训练
python run_task7.py --quick      # 每类 60 张的自检模式
```

打包：报告 PDF + 代码放进同一文件夹，压缩为 `Group_XX.zip`（XX 为组号），上传 CANVAS。
`dataset_2026_updated/` 体积较大（约 7 MB 数据集 + 8 MB 压缩包），可按课程要求决定是否随包提交。

## 9. 仍需小组确认的事项

- 报告是否统一使用英文，以及组号 `XX`、组员姓名学号的填写。
- 是否需要把数据集文件夹（`dataset_2026_updated/`）一并放入提交压缩包。
- §5 中 SVM 在实拍图上失效这一现象，是否需要在报告中额外补一个小实验
  （例如先用数据集图像做对比度退化训练，再看鲁棒性）——目前可作为“未来工作”一句话带过。
