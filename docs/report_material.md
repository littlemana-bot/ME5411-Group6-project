# 报告素材索引（2026-10-14 更新）

旧草稿中的随机划分成绩、SVM 计时、参数量比值和部分解释已停用。原随机划分存在完全重复图跨集合；以本次分组划分的复跑记录为准。

## 当前数据与实验事实

- 原始数据集：七类 `0 4 7 8 A D H`，每类 1,016 张 128×128 灰度图，共 7,112 张。
- 原芯片 BMP 为 990×367 RGB 图像，不使用旧草稿的 “16-bit chip image” 图注。
- 完全相同的解码像素按 SHA-256 分组，同组只能进入一个集合。
- 每类训练 660、验证 102、测试 254；训练与验证共同占 75%，测试占 25%；seed 42。
- 标准化、PCA 只在训练集拟合，所有选参只使用验证集，芯片图不用于训练。
- CNN 完成指定轮数后恢复验证最优权重，不是提前终止训练。

所有最新数值与审核结果见 [实验复跑记录](experiment_results_20261014.md)，由实际 JSON 自动生成。不要再引用旧版本的 99.10%/99.21% 或约 1400 倍差距作为当前实验结论。

## 报告插图

- 图像处理：`output/segmentation/01_original.png` 至 `13_projection.png`。
- CNN：`output/task7/cnn/preprocessing.png`、`training_curves.png`、`confusion_matrix.png`、`errors.png`。
- SVM：`output/task7/baseline/pca_variance.png`、`confusion_matrix.png`、`errors.png`。
- 比较：`output/task7/comparison/accuracy_bars.png`、`per_class_f1.png`、`training_time.png`、`latency.png`。
- 端到端：`output/task7/recognition/label_normalized.png`、`label_gray.png`、`label_gray_enhanced.png`。
- 切分敏感性：`output/task7/verification/segmentation_grid.png`。

## 报告骨架

1. 封面：Group 6、成员姓名学号。
2. 问题与数据：任务、噪声和实拍图特点、重复数据检查、分组划分。
3. Task 1–6：各步算法、参数、流程图、阶段截图、切分参数敏感性。
4. Task 7-1：CNN 结构与训练、抽样数值梯度校验、预处理/超参数实验、错误分析。
5. Task 7-2：标准化/PCA/SVM 流程、验证集选参、错误分析。
6. Task 7-3：相同划分下的准确率、F1、完整模型训练计时、推理与文件大小。
7. 端到端实验：三种输入及局限；一张芯片图不能证明普遍鲁棒性。
8. 讨论与结论：由小组根据实际实验形成，区分事实和待验证假设。
9. 外部资源引用；附录给出环境和逐步复现说明。

格式：11pt Arial、单倍行距、最多 50 页，通常 30–40 页。正式报告、流程图、课程方法范围确认和最终提交包仍需小组完成。
