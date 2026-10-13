# ME5411 芯片字符识别项目

本项目按 ME5411 Group Project 的七个任务实现：先做图像处理与字符切分（Task 1–6），再用
两套方法做字符分类（Task 7）。代码分为两条互相独立的流水线，全部结果写入 `output/`。

## 环境与运行

当前进度、结果审核注意事项及四人分工见 [进度与分工](docs/progress_and_roles.md)。

课程原始数据包和题目 PDF 不随 Git 仓库上传。请从课程渠道取得数据，并确保存在
`dataset_2026_updated/dataset_2026/0/` 等七个类别目录和同目录下的 `Test_Image_2026.bmp`。
仓库包含保存的模型与实验结果；`output/task7/cache/` 为运行时自动生成的缓存。

在项目目录下执行（需要 Python 3.10+，实测 3.14）：

```powershell
python -m pip install -r requirements.txt
```

### Task 1–6：图像处理与字符切分

```powershell
python segment_characters.py
```

默认输入 `dataset_2026_updated/dataset_2026/Test_Image_2026.bmp`，输出 `output/segmentation/`。
参数可调（ROI、平滑尺度、阈值、最小面积、输出目录）：

```powershell
python segment_characters.py --blur 3 --background 25 --threshold 10
python segment_characters.py --roi 10 185 980 350 --output output/alternative
```

### Task 7：字符分类

```powershell
python run_task7.py            # 完整实验：CNN、PCA+SVM、对比、端到端识别
python run_task7.py --quick    # 每类 60 张、少量 epoch，仅用于快速自检
```

也可以只跑其中一步：

```powershell
python -m task7.train_cnn      # Task 7-1：从零实现的 NumPy CNN + 预处理/超参数实验
python -m task7.baseline       # Task 7-2：PCA + RBF SVM
python -m task7.compare        # Task 7-3：对比表格与图
python -m task7.recognize      # 用两套模型识别 Task 6 切出的芯片字符
```

三个训练脚本共享 `output/task7/split.json` 记录的同一份 75/25 划分，保证对比公平。
`--per-class N` 只用于自检，不会覆盖正式划分文件。

## Task 1–6：处理流程

1. 保存原图、灰度对比度拉伸演示及 3×3、5×5、7×7 均值滤波对比图。
2. 根据固定图像选取 ROI `(10,185,980,350)`，坐标右、下边界不包含在裁剪内。
3. 对原始灰度 ROI 做 sigma=3 的高斯平滑，减去 sigma=25 的高斯背景估计，保留亮字符相对于局部背景的响应。
4. 对局部响应使用阈值 10。连通域至少包含 100 个前景像素，高度至少为 ROI 高度的 40%，去除小噪点。
5. 从较窄连通域估计典型字符宽度。宽度超过其 1.45 倍的连通域，按宽度估计包含字符数量，并在预期间隔附近寻找垂直投影低谷切开。
6. 由左到右编号，保存原灰度裁剪、白字黑底二值裁剪、64×64 黑字白底归一化图。归一化保留宽高比并补边，缩放使用抗锯齿，因此归一化图包含灰度值。

均值滤波对比图是实验输出。用于实际切分的是高斯平滑和背景扣除支路，并非将所有演示图依次串联。

### 主要结果（`output/segmentation/`）

- `11_annotated.png`：原图 ROI、字符框和位置编号。
- `12_character_sheet.png`：从左到右的二值字符图。
- `13_projection.png`：垂直前景投影，红线为粘连区域切分位置。
- `10_contours.png`：前景边界像素，包括字符孔洞边界。
- `characters/`：每个字符三种图像版本（`gray` 原灰度裁剪、`binary` 白字黑底、`normalized` 64×64 黑字白底），按位置命名，不标注类别。
- `segmentation.json`：参数、字符数量和原图/ROI 坐标。

切分结果为 10 个字符，顺序为 `H D 4 4 7 8 0 A 0 0`，与芯片上的 `HD44780A00` 一致。

## Task 7：两套分类方法

数据为 7 类字符（`0 4 7 8 A D H`），每类 1016 张 128×128 灰度图，共 7112 张。
按类别分层划分：25% 测试（1778 张，全程不参与调参）、10% 验证（714 张，用于选预处理与超参数）、
65% 训练（4620 张）。随机种子固定为 42。

### Task 7-1：从零实现的 CNN（`task7/cnn.py`）

只用 NumPy 实现：im2col 卷积、2×2 最大池化、全连接、dropout、softmax 交叉熵与 Adam。
结构为 `conv3×3 → pool → conv3×3 → pool → dense128 → dense7`，约 27 万参数。
反向传播用有限差分做了数值校验（`numerical_gradient_check`，float64 下最大相对误差约 1e-6）。

在验证集上比较了三种预处理（整幅缩放 / 仅裁剪 / 裁剪后补成正方形）和若干超参数
（学习率、通道数、是否做随机平移与对比度增强、输入尺寸）。

### Task 7-2：PCA + RBF SVM（`task7/baseline.py`）

标准化 → PCA 降维 → RBF 核 SVM。在验证集上依次选择预处理方式、PCA 维数，以及 `C`/`gamma` 网格，
再用同一份训练集重新拟合并在测试集上评估。

### Task 7-3：对比（`task7/compare.py`）

对比准确率、宏平均精确率/召回率/F1、每类 F1、训练时间、单张推理延迟、批推理延迟、
模型体积与参数量，并列出两个方法的主要混淆对。结果表格见
`output/task7/comparison/comparison.md` 与 `comparison.json`。

### 端到端识别（`task7/recognize.py`）

把 Task 6 切出的 10 个字符送入两个模型，输出识别字符串。测试三种输入：
`normalized`（切分后的干净二值字符）、`gray`（照片原始灰度裁剪）、
`gray_enhanced`（灰度裁剪经对比度拉伸 + 轻度去噪并用 Otsu 估计裁剪框）。
三种输入的差异量化了训练集（干净渲染字形）与实拍图之间的域差异。

## 目录结构

```
segment_characters.py      Task 1–6 的完整实现
run_task7.py               Task 7 总入口
task7/data.py              数据读取、分层划分、预处理
task7/cnn.py               从零实现的卷积网络与 Adam
task7/train_cnn.py         Task 7-1 实验与最终训练
task7/baseline.py          Task 7-2 PCA + SVM
task7/compare.py           Task 7-3 对比
task7/recognize.py         端到端识别
task7/metrics.py           指标与报告插图（仅用 Pillow 绘制）
output/segmentation/       Task 1–6 结果
output/task7/              Task 7 结果（模型、指标、图、对比表）
dataset_2026_updated/      提供的图像数据集
```

## 说明

- 本实现针对提供的固定图片，ROI 和阈值由观察该图选取。宽字符切分依赖字符宽度相近的假设，遇到不同字体、断笔或更多粘连时需要重新检查。框数量正确不等于字符内容正确，仍需检查字符图和后续识别结果。
- 项目要求的实验分析和报告结论须由小组根据实际结果形成；使用的 NumPy、Pillow、scikit-learn 以及其他外部资源应在报告中引用。
