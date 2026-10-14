"""Write a factual experiment summary from audited, freshly generated outputs."""
import json
from pathlib import Path

from .data import OUTPUT_ROOT, PROJECT_ROOT


def main():
    def read(name):
        return json.loads((OUTPUT_ROOT / name).read_text(encoding='utf-8'))
    audit = read('verification/data_audit.json')
    verified = read('verification/reloaded_models.json')
    cnn = read('cnn/metrics.json')
    svm = read('baseline/metrics.json')
    recognized = read('recognition/recognition.json')
    study = read('verification/segmentation_study.json')
    gradients = read('cnn/experiments.json')['gradient_check']
    lines = ['# 实验复跑与审核结果（2026-10-14）', '',
             '本文件由 `python -m task7.report_experiments` 从保存的 JSON 生成，记录实验事实；正式报告的解释与结论由小组完成。', '',
             '## 数据划分与旧结果的区别', '',
             f"- 数据共 7,112 张、7 类。原随机划分有 {audit['legacy_random_split']['cross_split_duplicate_groups']} 组完全重复图片跨集合，其中 {audit['legacy_random_split']['train_test_duplicate_groups']} 组同时出现在训练与测试集。",
             f"- 本次按解码像素 SHA-256 对重复图片分组，再按类别分层；跨集合完全重复图片为 {audit['cross_split_duplicate_groups']} 组。",
             '- 保持每类训练 660、验证 102、测试 254，即合计 4,620 / 714 / 1,778；训练与验证共同占 75%，测试占 25%。随机种子 42。',
             '- CNN 的七组配置和 SVM 的预处理/PCA/核参数选择均只使用验证集。芯片图片没有参与训练或分类器选择。',
             '- 旧模型、指标及源码已在本地 `tmp/experiments_before_20261014.zip` 备份，GitHub 历史也保留旧版本。当前结果基于新划分，不能与旧值当作同一测试集上的复现比较。',
             '- 像素哈希只能排除完全重复；未提供字体/作者分组元数据，近似重复和同字体跨集合仍是局限。', '',
             '## CNN 预处理与超参数实验', '',
             '| 配置 | 验证准确率 | 最优轮次 | 训练秒数（8 轮） |',
             '| --- | ---: | ---: | ---: |']
    for e in cnn['experiments']:
        lines.append(f"| {e['name']} | {e['val_accuracy']:.4f} | {e['best_epoch']} | {e['train_seconds']:.2f} |")
    lines += ['', f"最终选定 `{cnn['spec']['name']}`；完整训练 {cnn['epochs']} 轮，恢复第 {cnn['history']['best_epoch']} 轮的最佳验证权重。没有提前终止训练。",
              f"有限差分抽样梯度检查覆盖 {len(gradients)} 个参数张量，最大相对误差 {max(gradients.values()):.3g}。", '',
              '## 非 CNN 参数实验', '',
              '| 预处理 | 输入大小 | 验证准确率 |', '| --- | ---: | ---: |']
    for e in svm['study']['preprocessing']:
        lines.append(f"| {e['preprocessing']} | {e['size']} | {e['val_accuracy']:.4f} |")
    lines += ['', '| PCA 维数 | 验证准确率 | 解释方差 |', '| ---: | ---: | ---: |']
    for e in svm['study']['pca']:
        lines.append(f"| {e['pca_components']} | {e['val_accuracy']:.4f} | {e['explained_variance']:.4f} |")
    lines += ['', '| C | gamma | 验证准确率 |', '| ---: | --- | ---: |']
    for e in svm['study']['svm_grid']:
        lines.append(f"| {e['C']} | {e['gamma']} | {e['val_accuracy']:.4f} |")
    lines += ['', f"最终配置：`{svm['best']}`。标准化和 PCA 仅在训练集拟合。", '',
              '## 修正划分上的测试与效率结果', '',
              '| 指标 | CNN | PCA + SVM |', '| --- | ---: | ---: |',
              f"| 测试准确率 | {cnn['report']['accuracy']:.4f} | {svm['report']['accuracy']:.4f} |",
              f"| 宏平均 F1 | {cnn['report']['macro']['f1']:.4f} | {svm['report']['macro']['f1']:.4f} |",
              f"| 错误数 / 1778 | {verified['checks']['cnn']['errors']} | {verified['checks']['baseline']['errors']} |",
              f"| 最终模型训练秒数 | {cnn['train_seconds']:.2f} | {svm['train_seconds']:.2f} |",
              f"| 单张推理 ms | {cnn['timing']['ms_per_image_single']:.3f} | {svm['timing']['ms_per_image_single']:.3f} |",
              f"| 批推理 ms/张 | {cnn['timing']['ms_per_image_batched']:.3f} | {svm['timing']['ms_per_image_batched']:.3f} |",
              f"| 模型文件 bytes | {cnn['model_bytes']} | {svm['model_bytes']} |", '',
              '训练计时：两者均排除图像读盘/预处理和超参数搜索。SVM 包含标准化、PCA 拟合、变换和 SVC 拟合；CNN 包含拟合、逐轮验证及最优权重选择。这是各自实际训练流程成本，非相同优化算法工作量。', '',
              '推理计时：从已缩放图像张量到类别；SVM 包含标准化/PCA 变换。两者均预热，批大小 256，固定随机抽取 200 张单图，计时 3 次取中位数。原来的 SVM 计时遗漏已修正，不再使用旧的“约 1400 倍”说法。', '',
              '模型文件大小可直接对比；CNN 可训练参数与 SVM/PCA 存储状态没有等价的参数数量定义，因此删除旧参数量比值。', '',
              f"配对预测结果：`{verified['paired_outcomes']}`。这里只报告一个固定划分，不把小幅准确率差异解释为普遍优势。", '',
              '## 端到端芯片识别', '', '| 输入 | CNN | 正确字符 | PCA + SVM | 正确字符 |',
              '| --- | --- | ---: | --- | ---: |']
    for k, r in recognized.items():
        lines.append(f"| {k} | `{r['cnn_string']}` | {r['cnn_correct']}/10 | `{r['pca_svm_string']}` | {r['pca_svm_correct']}/10 |")
    lines += ['', '只有一张芯片图；这些结果用于检查输入差异，不足以说明对其他芯片图的普遍鲁棒性。SVM 展示分数不是校准概率。', '',
              '## 切分参数敏感性（固定 3×3 网格）', '',
              '| Gaussian sigma | 阈值 | 字符数 | CNN 字符串 | SVM 字符串 | 框 IoU 对默认值 |',
              '| ---: | ---: | ---: | --- | --- | ---: |']
    for r in study['settings']:
        iou = r.get('mean_box_iou_vs_default')
        text = f'{iou:.3f}' if iou is not None else '不适用'
        lines.append(f"| {r['blur']} | {r['threshold']} | {r['count']} | `{r['cnn_string']}` | `{r['pca_svm_string']}` | {text} |")
    lines += ['', '框 IoU 仅相对默认算法输出，未提供人工真实框，不能作为检测准确率。此实验不训练模型，也不根据芯片识别结果选择分类器配置。', '',
              '## 独立复核与复现命令', '',
              '重新加载两个模型，对未使用预处理缓存的测试图重新计算，预测数组、混淆矩阵和指标均与保存结果完全一致。测试集重复预测用于核查可复现性，没有据此选参。', '',
              '```powershell', "$env:OPENBLAS_NUM_THREADS='1'", "$env:OMP_NUM_THREADS='1'", "$env:MKL_NUM_THREADS='1'",
              'python -m task7.audit', 'python segment_characters.py', 'python run_task7.py',
              'python -m task7.verify_results', 'python -m task7.segmentation_study', 'python -m task7.report_experiments', '```', '',
              '环境、依赖版本、线程设置见 `output/task7/verification/data_audit.json`；训练日志见 `full_run.log`；源码 SHA-256 和模型复核见 `reloaded_models.json`。', '',
              f"本次环境：Python {audit['environment']['python'].split()[0]}；依赖 `{audit['environment']['versions']}`；处理器 `{audit['environment']['processor']}`。", '',
              '## 仍需小组完成', '',
              '- 确认 PCA/SVM 符合课程 Part 2 的方法范围。',
              '- 根据实验形成自己的解释与结论，完成流程图、参考文献、正式报告及成员信息。',
              '- 正式提交包与 Canvas 上传尚未完成。本次没有生成正式报告或代替小组完成项目分析。', '']
    target = PROJECT_ROOT / 'docs/experiment_results_20261014.md'
    target.write_text('\n'.join(lines), encoding='utf-8')
    print('Experiment summary written:', target)


if __name__ == '__main__':
    main()
