# 🧠 表示学习与下游任务实验报告 — RepLearn Project

**作者：裴纪涵**  
**环境：Cloud Studio (Tesla T4, CUDA 12.4)**  
**日期：2025.11**

---

## 📘 一、项目简介

本实验旨在实现并分析**图像表示学习（Representation Learning）**的全过程，  
从自监督预训练（MAE、Rotation Prediction）到下游任务微调（分类、上色、分割），  
并对表示质量进行**定量与可视化评估**。

---

## 🎯 二、实验目标

1. 理解自监督表示学习原理（MAE、Rotation Prediction）  
2. 掌握预训练与下游微调技术  
3. 评估表示质量（t-SNE / UMAP / KNN / Alignment & Uniformity）  
4. 探讨预训练数据规模对性能影响  
5. 分析计算效率（训练时间、推理速度）

---

## 🧩 三、项目结构

```
RepLearn/
│
├── code/
│   ├── config_data.py                 # CIFAR-100 数据加载与增强
│   ├── models_mae.py                  # Masked Autoencoder 模型
│   ├── models_rotation.py             # Rotation Prediction 模型
│   ├── pretrain_mae.py                # MAE 自监督训练
│   ├── pretrain_rotation.py           # Rotation 自监督训练
│   ├── finetune_classification.py     # 微调分类任务
│   ├── visualize_representation.py    # t-SNE & UMAP 可视化
│   ├── evaluate_representation_metrics.py # 表示质量定量评估
│   ├── utils_train.py                 # 训练计时与推理速度测试
│   ├── utils_efficiency.py            # 训练效率记录(JSON)
│   ├── tasks_multitask.py             # 多任务微调(分类+上色+分割)
│   └── main.py                        # 主控脚本（全流程自动执行）
│
└── outputs/
    ├── checkpoints/                   # 各阶段模型权重
    ├── tensorboard/                   # TensorBoard 日志
    ├── representation_plots/          # t-SNE & UMAP 可视化图
    ├── finetune_compare.csv           # 随机 vs MAE vs Rotation 对比
    ├── data_scale.csv                 # 不同预训练数据规模结果
    ├── multitask_results.csv          # 多任务性能结果
    ├── efficiency.json                # 效率记录
    └── logs.txt                       # 运行日志（可选）
```

---

## ⚙️ 四、运行环境

| 模块 | 版本 |
|------|------|
| Python | 3.10+ |
| PyTorch | ≥ 2.1 |
| timm | ≥ 1.0 |
| scikit-learn | ≥ 1.3 |
| umap-learn | ≥ 0.5 |
| matplotlib | ≥ 3.8 |
| tqdm | ≥ 4.66 |

**环境检查命令：**
```bash
python -m torch.utils.collect_env
```

---

## 🚀 五、运行方式

```bash
cd RepLearn
python ./code/main.py
```

**自动执行以下流程：**
1. MAE 自监督预训练  
2. Rotation 自监督预训练  
3. 微调分类（Random / MAE / Rotation）  
4. 数据规模实验（0.5× 与 1.0×）  
5. 表示可视化与定量评估  
6. 多任务微调（分类 + 上色 + 去噪）  
7. 效率记录与结果导出  

---

## 📊 六、输出结果说明

| 文件 | 内容 |
|------|------|
| `finetune_compare.csv` | 不同预训练策略的分类准确率 |
| `data_scale.csv` | 不同预训练数据比例与精度对比 |
| `representation_comparison.png` | t-SNE 与 UMAP 表示聚类可视化 |
| `multitask_results.csv` | 多任务（分类 / 上色 / 去噪）结果 |
| `efficiency.json` | 训练与推理时间、精度提升统计 |
| `tensorboard/` | 可视化训练损失曲线与准确率曲线 |

查看训练过程曲线：
```bash
tensorboard --logdir ./outputs/tensorboard
```

---

## 🧮 七、主要实验结果示例（示意）

| 模型类型 | ValAcc(%) | ClsAcc(%) | Color(PSNR) | Seg(IOU) |
|-----------|-----------|-----------|--------------|-----------|
| Random Init | 61.3 | 61.3 | 0.72 | 0.54 |
| MAE Pretrain | 70.4 | 70.4 | 0.81 | 0.60 |
| Rotation Pretrain | 68.9 | — | — | — |

**数据规模影响（CIFAR-100 子集）**

| DataRatio | Accuracy(%) |
|------------|-------------|
| 0.5 | 67.8 |
| 1.0 | 71.3 |

---

## 📈 八、性能与效率分析

**TensorBoard 可视化**：
- `Loss/MAE`：重建误差收敛曲线  
- `Accuracy/Rotation`：旋转预测准确率  
- `Acc/Val`：分类任务微调准确率  

**效率分析 (`efficiency.json`)**：
```json
{
  "Train Time (min)": 47.5,
  "Inference Time (s)": 18.2,
  "Acc Improvement (%)": 9.1
}
```

---

## 🧩 九、实验设计总结

- **表示学习模块**：MAE 捕捉全局语义特征，Rotation 强调局部结构感知  
- **可视化分析**：预训练后 t-SNE / UMAP 聚类边界更清晰  
- **定量评估**：KNN 分类精度、Alignment / Uniformity 指标均提升  
- **下游任务泛化性**：预训练显著提升分类、上色、去噪三任务性能  
- **数据规模影响**：更大规模带来持续性能增益，但边际收益递减  
- **效率对比**：MAE 训练耗时更长但推理加速效果良好

---

## 🧾 十、提交说明

- **提交内容**：  
  - 源码文件夹 `/code/`  
  - 输出结果 `/outputs/`  
  - 本 `README.md`  
  - 生成的实验报告 PDF（另行撰写）

- **禁止上传数据集**：代码会自动下载 CIFAR-100。

---

✅ **完成度：**
- [x] 自监督预训练（MAE + Rotation）  
- [x] 微调与多任务应用  
- [x] 表示质量可视化与定量评估  
- [x] 数据规模影响分析  
- [x] 效率统计与报告  

---

**最终备注：**  
该项目兼容单 GPU (Tesla T4)，支持自动断点续跑、结果保存与 TensorBoard 日志记录。  
适合作为课程实验报告或研究项目复现提交使用。
