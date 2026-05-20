<!-- BILINGUAL README | 双语 README -->
<!-- Switch language: click the badge below / 切换语言：点击下方徽章 -->

<div align="right">

<!-- Language Toggle: Click the badge to jump to the other language / 点击徽章跳转至另一语言 -->
[🇬🇧 English](#dsa5203-project-3--scene-recognition-cnn) · [🇨🇳 中文](#dsa5203-项目三--场景识别cnn)

</div>

---

# DSA5203 Project 3 — Scene Recognition (CNN)

15-class scene classification on the Scene-15 dataset using a lightweight CNN trained from scratch.

**Results:** Best Validation Accuracy: **81.00%**

---

## File Structure

```
├── scene_recog_cnn.py   # Main script (train / test entry points)
├── model.py             # SceneCNN architecture
├── dataset.py           # Dataset loader and data augmentation
├── trained_cnn.pth      # Saved model weights (best checkpoint)
└── report.pdf           # Technical report
```

---

## Requirements

```bash
pip install torch torchvision pillow
```

---

## Usage

**Training**
```bash
python scene_recog_cnn.py --phase train \
    --train_data_dir ./data/train \
    --log_file ./log.txt \
    --model_dir trained_cnn.pth \
    --epochs 80
```

**Testing**
```bash
python scene_recog_cnn.py --phase test \
    --test_data_dir ./data/test \
    --model_dir trained_cnn.pth
```

---

## Data Format

Training and test directories must follow this structure:

```
data/train/
├── bedroom/
├── Coast/
├── Forest/
└── ...        # one sub-folder per class (15 total)
```

---

## Label Mapping

| Label | Class | Label | Class | Label | Class |
|-------|-------|-------|-------|-------|-------|
| 1 | bedroom | 6 | Insidecity | 11 | OpenCountry |
| 2 | Coast | 7 | kitchen | 12 | store |
| 3 | Forest | 8 | livingroom | 13 | Street |
| 4 | Highway | 9 | Mountain | 14 | Suburb |
| 5 | industrial | 10 | Office | 15 | TallBuilding |

---

## Acknowledgement

* **Course Project:** This repository originates from a course assignment for NUS DSA5203.
* **Data Statement:** The 1,500-image subset used for training and evaluation was provided as part of the course materials and is not publicly distributed in this repository. It is a sampled subset of the publicly available Scene-15 dataset.
* **Third-Party Code:** The data augmentation pipeline in `dataset.py` is adapted from the [MAE](https://github.com/facebookresearch/mae) open-source project by Meta Platforms, Inc.

---
---

<div align="right">

[🇬🇧 English](#dsa5203-project-3--scene-recognition-cnn) · [🇨🇳 中文](#dsa5203-项目三--场景识别cnn)

</div>

---

# DSA5203 项目三 — 场景识别（CNN）

基于轻量级卷积神经网络的 15 类场景图像分类，在 Scene-15 数据集上从头训练。

**结果：** 最佳验证集准确率：**81.00%**

---

## 文件结构

```
├── scene_recog_cnn.py   # 主脚本（训练 / 测试入口）
├── model.py             # SceneCNN 网络架构
├── dataset.py           # 数据集加载与数据增强
├── trained_cnn.pth      # 模型权重（最佳检查点）
└── report.pdf           # 技术报告
```

---

## 环境依赖

```bash
pip install torch torchvision pillow
```

---

## 使用方法

**训练**
```bash
python scene_recog_cnn.py --phase train \
    --train_data_dir ./data/train \
    --model_dir trained_cnn.pth \
    --epochs 80
```

**测试**
```bash
python scene_recog_cnn.py --phase test \
    --test_data_dir ./data/test \
    --model_dir trained_cnn.pth
```

---

## 数据目录格式

训练集与测试集目录需按如下结构组织：

```
data/train/
├── bedroom/
├── Coast/
├── Forest/
└── ...        # 每个类别一个子文件夹，共 15 类
```

---

## 标签映射

| 标签 | 类别 | 标签 | 类别 | 标签 | 类别 |
|------|------|------|------|------|------|
| 1 | bedroom | 6 | Insidecity | 11 | OpenCountry |
| 2 | Coast | 7 | kitchen | 12 | store |
| 3 | Forest | 8 | livingroom | 13 | Street |
| 4 | Highway | 9 | Mountain | 14 | Suburb |
| 5 | industrial | 10 | Office | 15 | TallBuilding |

---

## 声明 (Acknowledgement)

* **课程项目：** 本项目最初为 NUS DSA5203 课程的作业项目。
* **数据声明：** 用于训练和验证的 1500 张图像样本为课程内部提供的教学资料，故未在本仓库中公开分发。该数据是公开数据集 Scene-15 的一个子集。
* **第三方代码：** `dataset.py` 中的数据增强流水线参考并改编自 Meta Platforms, Inc. 开源的 [MAE](https://github.com/facebookresearch/mae) 项目。
