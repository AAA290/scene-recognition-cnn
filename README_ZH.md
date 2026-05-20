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
