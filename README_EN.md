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