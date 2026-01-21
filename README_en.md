# Data Annotation and Processing Pipeline

[中文](README.md) | [English](README_en.md)

This project is designed for the streamlined manual annotation of source datasets, execution of data augmentation, and splitting them into YOLO-formatted training, validation, and testing sets.

## 1. Project Structure

```text
-data/                  # Data root directory
  ├── Apples/           # Raw image folder (Class 1)
  ├── Bananas/          # Raw image folder (Class 2)
  ├── Oranges/          # Raw image folder (Class 3)
  ├── data.yaml         # Dataset configuration file
  └── train/            # [Auto-generated] Training set (images/labels)
  └── valid/            # [Auto-generated] Validation set (images/labels)
  └── test/             # [Auto-generated] Test set (images/labels)

-src/
  ├── annotation_pipeline.py  # Core pipeline script (Labeling -> Splitting -> Augmentation)
  ├── train.py                # Training script
  ├── validate.py             # Validation script
  └── diagram.ipynb           # Result visualization
```

### data.yaml Configuration Format
Please ensure `data.yaml` contains the `names` field, and the names strictly match the subfolder names of the raw images:
```yaml
names:
  - Apples
  - Bananas
  - Oranges
nc: 3 
```

## 2. Usage

Enter the src folder directory in terminal:
```bash
cd ./src
```

### 2.1 Running the Pipeline

The script supports various parameter configurations. Here are some common commands:

**1. Default Run (Labeling + Splitting)**
Perform manual labeling and splitting according to the default ratio (train:0.7, val:0.2, test:0.1) without data augmentation.
```bash
python annotation_pipeline.py
```

**2. Enable Data Augmentation**
Use the `--aug_ratio` parameter. For example: generate 2 augmented images per original image (rotation, noise, blur, etc.).
*Note: Augmentation occurs after dataset splitting to prevent data leakage.*
```bash
python annotation_pipeline.py --aug_ratio 2
```

**3. Skip Labeling Step**
If you have already completed labeling (data exists in temporary folders), or just want to re-split/augment data, you can use `--skip_labeling`.
```bash
python annotation_pipeline.py --skip_labeling --aug_ratio 1
```

**4. Custom Split Ratios**
```bash
python annotation_pipeline.py --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
```

### 2.2 Parameter Details

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--aug_ratio` | int | 0 | Number of augmented images generated per original image. 0 means no augmentation. |
| `--train_ratio` | float | 0.7 | Training set ratio |
| `--val_ratio` | float | 0.2 | Validation set ratio |
| `--test_ratio` | float | 0.1 | Test set ratio |
| `--skip_labeling` | flag | False | Adding this parameter will skip the manual labeling phase and directly process existing temporary data. |

## 3. Labeling Tool Operation Guide

When the program enters the labeling phase, an OpenCV window will pop up. The operation instructions are as follows:

*   **Left Mouse Click (Twice)**:
    *   First click: Determine the top-left corner (or one corner) of the bounding box.
    *   Second click: Determine the bottom-right corner (or diagonal corner). A red rectangle will automatically be drawn and cached.
*   **Enter Key**:
    *   Confirm all bounding boxes for the current image, save the image and label, and automatically switch to the **next** image.
    *   If the current image has no boxes, pressing Enter will skip saving and proceed to the next image.
*   **c Key (Clear)**:
    *   Clear all cached bounding boxes drawn on the current image (reset current image).
*   **s Key (Skip)**:
    *   Skip the current image (save nothing) and proceed to the next image.
*   **q Key (Quit)**:
    *   Directly exit the entire program.

## 4. Output Results

After the script finishes running, data will be generated in the `../data` directory:

*   `../data/train`
*   `../data/valid`
*   `../data/test`

Each directory contains two subfolders: `images` and `labels` (txt format), which can be directly used for YOLO model training.
