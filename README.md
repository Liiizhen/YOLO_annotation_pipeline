# 数据标注与处理流水线

本项目用于流程化手动标注源数据集，执行数据增强，并将其分割为 YOLO 格式的训练集、验证集和测试集。

## 1. 项目构成

```text
-data/                  # 数据根目录
  ├── Apples/           # 原始图片文件夹 (类别1)
  ├── Bananas/          # 原始图片文件夹 (类别2)
  ├── Oranges/          # 原始图片文件夹 (类别3)
  ├── data.yaml         # 数据集配置文件
  └── train/            # [自动生成] 训练集 (images/labels)
  └── valid/            # [自动生成] 验证集 (images/labels)
  └── test/             # [自动生成] 测试集 (images/labels)

-src/
  ├── annotation_pipeline.py  # 核心流水线脚本 (标注 -> 分割 -> 增强)
  ├── train.py                # 训练脚本
  ├── validate.py             # 验证脚本
  └── diagram.ipynb           # 结果绘制
```

### data.yaml 配置格式
请确保 `data.yaml` 包含 `names` 字段，且名称与原始图片子文件夹名称严格一致：
```yaml
names:
  - Apples
  - Bananas
  - Oranges
nc: 3 
```

## 2. 使用方法

进入 src 文件夹终端：
```bash
cd ./src
```

### 2.1 运行流水线

脚本支持多种参数配置，以下是常用命令：

**1. 默认运行（标注 + 分割）**
按照默认比例 (train:0.7, val:0.2, test:0.1) 进行手动标注和分割，不进行数据增强。
```bash
python annotation_pipeline.py
```

**2. 启用数据增强**
使用 `--aug_ratio` 参数。例如：每张原图生成 2 张增强图（旋转、噪声、模糊等）。
*注：增强操作发生的数据集分割之后，防止数据泄漏。*
```bash
python annotation_pipeline.py --aug_ratio 2
```

**3. 跳过标注步骤**
如果你已经完成了标注（临时文件夹中有数据），或者只想重新分割/增强数据，可以使用 `--skip_labeling`。
```bash
python annotation_pipeline.py --skip_labeling --aug_ratio 1
```

**4. 自定义分割比例**
```bash
python annotation_pipeline.py --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
```

### 2.2 参数详解

| 参数 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `--aug_ratio` | int | 0 | 每张原图生成的增强图片数量。0 表示不增强。 |
| `--train_ratio` | float | 0.7 | 训练集比例 |
| `--val_ratio` | float | 0.2 | 验证集比例 |
| `--test_ratio` | float | 0.1 | 测试集比例 |
| `--skip_labeling` | flag | False | adding此参数将跳过手动标注阶段，直接处理现有临时数据。 |

## 3. 标注工具操作指南

当程序进入标注阶段，将会弹出 OpenCV 窗口。操作方法如下：

*   **鼠标左键 (点击两次)**：
    *   第一次点击：确定框的左上角（或一个角）。
    *   第二次点击：确定框的右下角（或对角），此时会自动绘制红色矩形框并暂存。
*   **Enter (回车键)**：
    *   确认当前图片的所有标注框，保存图片和标签，并自动切换到**下一张**图片。
    *   如果当前图片没有框，按回车将跳过保存，直接处理下一张。
*   **c 键 (Clear)**：
    *   清除当前图片上已画好的所有暂存框（重置当前图片）。
*   **s 键 (Skip)**：
    *   跳过当前图片（不保存任何内容），直接处理下一张。
*   **q 键 (Quit)**：
    *   直接退出整个程序。

## 4. 输出结果

脚本运行结束后，数据将生成在 `../data` 目录下：

*   `../data/train`
*   `../data/valid`
*   `../data/test`

每一个目录下均包含 `images` (图片) 和 `labels` (txt格式标签) 两个子文件夹，可直接用于 YOLO 模型训练。

