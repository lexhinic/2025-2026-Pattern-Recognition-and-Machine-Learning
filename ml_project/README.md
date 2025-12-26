# CUB-200 鸟类图像分类项目

基于 CUB-200 数据集的鸟类图像分类项目，包含传统机器学习方法和深度学习方法的实现与对比。

## 项目概述

本项目是一个机器学习课程大作业，主要包含两个任务：

- **任务1**: 使用传统机器学习方法（KNN、Softmax、SVM）进行图像分类
- **任务2**: 使用深度学习方法（SimpleCNN、ResNet 及其变体）进行图像分类

## 项目结构

```
ml_project/
├── task1_train.py          # 任务1训练脚本
├── task1_test.py           # 任务1测试脚本
├── task2_train.py          # 任务2训练脚本
├── task2_test.py           # 任务2测试脚本
├── requirements.txt        # 项目依赖
├── cub200_dataset.zip      # 数据集压缩包
├── data/                   # 数据目录
│   ├── train/              # 训练集
│   └── val/                # 验证集/测试集
├── src/                    # 源代码
│   ├── data_loader.py      # 数据加载器
│   ├── traditional_models.py  # 传统机器学习模型
│   ├── deep_models.py      # 深度学习模型
│   ├── trainer.py          # 训练器
│   ├── evaluator.py        # 评估器
│   ├── utils.py            # 工具函数
│   └── count_params.py     # 参数统计
├── scripts/                # 运行脚本
│   ├── run_task1.sh        # 任务1运行脚本
│   ├── run_task2.sh        # 任务2运行脚本（ResNet）
│   ├── run_task2_simplecnn.sh
│   ├── run_task2_smallresnet.sh
│   └── run_task2_largeresnet.sh
└── checkpoints/            # 模型检查点
    ├── task1/              # 任务1模型及结果
    └── task2/              # 任务2模型及结果
```

## 环境配置

### 依赖安装

```bash
pip install -r requirements.txt
```

### 依赖列表

- PyTorch >= 1.10.0
- TorchVision >= 0.11.0
- NumPy >= 1.19.0
- Pillow >= 8.0.0
- Matplotlib >= 3.3.0
- Seaborn >= 0.11.0
- tqdm >= 4.62.0
- scikit-learn >= 0.24.0
- SciPy >= 1.5.0

## 数据集

本项目使用 **CUB-200** 鸟类数据集，包含 200 个鸟类类别的图像。

数据集已预处理并存放在 `data/` 目录下：
- `data/train/`: 训练集图像
- `data/val/`: 验证集/测试集图像

## 快速开始

### 任务1: 传统机器学习方法

#### 训练模型并测试

```bash
# 使用脚本运行
bash scripts/run_task1.sh
```

### 任务2: 深度学习方法

#### 训练模型并测试

```bash
# 使用脚本运行 ResNet
bash scripts/run_task2.sh
```

#### 训练并测试其他模型

```bash
# SimpleCNN
bash scripts/run_task2_simplecnn.sh

# SmallResNet
bash scripts/run_task2_smallresnet.sh

# LargeResNet
bash scripts/run_task2_largeresnet.sh
```


## 模型介绍

### 任务1 - 传统机器学习模型

| 模型 | 描述 |
|------|------|
| **KNN** | K近邻分类器，基于欧氏距离的向量化实现 |
| **Softmax** | Softmax分类器，使用梯度下降训练 |
| **SVM** | 支持向量机分类器 |

### 任务2 - 深度学习模型

| 模型 | 描述 |
|------|------|
| **SimpleCNN** | 基础CNN模型（ResNet去掉残差连接的版本） |
| **ResNet** | 标准残差网络 |
| **SmallResNet** | 缩小版ResNet |
| **LargeResNet** | 更大版本的ResNet |

## 训练参数说明

### 任务2 主要训练参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--model` | resnet | 模型类型 |
| `--epochs` | 200 | 最大训练轮数 |
| `--batch_size` | 32 | 批次大小 |
| `--lr` | 0.01 | 初始学习率 |
| `--optimizer` | sgd | 优化器类型 |
| `--lr_scheduler` | cosine | 学习率调度策略 |
| `--label_smoothing` | 0.1 | 标签平滑系数 |
| `--weight_decay` | 5e-4 | 权重衰减 |
| `--early_stopping_patience` | 15 | 早停耐心值 |
| `--image_size` | 224 | 输入图像大小 |

## 输出结果

训练完成后，结果将保存在 `checkpoints/` 目录下：

- `best_model.pth`: 最佳模型权重
- `training_history.json`: 训练历史记录
- `training_curves.png`: 训练曲线图
- `confusion_matrix.png`: 混淆矩阵
- `config.json`: 训练配置