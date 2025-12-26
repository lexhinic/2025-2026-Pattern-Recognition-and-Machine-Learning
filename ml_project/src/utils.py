"""
工具函数模块
"""

import os
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def set_seed(seed=42):
    """设置随机种子以保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(directory):
    """确保目录存在"""
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_json(data, filepath):
    """保存JSON文件"""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, indent=4, fp=f)


def load_json(filepath):
    """加载JSON文件"""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_confusion_matrix(cm, class_names, save_path=None, title='Confusion Matrix'):
    """绘制混淆矩阵"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存至: {save_path}")
    plt.close()


def plot_training_curves(train_losses, train_accs, val_accs, save_path=None):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    ax1.plot(train_losses, label='Train Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(train_accs, label='Train Accuracy', linewidth=2)
    ax2.plot(val_accs, label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存至: {save_path}")
    plt.close()


def get_class_mapping(data_dir):
    """
    获取类别名称到索引的映射
    返回: class_to_idx (dict), idx_to_class (dict)
    """
    train_dir = os.path.join(data_dir, 'train')
    classes = sorted([d for d in os.listdir(train_dir) 
                     if os.path.isdir(os.path.join(train_dir, d))])
    
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}
    
    return class_to_idx, idx_to_class


class AverageMeter:
    """计算和存储平均值与当前值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count