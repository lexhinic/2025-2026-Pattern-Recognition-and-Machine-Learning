"""
数据加载模块
"""

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class CUBDataset(Dataset):
    """
    CUB-200 数据集类
    支持加载原始图像或预提取的特征向量
    """
    
    def __init__(self, root_dir, split='train', load_feature=False, 
                 selected_classes=None, transform=None, indices=None):
        """
        Args:
            root_dir: 数据集根目录
            split: 'train', 'val', 或 'test'
            load_feature: 是否加载预提取的特征 (.pt文件)
            selected_classes: 选择的类别列表 (用于任务1的10分类)
            transform: 图像变换
            indices: 样本索引列表（用于划分训练/验证集）
        """
        # 对于test，实际读取val目录
        actual_split = 'val' if split == 'test' else split
        self.root_dir = os.path.join(root_dir, actual_split)
        self.load_feature = load_feature
        self.transform = transform
        
        # 获取所有类别
        all_classes = sorted([d for d in os.listdir(self.root_dir) 
                             if os.path.isdir(os.path.join(self.root_dir, d))])
        
        # 如果指定了类别，只使用这些类别
        if selected_classes is not None:
            self.classes = [c for c in all_classes if c in selected_classes]
        else:
            self.classes = all_classes
        
        # 创建类别到索引的映射
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # 加载所有样本
        self.samples = []
        self._load_samples()
        
        # 如果指定了索引，只使用这些样本
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]
        
        print(f"加载 {split} 集: {len(self.classes)} 类, {len(self.samples)} 样本")
    
    def _load_samples(self):
        """加载所有样本路径"""
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            # 获取所有jpg文件
            image_files = sorted([f for f in os.listdir(class_dir) if f.endswith('.jpg')])
            
            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)
                feature_path = img_path.replace('.jpg', '.pt')
                
                self.samples.append({
                    'image_path': img_path,
                    'feature_path': feature_path,
                    'label': class_idx,
                    'class_name': class_name
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = sample['label']
        
        if self.load_feature:
            feature = torch.load(sample['feature_path'])
            return feature, label
        else:
            image = Image.open(sample['image_path']).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
            
            return image, label
    
    def get_class_names(self):
        """返回类别名称列表"""
        return self.classes


def get_data_loaders_task1(data_dir, selected_classes, batch_size=32, num_workers=4):
    """
    任务1数据加载器 - 加载预提取的特征向量
    """
    train_dataset = CUBDataset(
        root_dir=data_dir,
        split='train',
        load_feature=True,
        selected_classes=selected_classes
    )
    
    test_dataset = CUBDataset(
        root_dir=data_dir,
        split='test',
        load_feature=True,
        selected_classes=selected_classes
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader, train_dataset.get_class_names()


def get_data_loaders_task2(data_dir, batch_size=32, num_workers=4, image_size=224, val_split=0.1):
    """
    任务2数据加载器 - 加载原始图像
    从训练集中划分验证集，原val目录作为测试集
    
    Args:
        data_dir: 数据集根目录
        batch_size: 批次大小
        num_workers: 工作进程数
        image_size: 图像大小
        val_split: 从训练集中划分的验证集比例
    
    Returns:
        train_loader, val_loader, test_loader, num_classes, class_names
    """
    # 数据变换
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    temp_dataset = CUBDataset(
        root_dir=data_dir,
        split='train',
        load_feature=False,
        transform=None
    )
    
    # 划分训练/验证索引
    dataset_size = len(temp_dataset)
    indices = list(range(dataset_size))
    split_idx = int(np.floor(val_split * dataset_size))
    
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_indices = indices[split_idx:]
    val_indices = indices[:split_idx]
    
    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_indices)} 样本")
    print(f"  验证集: {len(val_indices)} 样本 (从训练集划分)")
    
    # 创建训练和验证数据集
    train_dataset = CUBDataset(
        root_dir=data_dir,
        split='train',
        load_feature=False,
        transform=train_transform,
        indices=train_indices
    )
    
    val_dataset = CUBDataset(
        root_dir=data_dir,
        split='train',
        load_feature=False,
        transform=val_test_transform,
        indices=val_indices
    )
    
    # 创建测试数据集（使用原val目录）
    test_dataset = CUBDataset(
        root_dir=data_dir,
        split='test',
        load_feature=False,
        transform=val_test_transform
    )
    
    print(f"  测试集: {len(test_dataset)} 样本 (独立测试集)\n")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    num_classes = len(temp_dataset.get_class_names())
    
    return train_loader, val_loader, test_loader, num_classes, temp_dataset.get_class_names()


def select_classes_for_task1(data_dir, n_classes=10, seed=42):
    """为任务1随机选择n个类别"""
    train_dir = os.path.join(data_dir, 'train')
    all_classes = sorted([d for d in os.listdir(train_dir) 
                         if os.path.isdir(os.path.join(train_dir, d))])
    
    np.random.seed(seed)
    selected_classes = np.random.choice(all_classes, n_classes, replace=False).tolist()
    
    print(f"\n为任务1选择的{n_classes}个类别:")
    for i, cls in enumerate(selected_classes, 1):
        print(f"  {i}. {cls}")
    print()
    
    return selected_classes