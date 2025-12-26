"""
任务1: 使用传统机器学习方法训练模型
"""

import os
import sys
import argparse
import numpy as np
import torch
import pickle
from tqdm import tqdm

from src.utils import set_seed, ensure_dir, save_json
from src.data_loader import select_classes_for_task1
from src.traditional_models import KNNClassifier, SoftmaxClassifier, SVMClassifier
from src.evaluator import Evaluator


def load_features_and_labels(data_dir, split, selected_classes, class_to_idx):
    """
    加载预提取的特征向量和标签
    
    Args:
        data_dir: 数据集根目录
        split: 'train' 或 'val'
        selected_classes: 选择的类别列表
        class_to_idx: 类别到索引的映射
    
    Returns:
        X: 特征矩阵 (n_samples, n_features)
        y: 标签数组 (n_samples,)
    """
    split_dir = os.path.join(data_dir, split)
    
    features = []
    labels = []
    
    print(f"\n加载 {split} 集的特征...")
    for class_name in tqdm(selected_classes):
        class_dir = os.path.join(split_dir, class_name)
        class_idx = class_to_idx[class_name]
        
        # 获取所有.pt文件
        pt_files = [f for f in os.listdir(class_dir) if f.endswith('.pt')]
        
        for pt_file in pt_files:
            pt_path = os.path.join(class_dir, pt_file)
            
            # 加载特征
            feature = torch.load(pt_path)
            if isinstance(feature, torch.Tensor):
                feature = feature.cpu().numpy()
            
            # 展平为一维
            feature = feature.flatten()
            
            features.append(feature)
            labels.append(class_idx)
    
    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    
    print(f"加载完成: {X.shape[0]} 个样本, 特征维度: {X.shape[1]}")
    
    return X, y


def normalize_features(X_train, X_test):
    """
    标准化特征
    
    Args:
        X_train: 训练特征
        X_test: 测试特征
    
    Returns:
        X_train_norm: 标准化后的训练特征
        X_test_norm: 标准化后的测试特征
    """
    # 计算训练集的均值和标准差
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    
    # 标准化
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    
    return X_train_norm, X_test_norm


def train_model(model_name, X_train, y_train, X_test, y_test, class_names, save_dir):
    """
    训练并评估模型
    
    Args:
        model_name: 模型名称 ('knn', 'softmax', 'svm')
        X_train: 训练特征
        y_train: 训练标签
        X_test: 测试特征
        y_test: 测试标签
        class_names: 类别名称列表
        save_dir: 保存目录
    
    Returns:
        metrics_dict: 评估指标字典
    """
    print("\n" + "="*80)
    print(f"训练 {model_name.upper()} 模型")
    print("="*80)
    
    # 创建模型
    if model_name == 'knn':
        model = KNNClassifier(k=5)
    elif model_name == 'softmax':
        model = SoftmaxClassifier(learning_rate=0.1, reg=0.001, n_epochs=100, batch_size=128)
    elif model_name == 'svm':
        model = SVMClassifier(learning_rate=0.1, reg=0.001, n_epochs=100, batch_size=128)
    else:
        raise ValueError(f"未知的模型名称: {model_name}")
    
    # 训练
    model.fit(X_train, y_train)
    
    # 评估
    metrics_dict, y_pred = Evaluator.evaluate_traditional_model(
        model, X_test, y_test, class_names
    )
    
    # 保存模型
    model_save_dir = os.path.join(save_dir, model_name)
    ensure_dir(model_save_dir)
    
    model_path = os.path.join(model_save_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"模型已保存至: {model_path}")
    
    # 保存评估结果
    results = {
        'accuracy': float(metrics_dict['accuracy']),
        'avg_precision': float(metrics_dict['avg_precision']),
        'avg_recall': float(metrics_dict['avg_recall']),
        'avg_f1': float(metrics_dict['avg_f1']),
    }
    results_path = os.path.join(model_save_dir, 'results.json')
    save_json(results, results_path)
    print(f"结果已保存至: {results_path}")
    
    return metrics_dict


def main(args):
    # 设置随机种子
    set_seed(args.seed)
    
    # 选择类别
    print("\n选择用于10分类的类别...")
    selected_classes = select_classes_for_task1(args.data_dir, n_classes=10, seed=args.seed)
    
    # 创建类别映射
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(selected_classes)}
    
    # 保存选择的类别
    ensure_dir(args.save_dir)
    selected_classes_path = os.path.join(args.save_dir, 'selected_classes.json')
    save_json({
        'classes': selected_classes,
        'class_to_idx': class_to_idx
    }, selected_classes_path)
    
    # 加载特征和标签
    X_train, y_train = load_features_and_labels(
        args.data_dir, 'train', selected_classes, class_to_idx
    )
    X_test, y_test = load_features_and_labels(
        args.data_dir, 'val', selected_classes, class_to_idx
    )
    
    # 标准化特征
    print("\n标准化特征...")
    X_train, X_test = normalize_features(X_train, X_test)
    
    # 训练所有模型
    all_results = {}
    
    for model_name in args.models:
        metrics = train_model(
            model_name, X_train, y_train, X_test, y_test, 
            selected_classes, args.save_dir
        )
        all_results[model_name] = {
            'accuracy': float(metrics['accuracy']),
            'avg_precision': float(metrics['avg_precision']),
            'avg_recall': float(metrics['avg_recall']),
            'avg_f1': float(metrics['avg_f1']),
        }
    
    # 保存汇总结果
    summary_path = os.path.join(args.save_dir, 'summary.json')
    save_json(all_results, summary_path)
    
    # 打印汇总
    print("\n" + "="*80)
    print("任务1 汇总结果")
    print("="*80)
    print(f"{'模型':<15} {'准确率':<12} {'精确率':<12} {'召回率':<12} {'F1分数':<12}")
    print("-"*80)
    for model_name, results in all_results.items():
        print(f"{model_name.upper():<15} {results['accuracy']:<12.2f} "
              f"{results['avg_precision']:<12.2f} {results['avg_recall']:<12.2f} "
              f"{results['avg_f1']:<12.2f}")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='任务1: 传统机器学习方法训练')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='数据集根目录')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/task1',
                        help='模型保存目录')
    parser.add_argument('--models', nargs='+', default=['knn', 'softmax', 'svm'],
                        choices=['knn', 'softmax', 'svm'],
                        help='要训练的模型列表')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    main(args)