"""
模型评估模块
"""

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm


class Evaluator:
    """
    模型评估器
    """
    
    @staticmethod
    def compute_accuracy(y_true, y_pred):
        """
        计算准确率
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
        
        Returns:
            accuracy: 准确率 (%)
        """
        return np.mean(y_true == y_pred) * 100
    
    @staticmethod
    def compute_confusion_matrix(y_true, y_pred, num_classes):
        """
        计算混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            num_classes: 类别数
        
        Returns:
            cm: 混淆矩阵
        """
        return confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    @staticmethod
    def compute_metrics(y_true, y_pred, class_names):
        """
        计算详细的评估指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            class_names: 类别名称列表
        
        Returns:
            metrics_dict: 包含各种指标的字典
        """
        # 总体准确率
        accuracy = Evaluator.compute_accuracy(y_true, y_pred)
        
        # 混淆矩阵
        cm = Evaluator.compute_confusion_matrix(y_true, y_pred, len(class_names))
        
        # 每个类别的精确率、召回率、F1分数
        precision = np.zeros(len(class_names))
        recall = np.zeros(len(class_names))
        f1 = np.zeros(len(class_names))
        
        for i in range(len(class_names)):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            
            precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) \
                    if (precision[i] + recall[i]) > 0 else 0
        
        metrics_dict = {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_precision': np.mean(precision),
            'avg_recall': np.mean(recall),
            'avg_f1': np.mean(f1)
        }
        
        return metrics_dict
    
    @staticmethod
    def print_evaluation_report(metrics_dict, class_names):
        """
        打印评估报告
        
        Args:
            metrics_dict: 评估指标字典
            class_names: 类别名称列表
        """
        print("\n" + "="*80)
        print("评估报告")
        print("="*80)
        print(f"总体准确率: {metrics_dict['accuracy']:.2f}%")
        print("="*80)
        print(f"{'类别':<30} {'精确率':<12} {'召回率':<12} {'F1分数':<12}")
        print("-"*80)
        
        for i, name in enumerate(class_names):
            # 截断过长的类别名
            display_name = name if len(name) <= 28 else name[:25] + "..."
            print(f"{display_name:<30} {metrics_dict['precision'][i]:<12.4f} "
                  f"{metrics_dict['recall'][i]:<12.4f} {metrics_dict['f1'][i]:<12.4f}")
        
        print("-"*80)
        print(f"{'平均':<30} {metrics_dict['avg_precision']:<12.4f} "
              f"{metrics_dict['avg_recall']:<12.4f} {metrics_dict['avg_f1']:<12.4f}")
        print("="*80 + "\n")
    
    @staticmethod
    def evaluate_traditional_model(model, X_test, y_test, class_names):
        """
        评估传统机器学习模型
        
        Args:
            model: 训练好的模型
            X_test: 测试特征
            y_test: 测试标签
            class_names: 类别名称列表
        
        Returns:
            metrics_dict: 评估指标字典
            y_pred: 预测标签
        """
        print("\n开始评估模型...")
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 计算指标
        metrics_dict = Evaluator.compute_metrics(y_test, y_pred, class_names)
        
        # 打印报告
        Evaluator.print_evaluation_report(metrics_dict, class_names)
        
        return metrics_dict, y_pred
    
    @staticmethod
    def evaluate_deep_model(model, data_loader, device, class_names):
        """
        评估深度学习模型
        
        Args:
            model: 训练好的模型
            data_loader: 数据加载器
            device: 设备
            class_names: 类别名称列表
        
        Returns:
            metrics_dict: 评估指标字典
            y_pred: 预测标签
            y_true: 真实标签
        """
        print("\n开始评估模型...")
        
        model.eval()
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for images, labels in tqdm(data_loader, desc="评估中"):
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                _, predicted = outputs.max(1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # 计算指标
        metrics_dict = Evaluator.compute_metrics(y_true, y_pred, class_names)
        
        # 打印报告
        Evaluator.print_evaluation_report(metrics_dict, class_names)
        
        return metrics_dict, y_pred, y_true