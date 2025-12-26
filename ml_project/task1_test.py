"""
任务1: 测试传统机器学习模型
"""

import os
import sys
import argparse
import numpy as np
import pickle

from src.utils import set_seed, load_json, plot_confusion_matrix
from src.evaluator import Evaluator
from task1_train import load_features_and_labels, normalize_features


def main(args):
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载类别信息
    selected_classes_path = os.path.join(args.checkpoint_dir, 'selected_classes.json')
    class_info = load_json(selected_classes_path)
    selected_classes = class_info['classes']
    class_to_idx = class_info['class_to_idx']
    
    print(f"测试类别: {len(selected_classes)} 个")
    
    # 加载测试数据
    X_train_dummy, _ = load_features_and_labels(
        args.data_dir, 'train', selected_classes, class_to_idx
    )
    X_test, y_test = load_features_and_labels(
        args.data_dir, 'val', selected_classes, class_to_idx
    )
    
    # 标准化
    _, X_test = normalize_features(X_train_dummy, X_test)
    
    # 测试每个模型
    for model_name in args.models:
        print("\n" + "="*80)
        print(f"测试 {model_name.upper()} 模型")
        print("="*80)
        
        # 加载模型
        model_path = os.path.join(args.checkpoint_dir, model_name, 'model.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"已加载模型: {model_path}")
        
        # 评估
        metrics_dict, y_pred = Evaluator.evaluate_traditional_model(
            model, X_test, y_test, selected_classes
        )
        
        # 绘制混淆矩阵
        cm_path = os.path.join(args.checkpoint_dir, model_name, 'confusion_matrix.png')
        plot_confusion_matrix(
            metrics_dict['confusion_matrix'],
            selected_classes,
            save_path=cm_path,
            title=f'{model_name.upper()} Confusion Matrix'
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='任务1: 测试传统机器学习模型')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='数据集根目录')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/task1',
                        help='模型检查点目录')
    parser.add_argument('--models', nargs='+', default=['knn', 'softmax', 'svm'],
                        choices=['knn', 'softmax', 'svm'],
                        help='要测试的模型列表')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    main(args)