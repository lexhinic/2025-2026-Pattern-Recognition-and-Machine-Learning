"""
任务2: 测试深度学习模型
使用独立的测试集进行最终评估
"""

import os
import sys
import argparse
import torch

from src.utils import set_seed, load_json, plot_confusion_matrix
from src.data_loader import get_data_loaders_task2
from src.deep_models import get_model
from src.trainer import DeepLearningTrainer
from src.evaluator import Evaluator


def main(args):
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载类别信息
    class_info_path = os.path.join(args.checkpoint_dir, 'class_info.json')
    class_info = load_json(class_info_path)
    num_classes = class_info['num_classes']
    class_names = class_info['class_names']
    
    print(f"类别数: {num_classes}")
    
    # 加载数据
    print("\n加载数据...")
    _, _, test_loader, _, _ = get_data_loaders_task2(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size
    )
    
    print(f"测试样本数: {len(test_loader.dataset)}")
    
    # 创建模型
    print(f"\n加载模型: {args.model}")
    model = get_model(args.model, num_classes=num_classes)
    
    # 创建训练器并加载检查点
    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    trainer = DeepLearningTrainer(model, device=device)
    
    checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
    trainer.load_checkpoint(checkpoint_path)
    
    # 在测试集上评估
    print("\n在独立测试集上评估模型...")
    metrics_dict, y_pred, y_true = Evaluator.evaluate_deep_model(
        model, test_loader, device, class_names
    )
    
    # 绘制混淆矩阵
    cm_path = os.path.join(args.checkpoint_dir, 'confusion_matrix_test.png')
    plot_confusion_matrix(
        metrics_dict['confusion_matrix'],
        class_names,
        save_path=cm_path,
        title=f'{args.model.upper()} Test Set Confusion Matrix'
    )
    
    print("\n任务2测试完成！")
    print(f"测试集准确率: {metrics_dict['accuracy']:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='任务2: 测试深度学习模型')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='数据集根目录')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/task2',
                        help='模型检查点目录')
    parser.add_argument('--model', type=str, default='resnet',
                        choices=['simplecnn', 'resnet', 'smallresnet', 'largeresnet'],
                        help='模型类型')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--image_size', type=int, default=224,
                        help='输入图像大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载的工作进程数')
    parser.add_argument('--no_cuda', action='store_true',
                        help='禁用CUDA')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    main(args)