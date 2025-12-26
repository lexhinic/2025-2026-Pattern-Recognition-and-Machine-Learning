"""
任务2: 使用深度学习方法训练模型
"""

import os
import sys
import argparse
import torch

from src.utils import set_seed, ensure_dir, plot_training_curves, save_json
from src.data_loader import get_data_loaders_task2
from src.deep_models import get_model
from src.trainer import DeepLearningTrainer


def main(args):
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载数据
    print("\n加载数据...")
    train_loader, val_loader, test_loader, num_classes, class_names = get_data_loaders_task2(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        val_split=args.val_split
    )
    
    print(f"类别数: {num_classes}")
    print(f"训练样本数: {len(train_loader.dataset)}")
    print(f"验证样本数: {len(val_loader.dataset)}")
    print(f"测试样本数: {len(test_loader.dataset)}")
    
    # 创建模型
    print(f"\n创建模型: {args.model}")
    model = get_model(args.model, num_classes=num_classes)
    
    # 创建训练器
    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    trainer = DeepLearningTrainer(model, device=device)
    
    # 训练
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir,
        lr_scheduler_type=args.lr_scheduler,
        optimizer_type=args.optimizer,
        label_smoothing=args.label_smoothing,
        early_stopping_patience=args.early_stopping_patience  
    )
    
    # 保存训练历史
    history = {
        'train_losses': [float(x) for x in trainer.train_losses],
        'train_accs': [float(x) for x in trainer.train_accs],
        'val_accs': [float(x) for x in trainer.val_accs],
        'best_val_acc': float(trainer.best_val_acc)
    }
    history_path = os.path.join(args.save_dir, 'training_history.json')
    save_json(history, history_path)
    print(f"训练历史已保存至: {history_path}")
    
    # 绘制训练曲线
    curves_path = os.path.join(args.save_dir, 'training_curves.png')
    plot_training_curves(
        trainer.train_losses,
        trainer.train_accs,
        trainer.val_accs,
        save_path=curves_path
    )
    
    # 保存类别信息和训练配置
    class_info = {
        'num_classes': num_classes,
        'class_names': class_names
    }
    class_info_path = os.path.join(args.save_dir, 'class_info.json')
    save_json(class_info, class_info_path)
    
    config = {
        'model': args.model,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'optimizer': args.optimizer,
        'lr_scheduler': args.lr_scheduler,
        'label_smoothing': args.label_smoothing,
        'best_val_acc': float(trainer.best_val_acc)
    }
    config_path = os.path.join(args.save_dir, 'config.json')
    save_json(config, config_path)
    
    print("\n任务2训练完成！")
    print("注意: 训练过程使用从训练集划分的验证集")
    print("     最终评估请运行 task2_test.py 使用独立测试集")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='任务2: 深度学习方法训练')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='数据集根目录')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/task2',
                        help='模型保存目录')
    parser.add_argument('--model', type=str, default='simplecnn',
                        choices=['simplecnn', 'resnet', 'smallresnet', 'largeresnet'],  
                        help='模型类型')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='学习率')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam'],
                        help='优化器类型')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        choices=['step', 'cosine', 'none'],
                        help='学习率调度器')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='标签平滑系数')
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                        help='早停耐心值')
    parser.add_argument('--image_size', type=int, default=224,
                        help='输入图像大小')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='从训练集划分的验证集比例')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载的工作进程数')
    parser.add_argument('--no_cuda', action='store_true',
                        help='禁用CUDA')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    main(args)