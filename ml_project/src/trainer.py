"""
深度学习模型训练器
支持早停机制防止过拟合
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from src.utils import AverageMeter, ensure_dir


class DeepLearningTrainer:
    """
    深度学习训练器
    """
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            model: 模型
            device: 设备 ('cuda' 或 'cpu')
        """
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0.0
        
        print(f"使用设备: {self.device}")
        print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    def train_epoch(self, train_loader, criterion, optimizer):
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            criterion: 损失函数
            optimizer: 优化器
        
        Returns:
            avg_loss: 平均损失
            avg_acc: 平均准确率
        """
        self.model.train()
        
        losses = AverageMeter()
        accs = AverageMeter()
        
        pbar = tqdm(train_loader, desc="训练中")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 计算准确率
            _, predicted = outputs.max(1)
            acc = predicted.eq(labels).sum().item() / labels.size(0) * 100
            
            # 更新统计
            losses.update(loss.item(), labels.size(0))
            accs.update(acc, labels.size(0))
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{accs.avg:.2f}%'
            })
        
        return losses.avg, accs.avg
    
    def validate(self, val_loader):
        """
        验证
        
        Args:
            val_loader: 验证数据加载器
        
        Returns:
            avg_acc: 平均准确率
        """
        self.model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="验证中"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        acc = 100. * correct / total
        return acc
    
    def train(self, train_loader, val_loader, n_epochs=100, lr=0.01, 
              weight_decay=1e-4, save_dir='checkpoints/task2', 
              lr_scheduler_type='cosine', optimizer_type='sgd',
              label_smoothing=0.1, early_stopping_patience=15):
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            n_epochs: 训练轮数
            lr: 学习率
            weight_decay: 权重衰减
            save_dir: 模型保存目录
            lr_scheduler_type: 学习率调度器类型
            optimizer_type: 优化器类型 ('sgd' 或 'adam')
            label_smoothing: 标签平滑系数
            early_stopping_patience: 早停耐心值
        """
        ensure_dir(save_dir)
        
        # 定义损失函数
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # 定义优化器
        if optimizer_type == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=lr, 
                                 momentum=0.9, weight_decay=weight_decay,
                                 nesterov=True)
        elif optimizer_type == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=lr, 
                                  weight_decay=weight_decay)
        else:
            raise ValueError(f"未知的优化器类型: {optimizer_type}")
        
        # 学习率调度器
        if lr_scheduler_type == 'step':
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[30, 60, 90], gamma=0.1
            )
        elif lr_scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_epochs, eta_min=1e-6
            )
        else:
            scheduler = None
        
        print("\n" + "="*80)
        print("开始训练")
        print("="*80)
        print(f"训练轮数: {n_epochs}")
        print(f"初始学习率: {lr}")
        print(f"优化器: {optimizer_type.upper()}")
        print(f"权重衰减: {weight_decay}")
        print(f"学习率调度: {lr_scheduler_type}")
        print(f"标签平滑: {label_smoothing}")
        print(f"早停耐心值: {early_stopping_patience} epochs")
        print("="*80 + "\n")
        
        # 早停
        patience_counter = 0
        best_epoch = 0
        
        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch+1}/{n_epochs}")
            print("-"*80)
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # 验证
            val_acc = self.validate(val_loader)
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            # 计算训练-验证gap
            train_val_gap = train_acc - val_acc
            
            # 打印统计信息
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch+1} 结果:")
            print(f"  训练损失: {train_loss:.4f}")
            print(f"  训练准确率: {train_acc:.2f}%")
            print(f"  验证准确率: {val_acc:.2f}%")
            print(f"  训练-验证Gap: {train_val_gap:.2f}%")
            print(f"  当前学习率: {current_lr:.6f}")
            
            # 过拟合警告
            if train_val_gap > 20:
                print(f"警告: 训练-验证gap过大 ({train_val_gap:.2f}%)，可能存在过拟合！")
            
            # 保存最佳模型和早停判断
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                best_epoch = epoch + 1
                patience_counter = 0  # 重置早停计数器
                
                best_model_path = os.path.join(save_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'train_acc': train_acc,
                    'train_loss': train_loss,
                }, best_model_path)
                print(f"保存最佳模型 (验证准确率: {val_acc:.2f}%)")
                print(f"早停计数器重置")
            else:
                patience_counter += 1
                print(f"验证准确率未提升 (早停: {patience_counter}/{early_stopping_patience})")
                
                # 早停触发
                if patience_counter >= early_stopping_patience:
                    print(f"\n" + "="*80)
                    print(f"早停触发！")
                    print(f"验证准确率已连续 {early_stopping_patience} 个epoch未提升")
                    print(f"最佳验证准确率: {self.best_val_acc:.2f}% (Epoch {best_epoch})")
                    print(f"当前Epoch: {epoch + 1}")
                    print("="*80)
                    break
            
            # 定期保存检查点
            if (epoch + 1) % 20 == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'train_acc': train_acc,
                    'train_loss': train_loss,
                }, checkpoint_path)
                print(f"保存检查点 (epoch_{epoch+1}.pth)")
            
            # 更新学习率
            if scheduler is not None:
                scheduler.step()
        
        # 训练结束总结
        print("\n" + "="*80)
        print("训练完成！")
        print("="*80)
        print(f"总训练轮数: {epoch + 1}/{n_epochs}")
        print(f"最佳验证准确率: {self.best_val_acc:.2f}% (Epoch {best_epoch})")
        print(f"最终训练准确率: {train_acc:.2f}%")
        print(f"最终验证准确率: {val_acc:.2f}%")
        print(f"最终训练-验证Gap: {train_acc - val_acc:.2f}%")
        
        if patience_counter >= early_stopping_patience:
            print(f"\n注意: 训练通过早停机制提前结束")
        else:
            print(f"\n注意: 训练完成所有 {n_epochs} 个epoch")
        
        print("="*80 + "\n")
    
    def load_checkpoint(self, checkpoint_path):
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"已加载检查点: {checkpoint_path}")
        
        if 'val_acc' in checkpoint:
            print(f"该检查点的验证准确率: {checkpoint['val_acc']:.2f}%")
        if 'train_acc' in checkpoint:
            print(f"该检查点的训练准确率: {checkpoint['train_acc']:.2f}%")
        if 'epoch' in checkpoint:
            print(f"该检查点的训练轮数: {checkpoint['epoch']}")