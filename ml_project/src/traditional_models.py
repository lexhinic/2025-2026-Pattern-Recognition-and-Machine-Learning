"""
传统机器学习模型实现
包括: KNN, Softmax分类器, SVM
"""

import numpy as np
from tqdm import tqdm


class KNNClassifier:
    """
    K近邻分类器
    """
    
    def __init__(self, k=5):
        """
        Args:
            k: 近邻数量
        """
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """
        训练KNN
        
        Args:
            X: 训练特征 (n_samples, n_features)
            y: 训练标签 (n_samples,)
        """
        self.X_train = X
        self.y_train = y
        print(f"KNN训练完成: {len(X)} 个训练样本, k={self.k}")
    
    def _compute_distances(self, X):
        """
        计算测试样本到所有训练样本的欧氏距离
        使用向量化计算提高效率
        
        Args:
            X: 测试特征 (n_test, n_features)
        
        Returns:
            distances: 距离矩阵 (n_test, n_train)
        """
        # 使用公式: ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x*y
        X_squared = np.sum(X**2, axis=1, keepdims=True)  # (n_test, 1)
        X_train_squared = np.sum(self.X_train**2, axis=1, keepdims=True).T  # (1, n_train)
        cross_term = X @ self.X_train.T  # (n_test, n_train)
        
        distances = np.sqrt(np.maximum(X_squared + X_train_squared - 2 * cross_term, 0))
        
        return distances
    
    def predict(self, X, batch_size=100):
        """
        预测
        
        Args:
            X: 测试特征 (n_samples, n_features)
            batch_size: 批次大小（避免内存溢出）
        
        Returns:
            y_pred: 预测标签 (n_samples,)
        """
        n_test = X.shape[0]
        y_pred = np.zeros(n_test, dtype=self.y_train.dtype)
        
        # 分批预测以节省内存
        for i in tqdm(range(0, n_test, batch_size), desc="KNN预测"):
            batch_end = min(i + batch_size, n_test)
            batch_X = X[i:batch_end]
            
            # 计算距离
            distances = self._compute_distances(batch_X)
            
            # 找到k个最近邻并投票
            for j in range(batch_X.shape[0]):
                # 修正：使用argsort获取最小的k个距离的索引
                k_nearest_indices = np.argsort(distances[j])[:self.k]
                k_nearest_labels = self.y_train[k_nearest_indices]
                
                # 投票决定类别
                y_pred[i + j] = np.bincount(k_nearest_labels).argmax()
        
        return y_pred


class SoftmaxClassifier:
    """
    Softmax分类器
    使用梯度下降训练
    """
    
    def __init__(self, learning_rate=0.01, reg=0.001, n_epochs=100, batch_size=128):
        """
        Args:
            learning_rate: 学习率
            reg: 正则化系数
            n_epochs: 训练轮数
            batch_size: 批次大小
        """
        self.lr = learning_rate
        self.reg = reg
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.W = None
        self.b = None
        self.loss_history = []
    
    def _softmax(self, scores):
        """
        Softmax函数
        
        Args:
            scores: (n_samples, n_classes)
        
        Returns:
            probs: (n_samples, n_classes)
        """
        # 减去最大值以提高数值稳定性
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs
    
    def fit(self, X, y):
        """
        训练Softmax分类器
        
        Args:
            X: 训练特征 (n_samples, n_features)
            y: 训练标签 (n_samples,)
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # 初始化权重和偏置
        self.W = np.random.randn(n_features, n_classes) * 0.01
        self.b = np.zeros(n_classes)
        
        print(f"开始训练Softmax分类器: {n_samples}样本, {n_features}特征, {n_classes}类")
        
        # 训练
        for epoch in range(self.n_epochs):
            # 随机打乱数据
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            n_batches = 0
            
            # 小批量梯度下降
            for i in range(0, n_samples, self.batch_size):
                batch_end = min(i + self.batch_size, n_samples)
                X_batch = X_shuffled[i:batch_end]
                y_batch = y_shuffled[i:batch_end]
                
                # 前向传播
                scores = X_batch @ self.W + self.b  # (batch_size, n_classes)
                probs = self._softmax(scores)
                
                # 计算损失
                batch_size_actual = X_batch.shape[0]
                correct_log_probs = -np.log(probs[range(batch_size_actual), y_batch] + 1e-10)
                data_loss = np.sum(correct_log_probs) / batch_size_actual
                reg_loss = 0.5 * self.reg * np.sum(self.W * self.W)
                loss = data_loss + reg_loss
                
                epoch_loss += loss
                n_batches += 1
                
                # 反向传播
                dscores = probs.copy()
                dscores[range(batch_size_actual), y_batch] -= 1
                dscores /= batch_size_actual
                
                # 计算梯度
                dW = X_batch.T @ dscores + self.reg * self.W
                db = np.sum(dscores, axis=0)
                
                # 更新参数
                self.W -= self.lr * dW
                self.b -= self.lr * db
            
            avg_loss = epoch_loss / n_batches
            self.loss_history.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {avg_loss:.4f}")
    
    def predict(self, X):
        """
        预测
        
        Args:
            X: 测试特征 (n_samples, n_features)
        
        Returns:
            y_pred: 预测标签 (n_samples,)
        """
        scores = X @ self.W + self.b
        return np.argmax(scores, axis=1)
    
    def predict_proba(self, X):
        """
        预测概率
        
        Args:
            X: 测试特征 (n_samples, n_features)
        
        Returns:
            probs: 预测概率 (n_samples, n_classes)
        """
        scores = X @ self.W + self.b
        return self._softmax(scores)


class SVMClassifier:
    """
    支持向量机分类器
    使用Hinge Loss和梯度下降
    采用one-vs-all策略
    """
    
    def __init__(self, learning_rate=0.01, reg=0.001, n_epochs=100, batch_size=128):
        """
        Args:
            learning_rate: 学习率
            reg: 正则化系数
            n_epochs: 训练轮数
            batch_size: 批次大小
        """
        self.lr = learning_rate
        self.reg = reg
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.W = None
        self.b = None
        self.loss_history = []
    
    def fit(self, X, y):
        """
        训练SVM分类器
        
        Args:
            X: 训练特征 (n_samples, n_features)
            y: 训练标签 (n_samples,)
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # 初始化权重和偏置
        self.W = np.random.randn(n_features, n_classes) * 0.01
        self.b = np.zeros(n_classes)
        
        print(f"开始训练SVM分类器: {n_samples}样本, {n_features}特征, {n_classes}类")
        
        # 训练
        for epoch in range(self.n_epochs):
            # 随机打乱数据
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            n_batches = 0
            
            # 小批量梯度下降
            for i in range(0, n_samples, self.batch_size):
                batch_end = min(i + self.batch_size, n_samples)
                X_batch = X_shuffled[i:batch_end]
                y_batch = y_shuffled[i:batch_end]
                
                # 前向传播
                scores = X_batch @ self.W + self.b  # (batch_size, n_classes)
                
                # 计算Hinge Loss
                batch_size_actual = X_batch.shape[0]
                correct_class_scores = scores[range(batch_size_actual), y_batch].reshape(-1, 1)
                
                # margin = max(0, s_j - s_yi + delta)
                margins = np.maximum(0, scores - correct_class_scores + 1.0)
                margins[range(batch_size_actual), y_batch] = 0
                
                data_loss = np.sum(margins) / batch_size_actual
                reg_loss = 0.5 * self.reg * np.sum(self.W * self.W)
                loss = data_loss + reg_loss
                
                epoch_loss += loss
                n_batches += 1
                
                # 反向传播
                # 计算梯度
                dscores = np.zeros_like(scores)
                # 对于违反margin的类别，梯度为1
                dscores[margins > 0] = 1
                # 对于正确类别，梯度为负的违反margin的类别数量
                num_violations = np.sum(margins > 0, axis=1)
                dscores[range(batch_size_actual), y_batch] -= num_violations
                dscores /= batch_size_actual
                
                # 计算权重梯度
                dW = X_batch.T @ dscores + self.reg * self.W
                db = np.sum(dscores, axis=0)
                
                # 更新参数
                self.W -= self.lr * dW
                self.b -= self.lr * db
            
            avg_loss = epoch_loss / n_batches
            self.loss_history.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {avg_loss:.4f}")
    
    def predict(self, X):
        """
        预测
        
        Args:
            X: 测试特征 (n_samples, n_features)
        
        Returns:
            y_pred: 预测标签 (n_samples,)
        """
        scores = X @ self.W + self.b
        return np.argmax(scores, axis=1)