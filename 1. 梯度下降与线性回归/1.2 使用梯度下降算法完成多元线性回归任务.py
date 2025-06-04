# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
# 多元线性回归损失函数（MSE）
def linear_regression_loss(X, y, w):
    m = len(y)
    predictions = X.dot(w)
    loss = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return loss
def linear_regression_gradients(X, y, w):
    m = len(y)
    predictions = X.dot(w)
    gradients = (1 / m) * X.T.dot(predictions - y)
    return gradients
# 优化器类（上一题已经写出）
class Optimizer:
    def __init__(self, learning_rate, beta, epsilon):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.momentum = 0
        self.squared_grads = 0
    def update(self, params, grads):
        self.momentum = self.beta * self.momentum + (1 - self.beta) * grads
        self.squared_grads = self.beta * self.squared_grads + (1 - self.beta) * (grads ** 2)
        adjusted_grads = self.momentum / (np.sqrt(self.squared_grads) + self.epsilon)
        params -= self.learning_rate * adjusted_grads
        return params
def generate_data(n_samples=100, n_features=2, noise=0.1):
    # 随机生成样本数量为n_samples，特征数量为n_features的输入数据
    X = np.random.rand(n_samples, n_features)
    true_weights = np.array([3.0, -2.0])  # 真实的参数
    y = X.dot(true_weights) + noise * np.random.randn(n_samples)  # 加入噪声
    return X, y

def train_linear_regression(X, y, optimizer, num_iterations=200, batch_size=32):
    w = np.zeros(X.shape[1])
    loss_history = []
    for i in range(num_iterations):
        #  随机选取batch!
        indices = np.random.choice(len(X), batch_size)
        X_batch = X[indices]
        y_batch = y[indices]
        grads = linear_regression_gradients(X_batch, y_batch, w)
        # 更新参数
        w = optimizer.update(w, grads)
        # 计算当前的损失
        loss = linear_regression_loss(X_batch, y_batch, w)
        loss_history.append(loss)

    return w, loss_history
# 画出不同学习率下的收敛情况
def plot_convergence(X, y):
    learning_rates = [0.01, 0.1, 0.5]  # 三种不同的学习率
    plt.figure(figsize=(12, 8))

    for lr in learning_rates:
        optimizer = Optimizer(learning_rate=lr, beta=0.9, epsilon=1e-8)
        w, loss_history = train_linear_regression(X, y, optimizer, num_iterations=200)
        # 绘制损失曲线
        plt.plot(loss_history, label=f'Learning Rate = {lr}')
        print(f'Final weights for learning rate {lr}: {w}')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Convergence with Different Learning Rates')
    plt.legend()
    plt.grid(True)
    plt.show()
# 生成数据
X, y = generate_data(n_samples=200, n_features=2, noise=0.1)
# 画出不同学习率下的收敛情况
plot_convergence(X, y)
