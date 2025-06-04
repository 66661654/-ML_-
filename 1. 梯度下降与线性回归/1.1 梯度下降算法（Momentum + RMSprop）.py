import numpy as np
import matplotlib.pyplot as plt
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
def loss_function(x, y):
    return np.sin(x) * np.cos(y)
def loss_gradients(x, y):
    grad_x = np.cos(x) * np.cos(y)  # 对 x 的偏导数
    grad_y = -np.sin(x) * np.sin(y)  # 对 y 的偏导数
    return np.array([grad_x, grad_y])
# 测试优化器并绘制 3D 路径图
def test_optimizer_3d():
    # 创建优化器，使用 Momentum 和 RMSprop
    optimizer = Optimizer(learning_rate=0.1, beta=0.9, epsilon=1e-8)
    # 初始化参数，选择一个更合适的初始点
    params = np.array([1.5, 1.5])  # 更好的起始点
    param_history = [params.copy()]  # 记录每一步的参数
    num_iterations = 50
    for _ in range(num_iterations):
        grads = loss_gradients(params[0], params[1])  # 计算梯度
        params = optimizer.update(params, grads)  # 更新参数
        param_history.append(params.copy())  # 保存历史路径
    param_history = np.array(param_history)  # 转换为 NumPy 数组
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = loss_function(X, Y)
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

    # 绘制下降路径
    path_x = param_history[:, 0]
    path_y = param_history[:, 1]
    path_z = loss_function(path_x, path_y)
    ax.plot(path_x, path_y, path_z, color='red', marker='o', label="Descent Path")
    ax.set_xlabel(r'$\theta_0$', fontsize=12)
    ax.set_ylabel(r'$\theta_1$', fontsize=12)
    ax.set_zlabel(r'$J(\theta_0, \theta_1)$', fontsize=12)
    ax.set_title("3D Gradient Descent Path with Momentum & RMSprop", fontsize=14)
    ax.legend()
    plt.show()
test_optimizer_3d()