import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
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
# 多元线性回归损失函数（MSE）
def linear_regression_loss(X, y, w):
    m = len(y)
    predictions = X.dot(w)
    loss = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return loss
# 多元线性回归梯度
def linear_regression_gradients(X, y, w):
    m = len(y)
    predictions = X.dot(w)
    gradients = (1 / m) * X.T.dot(predictions - y)
    return gradients
def train_linear_regression(X, y, optimizer, num_iterations=200, batch_size=32):
    # 初始化权重
    w = np.zeros(X.shape[1])
    loss_history = []
    for i in range(num_iterations):
        # 随机选取batch
        indices = np.random.choice(len(X), batch_size)
        X_batch = X[indices]
        y_batch = y[indices]
        grads = linear_regression_gradients(X_batch, y_batch, w)
        w = optimizer.update(w, grads)
        loss = linear_regression_loss(X_batch, y_batch, w)
        loss_history.append(loss)
    return w, loss_history
def load_data():
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y
def compare_solutions(X, y):
    optimizer = Optimizer(learning_rate=0.01, beta=0.9, epsilon=1e-8)
    w_gd, loss_history = train_linear_regression(X, y, optimizer, num_iterations=500)
    model = LinearRegression()
    model.fit(X, y)
    w_sklearn = model.coef_
    print("Gradient Descent Weights:", w_gd)
    print("Sklearn Weights:", w_sklearn)
    y_pred_gd = X.dot(w_gd)
    y_pred_sklearn = model.predict(X)
    mse_gd = mean_squared_error(y, y_pred_gd)
    mse_sklearn = mean_squared_error(y, y_pred_sklearn)
    print(f"Mean Squared Error (Gradient Descent): {mse_gd}")
    print(f"Mean Squared Error (Sklearn): {mse_sklearn}")
    return w_gd, w_sklearn, mse_gd, mse_sklearn
# def plot_convergence(X, y):
#     learning_rates = [0.01, 0.1, 0.5]  # 三种不同的学习率
#     plt.figure(figsize=(12, 8))
#     for lr in learning_rates:
#         optimizer = Optimizer(learning_rate=lr, beta=0.9, epsilon=1e-8)
#         w, loss_history = train_linear_regression(X, y, optimizer, num_iterations=200)
#         plt.plot(loss_history, label=f'Learning Rate = {lr}')
#         print(f'Final weights for learning rate {lr}: {w}')
#     plt.xlabel('Iterations')
#     plt.ylabel('Loss')
#     plt.title('Convergence with Different Learning Rates')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
X, y = load_data()
w_gd, w_sklearn, mse_gd, mse_sklearn = compare_solutions(X, y)
# plot_convergence(X, y)
