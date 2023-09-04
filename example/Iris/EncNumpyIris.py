'''
-*- ecoding: utf-8 -*-
@Enviroment: ConcreteML
@ModuleName: EncNumpyIris
@Author: Sakura
@Time: 2023/9/2 16:19
@Software: PyCharm
功能描述:
1.使用Numpy训练Iris数据集，使用前馈神经网络模型进行训练
2.使用TFHE加密训练数据

实现步骤:
结果：
'''
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from concrete import fhe
# 加载Iris数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 将输出标签进行独热编码
encoder = OneHotEncoder(sparse=False)
y_one_hot = encoder.fit_transform(y.reshape(-1, 1))

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)


# 定义一个类来表示NumPy版的FFNN模型
class NumPyFFNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.weights1 = np.random.randn(input_dim, hidden_dim)
        self.bias1 = np.zeros(hidden_dim)
        self.weights2 = np.random.randn(hidden_dim, output_dim)
        self.bias2 = np.zeros(output_dim)

    def forward(self, x):
        out1 = np.dot(x, self.weights1) + self.bias1
        out1[out1 < 0] = 0  # ReLU activation function
        out2 = np.dot(out1, self.weights2) + self.bias2
        return out2
    def backward(self, x, y, learning_rate):
        # 前向传播
        out1 = np.dot(x, self.weights1) + self.bias1
        out1[out1 < 0] = 0  # ReLU activation function
        out2 = np.dot(out1, self.weights2) + self.bias2
        loss = np.mean((out2 - y) ** 2)  # 均方误差损失

        # 反向传播
        delta_out2 = 2 * (out2 - y) / len(x)
        delta_weights2 = np.dot(out1.T, delta_out2)
        delta_bias2 = np.sum(delta_out2, axis=0)
        delta_out1 = np.dot(delta_out2, self.weights2.T)
        delta_out1[out1 <= 0] = 0  # ReLU反向传播
        delta_weights1 = np.dot(x.T, delta_out1)
        delta_bias1 = np.sum(delta_out1, axis=0)

        # 更新参数
        self.weights2 -= learning_rate * delta_weights2
        self.bias2 -= learning_rate * delta_bias2
        self.weights1 -= learning_rate * delta_weights1
        self.bias1 -= learning_rate * delta_bias1

        return loss


# 模型参数
input_dim = X_train.shape[1]
hidden_dim = 64
output_dim = y_train.shape[1]

# 创建NumPy模型实例
numpy_model = NumPyFFNN(input_dim, hidden_dim, output_dim)

# 随机选择一批训练数据
batch_indices = np.random.choice(len(X_train), 32, replace=False)
x_batch = X_train[batch_indices]
y_batch = y_train[batch_indices]



#将输入到电路的数据量化为整数

# 假设原始数据集中的特征值范围是 [min_val, max_val]
min_val = np.min(x_batch)
max_val = np.max(x_batch)

# 选择整数范围 [-2^31, 2^31-1]
int_min = -2**7
int_max = 2**7 - 1

# 映射数据到整数范围
x_batch_mapped = ((x_batch - min_val) / (max_val - min_val)) * (int_max - int_min) + int_min

# 将特征值取整，如果需要
x_batch_integer = np.round(x_batch_mapped).astype(int)

# 前向传播函数
@fhe.compiler({"x": "encrypted"})
def enc_forward(x):
    out1 = np.dot(x, numpy_model.weights1.astype(np.int32)) + numpy_model.bias1.astype(np.int32)
    # out1[out1 < 0] = 0  # ReLU activation function
    # out2 = np.dot(out1, numpy_model.weights2) + numpy_model.bias2
    return out1.astype(np.int64)
#编译前向传播电路
enc_forward_circuit=enc_forward.compile(x_batch_integer)
print(enc_forward_circuit)

# # 训练模型
# learning_rate = 0.01
# num_epochs = 1000
#
# for epoch in range(num_epochs):
#     # 随机选择一批训练数据
#     batch_indices = np.random.choice(len(X_train), 32, replace=False)
#     x_batch = X_train[batch_indices]
#     y_batch = y_train[batch_indices]
#
#     # 执行一次前向传播和反向传播，并获得损失
#     loss = numpy_model.backward(x_batch, y_batch, learning_rate)
#
#     # 打印损失
#     if (epoch + 1) % 100 == 0:
#         print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss}')
#
# # 测试模型
#     # 前向传播
#     predictions = numpy_model.forward(X_test)
#
#     # 计算准确率
#     correct = (np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)).sum()
#     total = len(X_test)
#     accuracy = correct / total * 100
#
# print(f'Test Accuracy: {accuracy:.2f}%')
