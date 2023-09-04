'''
-*- ecoding: utf-8 -*-
@Enviroment: ConcreteML
@ModuleName: Enc_Iris
@Author: Sakura
@Time: 2023/9/2 16:19
@Software: PyCharm
功能描述: 使用TFHE加密算法和Iris数据集，训练前馈神经网络模型
实现步骤:
结果：
'''
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import brevitas.nn as qnn
import numpy
# 加载Iris数据集
iris = load_iris()
X, y = iris.data, iris.target

# 数据预处理
scaler = StandardScaler()
X = scaler.fit_transform(X)  # 数据标准化
y = torch.tensor(y, dtype=torch.int64)  # 将标签转换为PyTorch的张量类型

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建前馈神经网络模型
class FFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FFNN, self).__init__()
        self.fc1=nn.Linear(input_dim,hidden_dim)
        self.relu = qnn.QuantReLU(bit_width=4)
        self.fc2=nn.Linear(hidden_dim,output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 模型参数
input_dim = 4
hidden_dim = 64
output_dim = 3

# 创建模型实例
model = FFNN(input_dim, hidden_dim, output_dim)

#编译模型
from concrete.ml.torch.compile import compile_brevitas_qat_model
quantized_module = compile_brevitas_qat_model(
    model, # our model
    X_train, # a representative input-set to be used for both quantization and compilation
)
print(quantized_module)
print("最大量化位数", quantized_module.fhe_circuit.graph.maximum_integer_bit_width())


# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    # inputs = torch.tensor(X_train, dtype=torch.float32)
    labels = y_train

    optimizer.zero_grad()
    # outputs = model(inputs)
    outputs = quantized_module.forward(X_train, fhe="simulate")
    outputs= torch.tensor(outputs, dtype=torch.float32)
    outputs.requires_grad_()
    loss = criterion(outputs, labels)
    #未对outputs启用梯度追踪，所以反向传播这么写会报错
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# 评估模型（前向传播）
# 运行时间
import time
start = time.time()
with torch.no_grad():
    y_pred = quantized_module.forward(X_test, fhe="simulate")
    labels = y_test
    outputs= torch.tensor(y_pred)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == labels).sum().item() / labels.size(0)
    print(f'Test Accuracy: {accuracy:.4f}')
end = time.time()
print("运行时间：", end - start)