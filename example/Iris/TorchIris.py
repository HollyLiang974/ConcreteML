'''
-*- ecoding: utf-8 -*-
@Enviroment: ConcreteML
@ModuleName: Iris
@Author: Sakura
@Time: 2023/9/2 16:11
@Software: PyCharm
功能描述: 使用PyTorch和Iris数据集训练前馈神经网络
实现步骤:
结果：
'''
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.sigmoid= nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        return out

# 模型参数
input_dim = 4
hidden_dim = 64
output_dim = 3

# 创建模型实例
model = FFNN(input_dim, hidden_dim, output_dim)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    inputs = torch.tensor(X_train, dtype=torch.float32)
    labels = y_train

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 评估模型
with torch.no_grad():
    inputs = torch.tensor(X_test, dtype=torch.float32)
    labels = y_test
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == labels).sum().item() / labels.size(0)
    print(f'Test Accuracy: {accuracy:.4f}')
