'''
-*- ecoding: utf-8 -*-
@Enviroment: ConcreteML
@ModuleName: CompileQATModel
@Author: Sakura
@Time: 2023/7/21 13:13
@Software: PyCharm
功能描述:
实现步骤:
结果：
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import brevitas.nn as qnn
import time
import random
from concrete.ml.torch.compile import compile_brevitas_qat_model
import numpy

num_inputs = 2
num_examples = 100
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs,
                       dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                       dtype=torch.float32)
print(features[0], labels[0])

import torch.utils.data as Data

batch_size = 10
# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(features, labels)
# 随机读取小批量
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        # self.linear = nn.Linear(n_feature, 1)
        self.linear1 = qnn.QuantLinear(n_feature, 3, weight_bit_width=8, bias=False)
        self.linear2 = qnn.QuantLinear(3, 1, weight_bit_width=8, bias=False)

    # forward 定义前向传播
    def forward(self, x):
        x = self.linear1(x)
        x= torch.relu(x)
        x=self.linear2(x)
        return x

net = LinearNet(num_inputs)

quantized_module = compile_brevitas_qat_model(
    net, # our model
    features, # a representative input-set to be used for both quantization and compilation
)

x_test = numpy.array([numpy.random.randn(num_inputs)])

y_pred = quantized_module.forward(x_test, fhe="execute")

print(y_pred)