'''
-*- ecoding: utf-8 -*-
@Enviroment: ConcreteML
@ModuleName: PTQSigmoid
@Author: Sakura
@Time: 2023/8/6 17:25
@Software: PyCharm
功能描述:
实现步骤:
结果：
'''

import time

import torch
import torch.nn as nn
from concrete.ml.torch.compile import compile_torch_model


class net(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(x)
        return x


input = torch.randn(10, 1)
model = net()


def testSigmoidPTQ(n_bits):
    # 模型编译时间
    start_time = time.time()
    quantized_module = compile_torch_model(
        model,  # our model
        input,  # a representative input-set to be used for both quantization and compilation
        n_bits=n_bits,
    )
    end_time = time.time()
    print("编译时间", end_time - start_time)
    print("量化位数：", n_bits, "最大量化位数", quantized_module.fhe_circuit.graph.maximum_integer_bit_width())
    x_test = torch.randn(1, 1).numpy()
    # 推理时间
    start_time = time.time()
    y_pred = quantized_module.forward(x_test, fhe="execute")
    end_time = time.time()
    print("推理时间", end_time - start_time)
    print("输入值：", x_test, "输出值:", y_pred)
    print("*" * 50)

if __name__ == '__main__':
    testSigmoidPTQ(2)
    testSigmoidPTQ(4)
    testSigmoidPTQ(8)
    testSigmoidPTQ(16)

