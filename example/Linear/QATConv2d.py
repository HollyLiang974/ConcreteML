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
import brevitas.nn as qnn
from concrete.ml.torch.compile import compile_brevitas_qat_model

def testLinearQAT(bit_width):

    class QATNet(nn.Module):
        def __init__(self,bit_width):
            super().__init__()
            self.quant_inp = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True)
            self.conv2d = qnn.QuantConv2d(3,6,5, bias=True, weight_bit_width=bit_width)

        def forward(self, x):
            x= self.quant_inp(x)
            x = self.conv2d(x)
            return x
    input = torch.randn(10, 3, 50, 100)
    model = QATNet(bit_width)
    # 模型编译时间
    start_time = time.time()
    quantized_module = compile_brevitas_qat_model(
        model, # our model
        input, # a representative input-set to be used for both quantization and compilation

    )
    end_time = time.time()
    print("编译时间", end_time - start_time)
    print("量化位数：", bit_width, "最大量化位数", quantized_module.fhe_circuit.graph.maximum_integer_bit_width())
    # 推理时间
    start_time = time.time()
    x_test = torch.randn(1, 3, 50, 100).numpy()
    y_pred = quantized_module.forward(x_test, fhe="execute")
    end_time = time.time()
    print("推理时间", end_time - start_time)
    # print("输入值：", x_test, "输出值:", y_pred)
    print("*" * 50)

if __name__ == '__main__':
    testLinearQAT(2)
    testLinearQAT(4)
    testLinearQAT(8)
    testLinearQAT(16)
