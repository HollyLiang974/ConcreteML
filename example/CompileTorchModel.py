'''
-*- ecoding: utf-8 -*-
@Enviroment: ConcreteML
@ModuleName: CompileTorchModel
@Author: Sakura
@Time: 2023/7/21 13:59
@Software: PyCharm
功能描述: 编译Torch模型
实现步骤:
结果：偶尔会出现编译失败的情况
编译时间： 20.415812253952026
[[-0.1113784]]
推理时间： 60.37916564941406
bitwidth= 16

编译时间： 0.19651556015014648
biwidth: 5
[[-0.1603658]]
推理时间： 8.404378175735474

'''
import torch
import torch.nn as nn
import brevitas.nn as qnn
import numpy
from concrete.ml.torch.compile import compile_torch_model
import time

class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()
        self.linear1=nn.Linear(2,3)
        self.relu=nn.ReLU()
        self.linear2 = nn.Linear(3,1)

    def forward(self,x):
        x=self.linear1(x)
        x=self.relu(x)
        x=self.linear2(x)
        return x

net=model()
input=torch.randn(10,2)
#编译时间
start_time = time.time()
quantized_module = compile_torch_model(
    net, # our model
    input, # a representative input-set to be used for both quantization and compilation
    n_bits=3
)
end_time = time.time()
print("编译时间：",end_time-start_time)
bitwidth=quantized_module.fhe_circuit.graph.maximum_integer_bit_width()
print("biwidth:",bitwidth)


x_test = numpy.array([numpy.random.randn(2)])
#推理时间
start_time = time.time()
y_pred = quantized_module.forward(x_test, fhe="execute")
print(y_pred)
end_time = time.time()
print("推理时间：",end_time-start_time)


