'''
-*- ecoding: utf-8 -*-
@Enviroment: ConcreteML
@ModuleName: test
@Author: Sakura
@Time: 2023/7/21 13:59
@Software: PyCharm
功能描述:
实现步骤:
结果：
'''
import torch
import torch.nn as nn
import brevitas.nn as qnn
import numpy
from concrete.ml.torch.compile import compile_torch_model
class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()
        # self.linear=qnn.QuantLinear(2,1,weight_bit_width=8,bias=False)
        self.linear=nn.Linear(2,1)
        a=torch.tensor([[1,2],[3,4]])
        b=torch.tensor([[5,6],[7,8]])



    def forward(self,x):
        x=self.linear(x)
        return x

net=model()
input=torch.randn(10,2)

quantized_module = compile_torch_model(
    net, # our model
    input, # a representative input-set to be used for both quantization and compilation
)
x_test = numpy.array([numpy.random.randn(2)])

y_pred = quantized_module.forward(x_test, fhe="execute")
print(y_pred)


