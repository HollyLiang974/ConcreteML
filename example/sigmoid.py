'''
-*- ecoding: utf-8 -*-
@Enviroment: ConcreteML
@ModuleName: sigmoid
@Author: Sakura
@Time: 2023/8/1 11:16
@Software: PyCharm
功能描述:
实现步骤:
结果：
'''
import numpy as np
from concrete import fhe
@fhe.compiler({"x":"Encrypted"})
def sigmoid(x):
    return (1 / (1 + np.exp(-x))).astype(np.int64)
x = np.array([127,-128])
circuit=sigmoid.compile(x)
print(circuit)
print(circuit.encrypt_run_decrypt(5))

