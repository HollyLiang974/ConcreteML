'''
-*- ecoding: utf-8 -*-
@Enviroment: ConcreteML
@ModuleName: test3
@Author: Sakura
@Time: 2023/9/7 18:39
@Software: PyCharm
功能描述:
实现步骤:
结果：
'''
from concrete import fhe
import numpy as np
w= np.random.randint(0, 255, size=(576, 10))
b= np.random.randint(28, 245, size=(10,))
configuration = fhe.Configuration(
    enable_unsafe_features=True,
    show_mlir=False,
    show_graph=True,
)
@fhe.compiler({"x": "encrypted"})
def f_lr(x):
    res = x @ w+b
    return res

inputset = [np.random.randint(0, 15, size=(576,)) for i in range(10000)]

circuit = f_lr.compile(inputset,configuration=configuration)

input=np.random.randint(0, 15, size=(576,))

print("result",circuit.encrypt_run_decrypt(input))
print("real_result",np.dot(input, w)+b)
