'''
-*- ecoding: utf-8 -*-
@Enviroment: ConcreteML
@ModuleName: NumpyDot
@Author: Sakura
@Time: 2023/9/4 16:35
@Software: PyCharm
功能描述: Numpy的点乘操作，使用TFHE加密电路
实现步骤:
结果：只能计算一维数组和一维数组之间的点乘
'''

import numpy as np
from concrete import fhe

# 创建两个二维数组
matrix1 = np.random.randint(-32768, 32767, size=(4,))
matrix2 = np.random.randint(-32768, 32767, size=(4,))
print(matrix1.shape)
print(matrix2.shape)
# 计算点积，即矩阵乘法
dot_product = np.dot(matrix1, matrix2)

print("Dot Product (Matrix Multiplication):\n", dot_product)
@fhe.compiler({"x":"Encrypted"})
def dot(x):
    return np.dot(x,matrix2)


circuit=dot.compile([np.random.randint(-32768, 32767, size=(4,)) for i in range(100)])
print(circuit)
print(circuit.encrypt_run_decrypt(matrix1))

