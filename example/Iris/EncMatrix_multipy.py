'''
-*- ecoding: utf-8 -*-
@Enviroment: ConcreteML
@ModuleName: test1
@Author: Sakura
@Time: 2023/9/6 12:41
@Software: PyCharm
功能描述:
实现步骤:
结果：
'''
import numpy as np
from concrete import fhe

@fhe.compiler({"list1": "encrypted", "list2": "clear"})
def dot(list1,list2):
    result=np.dot(list1, list2)
    return result
inputset=[(np.random.randint(0, 50, size=(2,)),np.random.randint(0, 50, size=(2,))) for i in range(100)]
circuit = dot.compile(inputset)
#矩阵乘法方法
def matrix_multiply(list1, list2):
    result = []
    for row in list1:
        result_row = []
        for col in zip(*list2):
            dot_product = circuit.encrypt(row, list(col))
            result_row.append(dot_product)
        result.append(result_row)
    return result

list1=np.random.randint(0, 50, size=(3,2))
list2=np.random.randint(0, 50, size=(2,3))
result=matrix_multiply(list1,list2)
print("result",result)
print("real_result",np.dot(list1,list2))


