'''
-*- ecoding: utf-8 -*-
@Enviroment: ConcreteML
@ModuleName: binary_matrix
@Author: Sakura
@Time: 2023/9/22 14:15
@Software: PyCharm
功能描述:
实现步骤:
结果：
'''
import numpy
import numpy as np

def binary_matrix(matrix:numpy.ndarray):
    """
    将矩阵转换为二进制矩阵
    :param matrix: 实数矩阵
    :return: 8*matrix.shape[0]*matrix.shape[1]的二进制矩阵
    """
    bin_matrix = np.unpackbits(matrix.astype(np.uint8), axis=0)
    bits_matrix=[]
    for row in range(matrix.shape[0]):
        bits_matrix.append(bin_matrix[8 * row:8 * row + 8, :])
    bits_matrix=np.array(bits_matrix).transpose(1,0,-1)
    return bits_matrix
