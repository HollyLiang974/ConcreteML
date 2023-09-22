'''
-*- ecoding: utf-8 -*-
@Enviroment: ConcreteML
@ModuleName: sigmoid
@Author: Sakura
@Time: 2023/9/12 20:10
@Software: PyCharm
功能描述: sigmoid查找表
实现步骤:
结果：
'''
import numpy as np
import matplotlib.pyplot as plt


# 定义sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
#这一行创建了一个包含100个均匀分布的点的NumPy数组x，这些点从-10到10的范围内，用于表示x轴上的值。
x = np.linspace(-10, 10, 100)
y = sigmoid(x)

# 'g' 表示要绘制绿色的线条。这里的 'g' 控制了线条的颜色，使得绘制的曲线为绿色。
# Matplotlib 支持多种颜色的选项，以便你可以根据需要自定义曲线的颜色，
# 例如 'b' 表示蓝色，'r' 表示红色，等等。这些颜色选项允许你创建具有不同颜色的线条来区分不同的数据集或特征。
plt.plot(x, y, 'g')


# 用于创建查找表的点数
lut_size = 10
lut_x = np.linspace(-10, 10, lut_size)
lut_y = sigmoid(lut_x)
#'ro' 是用于指定绘图样式的字符串参数，其中:
# 'r' 表示颜色选项，代表红色 (red)。
# 'o' 表示标记选项，代表绘制圆形标记 (circle markers)。
plt.plot(lut_x, lut_y, 'ro')
plt.plot(lut_x, lut_y, 'r')


x = np.random.uniform(-15, 15, 30)
y = np.interp(x, lut_x, lut_y)
plt.plot(x, y, 'bo')



plt.show()
