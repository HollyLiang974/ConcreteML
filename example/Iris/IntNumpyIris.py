'''
-*- ecoding: utf-8 -*-
@Enviroment: ConcreteML
@ModuleName: INTNumpyIris
@Author: Sakura
@Time: 2023/9/17 19:39
@Software: PyCharm
功能描述: 使用整数训练前馈神经网络
实现步骤:
结果：
'''
from functools import partial

import numpy
import pytest
from concrete import fhe
from concrete.ml.quantization import QuantizedArray
from concrete.ml.quantization.quantized_ops import (
    QuantizedGemm,
    QuantizedSigmoid,
    QuantizedSub,
    QuantizedMul,
    QuantizedDiv,
    QuantizedReduceSum,
)
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# 加载Iris数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 将输出标签进行独热编码
encoder = OneHotEncoder(sparse=False)
y_one_hot = encoder.fit_transform(y.reshape(-1, 1))

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)


# 定义一个类来表示NumPy版的FFNN模型
class NumPyFFNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.weights1 = np.random.randn(input_dim, hidden_dim)
        self.bias1 = np.zeros(hidden_dim)
        self.weights2 = np.random.randn(hidden_dim, output_dim)
        self.bias2 = np.zeros(output_dim)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def q_sub(self, n_bits:int,input_0 :numpy.ndarray, input_1:numpy.ndarray):
        # Quantize the inputs with n_bits
        q_inputs_0 = QuantizedArray(n_bits, input_0, is_signed=True)
        q_inputs_1 = QuantizedArray(n_bits, input_1, is_signed=True)

        q_op = QuantizedSub(n_bits, QuantizedSub.__name__, int_input_names={"0", "1"})
        raw_output_vv = q_op.calibrate(input_0, input_1)
        # print("raw_output_vv:\n", raw_output_vv)
        quantized_output_vv = q_op(q_inputs_0, q_inputs_1).dequant()
        # print("quantized_output_vv:\n", quantized_output_vv)

        return  quantized_output_vv

    def q_mul(self, n_bits:int,input_0 :numpy.ndarray, input_1:numpy.ndarray):
        # Quantize the inputs with n_bits
        q_inputs_0 = QuantizedArray(n_bits, input_0, is_signed=True)
        q_inputs_1 = QuantizedArray(n_bits, input_1, is_signed=True)

        q_op = QuantizedMul(
            n_bits, QuantizedMul.__name__, int_input_names={"0"}, constant_inputs={"b": q_inputs_1}
        )
        raw_output_vv = q_op.calibrate(input_0)
        # print("raw_output_vv:\n", raw_output_vv)
        quantized_output_vv = q_op(q_inputs_0).dequant()
        # print("quantized_output_vv:\n", quantized_output_vv)
        return quantized_output_vv

    def q_div(self, n_bits:int,input_0 :numpy.ndarray, input_1:numpy.ndarray):
        # Quantize the inputs with n_bits
        q_inputs_0 = QuantizedArray(n_bits, input_0, is_signed=True)
        q_inputs_1 = QuantizedArray(n_bits, input_1, is_signed=True)

        q_op = QuantizedDiv(
            n_bits, QuantizedDiv.__name__, int_input_names={"0"}, constant_inputs={"b": q_inputs_1}
        )
        raw_output_vv = q_op.calibrate(input_0)
        # print("raw_output_vv:\n", raw_output_vv)
        quantized_output_vv = q_op(q_inputs_0).dequant()
        # print("quantized_output_vv:\n", quantized_output_vv)
        return quantized_output_vv

    def q_gemm_(
        self,
        n_bits: int,
        inputs: numpy.ndarray,
        weights: numpy.ndarray,
        bias: numpy.ndarray,
    ):
        OP_DEBUG_NAME = "Test_"
        q_inputs = QuantizedArray(n_bits, inputs)
        q_weights = QuantizedArray(n_bits, weights, is_signed=True)
        q_gemm = QuantizedGemm(
            n_bits,
            OP_DEBUG_NAME + "QuantizedGemm",
            int_input_names={"0"},
            constant_inputs={"b": q_weights, "c": bias},

        )

        # Calibrate the Quantized layer
        q_gemm.produces_graph_output = True
        q_gemm.calibrate(inputs)
        actual_gemm_output = q_gemm(q_inputs).dequant()
        # print("actual_gemm_output:\n", actual_gemm_output)

        return actual_gemm_output

    def q_gemm_no_b(
        self,
        n_bits: int,
        inputs: numpy.ndarray,
        weights: numpy.ndarray,
    ):
        OP_DEBUG_NAME = "Test_"
        q_inputs = QuantizedArray(n_bits, inputs)
        q_weights = QuantizedArray(n_bits, weights, is_signed=True)
        q_gemm = QuantizedGemm(
            n_bits,
            OP_DEBUG_NAME + "QuantizedGemm",
            int_input_names={"0"},
            constant_inputs={"b": q_weights},

        )

        # Calibrate the Quantized layer
        q_gemm.produces_graph_output = True
        q_gemm.calibrate(inputs)
        actual_gemm_output = q_gemm(q_inputs).dequant()
        # print("actual_gemm_output:\n", actual_gemm_output)

        return actual_gemm_output

    def q_sigmoid(self,n_bits: int, inputs:numpy.ndarray):
        q_inputs = QuantizedArray(n_bits, inputs,is_signed=True)
        quantized_op = QuantizedSigmoid(n_bits, QuantizedSigmoid)
        expected_output = quantized_op.calibrate(inputs)
        # print("expected_output:\n", expected_output)
        q_output = quantized_op(q_inputs)
        values = q_output.values
        # print("values:\n", values)
        return values

    def forward(self, x):
        out1 = np.dot(x, self.weights1) + self.bias1
        out1_sigmoid = self.sigmoid(out1)  # 使用Sigmoid激活函数
        out2 = np.dot(out1_sigmoid, self.weights2) + self.bias2
        return out2

    def q_forward(self, x):
        n_bits=4
        # 线性层
        out1 = self.q_gemm_(n_bits,x,self.weights1,self.bias1)
        # 激活层
        out1_sigmoid = self.q_sigmoid(n_bits,out1)  # 使用Sigmoid激活函数
        out2=self.q_gemm_(n_bits,out1_sigmoid, self.weights2, self.bias2)
        return out2

    def backward(self, x, y, learning_rate):
        # 前向传播
        out1 = np.dot(x, self.weights1) + self.bias1
        out1_sigmoid = self.sigmoid(out1)  # 使用Sigmoid激活函数
        out2 = np.dot(out1_sigmoid, self.weights2) + self.bias2
        loss = np.mean((out2 - y) ** 2)  # 均方误差损失

        # 反向传播
        delta_out2 = 2 * (out2 - y) / len(x)
        delta_weights2 = np.dot(out1_sigmoid.T, delta_out2)
        delta_bias2 = np.sum(delta_out2, axis=0)
        delta_out1 = np.dot(delta_out2, self.weights2.T)
        delta_out1_sigmoid = delta_out1 * out1_sigmoid * (1 - out1_sigmoid)  # Sigmoid的导数
        delta_weights1 = np.dot(x.T, delta_out1_sigmoid)
        delta_bias1 = np.sum(delta_out1_sigmoid, axis=0)

        # 更新参数
        self.weights2 -= learning_rate * delta_weights2
        self.bias2 -= learning_rate * delta_bias2
        self.weights1 -= learning_rate * delta_weights1
        self.bias1 -= learning_rate * delta_bias1

        return loss

    def q_backward(self, x, y, learning_rate):
        n_bits = 8
        # 前向传播
        out1 = self.q_gemm_(n_bits, x, self.weights1, self.bias1)
        out1_sigmoid = self.q_sigmoid(n_bits, out1)  # 使用Sigmoid激活函数
        out2 = self.q_gemm_(n_bits, out1_sigmoid, self.weights2, self.bias2)
        loss = np.mean((out2 - y) ** 2)  # 均方误差损失

        # 反向传播
        # delta_out2 = 2 * (out2 - y) / len(x)  #out2是密文，y是密文，所以delta_out2是密文
        delta_out2=self.q_div(n_bits,self.q_mul(n_bits,self.q_sub(n_bits,out2,y),2),len(x))
        # delta_weights2 = np.dot(out1_sigmoid.T, delta_out2) #out1_sigmoid.T是密文，delta_out2是密文，所以delta_weights2是密文，
        delta_weights2=self.q_gemm_no_b(n_bits,out1_sigmoid.T,delta_out2)                                             # 密文和密文矩阵乘法不能超过16位
        delta_bias2 = np.sum(delta_out2, axis=0) #delta_out2是密文，所以delta_bias2是密文
        # delta_out1 = np.dot(delta_out2, self.weights2.T)
        delta_out1=self.q_gemm_no_b(n_bits,delta_out2,self.weights2.T)
        # delta_out1_sigmoid = delta_out1 * out1_sigmoid * (1 - out1_sigmoid)  # Sigmoid的导数
        delta_out1_sigmoid=self.q_mul(n_bits,self.q_mul(n_bits,delta_out1,out1_sigmoid),self.q_sub(n_bits,1,out1_sigmoid))
        # delta_weights1 = np.dot(x.T, delta_out1_sigmoid)
        delta_weights1=self.q_gemm_no_b(n_bits,x.T,delta_out1_sigmoid)
        delta_bias1 = np.sum(delta_out1_sigmoid, axis=0)

        # 更新参数
        self.weights2 -= learning_rate * delta_weights2
        self.bias2 -= learning_rate * delta_bias2
        self.weights1 -= learning_rate * delta_weights1
        self.bias1 -= learning_rate * delta_bias1

        return loss


# 模型参数
input_dim = X_train.shape[1]
hidden_dim = 64
output_dim = y_train.shape[1]

# 创建NumPy模型实例
numpy_model = NumPyFFNN(input_dim, hidden_dim, output_dim)


# 训练模型
learning_rate = 0.01
num_epochs = 1000

for epoch in range(num_epochs):
    # 随机选择一批训练数据
    batch_indices = np.random.choice(len(X_train), 32, replace=False)
    x_batch = X_train[batch_indices]
    y_batch = y_train[batch_indices]

    # 执行一次前向传播和反向传播，并获得损失
    loss = numpy_model.q_backward(x_batch, y_batch, learning_rate)

    # 打印损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss}')

# 测试模型
# 前向传播
predictions = numpy_model.forward(X_test)

# 计算准确率
correct = (np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)).sum()
total = len(X_test)
accuracy = correct / total * 100

print(f'Test Accuracy: {accuracy:.2f}%')
