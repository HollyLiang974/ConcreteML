'''
-*- ecoding: utf-8 -*-
@Enviroment: concrete-ml-1.1.0
@ModuleName: test_gemm
@Author: Sakura
@Time: 2023/9/17 15:11
@Software: PyCharm
功能描述:
实现步骤:
结果：
'''

import numpy
from concrete.ml.quantization import QuantizedArray
from concrete.ml.quantization.quantized_ops import (
    QuantizedGemm,
)
from concrete import fhe


"""Test for gemm style ops."""
n_bits = 8
n_examples = 10
n_features = 20
n_neurons = 1

inputs = numpy.random.randn(n_examples, n_features)
weights = numpy.random.randn(n_features, n_neurons)
bias = numpy.zeros(n_neurons)

# Quantize the inputs and weights
q_inputs = QuantizedArray(n_bits, inputs)
q_weights = QuantizedArray(n_bits, weights, is_signed=True)
# 1- Test our QuantizedGemm layer
OP_DEBUG_NAME = "Test_"
q_gemm = QuantizedGemm(
    n_bits,
    OP_DEBUG_NAME + "QuantizedGemm",
    int_input_names={"0"},
    constant_inputs={"b": q_weights, "c": bias},
)
q_gemm.produces_graph_output = True
expected_gemm_outputs = q_gemm.calibrate(inputs)
print("expected_gemm_outputs:\n", expected_gemm_outputs)
actual_gemm_output = q_gemm(q_inputs).dequant()
print("actual_gemm_output:\n", actual_gemm_output)


# 电路的编译还存在问题
# @fhe.compiler({"q_inputs":"encrypted"})
# def test_gemm_circuit(q_inputs):
#     actual_gemm_output = q_gemm(q_inputs).update_quantized_values(qvalues)
#     return actual_gemm_output
#
# inputset=q_inputs.qvalues
# circuit=test_gemm_circuit.compile(inputset)
