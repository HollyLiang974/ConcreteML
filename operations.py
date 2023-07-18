'''
-*- ecoding: utf-8 -*-
@Enviroment: ConcreteML
@ModuleName: operations
@Author: Sakura
@Time: 2023/7/17 17:23
@Software: PyCharm
功能描述:
TFHE基础操作的性能，密+/*明、密+/*密、基础查表操作（2/4/8/16位）有符号数
实现步骤:
结果：
'''
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple, Union
from concrete import fhe
import time

from concrete.fhe import EncryptionStatus

x = 0
y = 1

print("*"*50,"TestAdd","*"*50)
def TestAdd(inputset, x, y,parameters: Mapping[str, Union[str, EncryptionStatus]]):
    @fhe.compiler(parameters)
    def add(x, y):
        return x + y
    circuit = add.compile(inputset)
    print(circuit)
    start_time = time.time()
    print("结果:", circuit.encrypt_run_decrypt(x, y))
    end_time = time.time()
    print("时间:", end_time - start_time)
    return 0

TestAdd([(-2, 1), (1, -2)], x, y, {"x": "encrypted", "y": "clear"})
TestAdd([(-2, 1), (1, -2)], x, y, {"x": "encrypted", "y": "encrypted"})
TestAdd([(-8, 7), (7, -8)], x, y, {"x": "encrypted", "y": "clear"})
TestAdd([(-8, 7), (7, -8)], x, y, {"x": "encrypted", "y": "encrypted"})
TestAdd([(-128, 127), (127, -128)], x, y, {"x": "encrypted", "y": "clear"})
TestAdd([(-128, 127), (127, -128)], x, y, {"x": "encrypted", "y": "encrypted"})
TestAdd([(-32768, 32767), (32767, -32768)], x, y, {"x": "encrypted", "y": "clear"})
TestAdd([(-32768, 32767), (32767, -32768)], x, y, {"x": "encrypted", "y": "encrypted"})

print("*"*50,"TestSub","*"*50)

def TestSub(inputset, x, y,parameters: Mapping[str, Union[str, EncryptionStatus]]):
    @fhe.compiler(parameters)
    def sub(x, y):
        return x - y
    circuit = sub.compile(inputset)
    print(circuit)
    start_time = time.time()
    print("结果:", circuit.encrypt_run_decrypt(x, y))
    end_time = time.time()
    print("时间:", end_time - start_time)
    return 0

TestSub([(-2, 1), (1, -2)], x, y, {"x": "encrypted", "y": "clear"})
TestSub([(-2, 1), (1, -2)], x, y, {"x": "encrypted", "y": "encrypted"})
TestSub([(-8, 7), (7, -8)], x, y, {"x": "encrypted", "y": "clear"})
TestSub([(-8, 7), (7, -8)], x, y, {"x": "encrypted", "y": "encrypted"})
TestSub([(-128, 127), (127, -128)], x, y, {"x": "encrypted", "y": "clear"})
TestSub([(-128, 127), (127, -128)], x, y, {"x": "encrypted", "y": "encrypted"})
TestSub([(-32768, 32767), (32767, -32768)], x, y, {"x": "encrypted", "y": "clear"})
TestSub([(-32768, 32767), (32767, -32768)], x, y, {"x": "encrypted", "y": "encrypted"})

print("*"*50,"TestMul","*"*50)
def TestMul(inputset, x, y,parameters: Mapping[str, Union[str, EncryptionStatus]]):
    @fhe.compiler(parameters)
    def mul(x, y):
        return x * y
    circuit = mul.compile(inputset)
    print(circuit)
    start_time = time.time()
    print("结果:", circuit.encrypt_run_decrypt(x, y))
    end_time = time.time()
    print("时间:", end_time - start_time)
    return 0

TestMul([(-2, 1), (1, -2)], x, y, {"x": "encrypted", "y": "clear"})
TestMul([(-2, 1), (1, -2)], x, y, {"x": "encrypted", "y": "encrypted"})
TestMul([(-8, 7), (7, -8)], x, y, {"x": "encrypted", "y": "clear"})
TestMul([(-8, 7), (7, -8)], x, y, {"x": "encrypted", "y": "encrypted"})
# TestMul([(-128, 127), (127, -128)], x, y, {"x": "encrypted", "y": "clear"})
# TestMul([(-128, 127), (127, -128)], x, y, {"x": "encrypted", "y": "encrypted"})
# TestMul([(-32768, 32767), (32767, -32768)], x, y, {"x": "encrypted", "y": "clear"})
# TestMul([(-32768, 32767), (32767, -32768)], x, y, {"x": "encrypted", "y": "encrypted"})
print("*"*50,"TestDiv","*"*50)
def TestDiv(inputset,x,parameters: Mapping[str, Union[str, EncryptionStatus]]):
    @fhe.compiler(parameters)
    def div(x):
        return 1 // x
    circuit = div.compile(inputset)
    print(circuit)
    start_time = time.time()
    print("结果:", circuit.encrypt_run_decrypt(x))
    end_time = time.time()
    print("时间:", end_time - start_time)
    return 0

TestDiv([(-2)], x, {"x": "encrypted"})
TestDiv([(-8)], x,  {"x": "encrypted"})
TestDiv([(-128) ], x, {"x": "encrypted"})
#TestDiv([(-32768)], x,  {"x": "encrypted"})


