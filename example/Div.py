'''
-*- ecoding: utf-8 -*-
@Enviroment: ConcreteML
@ModuleName: Div
@Author: Sakura
@Time: 2023/9/10 19:49
@Software: PyCharm
功能描述:
实现步骤:
结果：
'''
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple, Union
from concrete import fhe
import time
import numpy as np

from concrete.fhe import EncryptionStatus
config=fhe.Configuration(show_graph=True, enable_unsafe_features=True,show_optimizer=True)
# 创建一个空字典来存储时间
timings = {}
def TestDiv1(size,inputset,x,parameters: Mapping[str, Union[str, EncryptionStatus]]):
    @fhe.compiler(parameters)
    def div(x):
        return 1 // x
        # 生成电路时间

    start_time = time.time()
    circuit = div.compile(inputset, configuration=config)
    timings["生成电路时间", size] = time.time() - start_time

    # 加密时间
    start_time = time.time()
    enc = circuit.encrypt(x)
    timings["加密时间", size] = time.time() - start_time

    # 计算时间
    start_time = time.time()
    res_enc = circuit.run(enc)
    timings["计算时间", size] = time.time() - start_time

    # 解密时间
    start_time = time.time()
    plaintext = circuit.decrypt(res_enc)
    timings["解密时间", size] = time.time() - start_time
    return 0
print("*"*50,"TestDiv1 Unsigned","*"*50)
TestDiv1("uint2ec",[np.random.randint(1, 4)  for i in range(10000)], np.random.randint(1, 4), {"x": "encrypted"})
TestDiv1("uint4ec",[np.random.randint(1, 16) for i in range(10000)], np.random.randint(1, 16), {"x": "encrypted"})
TestDiv1("uint8ec",[np.random.randint(1, 256) for i in range(10000)], np.random.randint(1, 256), {"x": "encrypted"})
# TestDiv1("uint16ec",[np.random.randint(1, 65536) for i in range(10000)], np.random.randint(1, 65536), {"x": "encrypted"})
# TestDiv1("uint32ec",[np.random.randint(1, 4294967296) for i in range(10000)], np.random.randint(1, 4294967296), {"x": "encrypted"})


print("*"*50,"TestDiv1 Signed","*"*50)
TestDiv1("int2ec",[np.random.randint(-2, 1)  for i in range(10000)], np.random.randint(-2, 1), {"x": "encrypted"})
TestDiv1("int4ec",[np.random.randint(-8, 7)  for i in range(10000)], np.random.randint(-8, 7), {"x": "encrypted"})
TestDiv1("int8ec",[np.random.randint(-128, 127) for i in range(10000)], np.random.randint(-128, 127), {"x": "encrypted"})
# TestDiv1("int16ec",[np.random.randint(-32768, 32767) for i in range(10000)], np.random.randint(-32768, 32767), {"x": "encrypted"})
# TestDiv1("int32ec",[np.random.randint(-2147483648, 2147483647) for i in range(10000)], np.random.randint(-2147483648, 2147483647), {"x": "encrypted"})
# 打印时间字典
print("时间记录：")
count = 0  # 计数器，用于跟踪打印的键值对数量
for key, value in timings.items():
    print(key, ":", value)  # 打印键值对，不换行
    count += 1
    if count % 4 == 0:  # 每隔四个键值对
        print("\n\n")  # 打印一个空行
def TestDiv2(size,inputset,x,parameters: Mapping[str, Union[str, EncryptionStatus]]):
    @fhe.compiler(parameters)
    def div(x):
        return x // 1

    start_time = time.time()
    circuit = div.compile(inputset, configuration=config)
    timings["生成电路时间", size] = time.time() - start_time

    # 加密时间
    start_time = time.time()
    enc = circuit.encrypt(x)
    timings["加密时间", size] = time.time() - start_time

    # 计算时间
    start_time = time.time()
    res_enc = circuit.run(enc)
    timings["计算时间", size] = time.time() - start_time

    # 解密时间
    start_time = time.time()
    plaintext = circuit.decrypt(res_enc)
    timings["解密时间", size] = time.time() - start_time
    return 0


print("*"*50,"TestDiv2 Unsigned","*"*50)
TestDiv2("uint2ec",[np.random.randint(0, 4) for i in range(10000)], np.random.randint(0, 4), {"x": "encrypted"})
TestDiv2("uint4ec",[np.random.randint(0, 16) for i in range(10000)], np.random.randint(0, 16), {"x": "encrypted"})
TestDiv2("uint8ec",[np.random.randint(0, 256) for i in range(10000)], np.random.randint(0, 256), {"x": "encrypted"})
# TestDiv2("uint16ec",[np.random.randint(0, 65536) for i in range(10000)], np.random.randint(0, 65536), {"x": "encrypted"})
# TestDiv2("uint32ec",[np.random.randint(0, 4294967296) for i in range(10000)], np.random.randint(0, 4294967296), {"x": "encrypted"})


print("*"*50,"TestDiv2 Signed","*"*50)
TestDiv2("int2ec",[np.random.randint(-2, 1) for i in range(10000)], np.random.randint(-2, 1), {"x": "encrypted"})
TestDiv2("int4ec",[np.random.randint(-8, 7) for i in range(10000)], np.random.randint(-8, 7), {"x": "encrypted"})
TestDiv2("int8ec",[np.random.randint(-128, 127) for i in range(10000)], np.random.randint(-128, 127), {"x": "encrypted"})
# TestDiv2("int16ec",[np.random.randint(-32768, 32767) for i in range(10000)], np.random.randint(-32768, 32767), {"x": "encrypted"})
# TestDiv2("int32ec",[np.random.randint(-2147483648, 2147483647) for i in range(10000)], np.random.randint(-2147483648, 2147483647), {"x": "encrypted"})
# 打印时间字典
print("时间记录：")
count = 0  # 计数器，用于跟踪打印的键值对数量
for key, value in timings.items():
    print(key, ":", value)  # 打印键值对，不换行
    count += 1
    if count % 4 == 0:  # 每隔四个键值对
        print("\n\n")  # 打印一个空行