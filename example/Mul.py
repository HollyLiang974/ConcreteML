'''
-*- ecoding: utf-8 -*-
@Enviroment: ConcreteML
@ModuleName: Mul
@Author: Sakura
@Time: 2023/9/10 19:28
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
def TestMul(size,inputset, x, y,parameters: Mapping[str, Union[str, EncryptionStatus]]):
    @fhe.compiler(parameters)
    def mul(x, y):
        return x * y
    # 生成电路时间
    start_time = time.time()
    circuit = mul.compile(inputset, configuration=config)
    timings["生成电路时间", size] = time.time() - start_time

    # 加密时间
    start_time = time.time()
    enc = circuit.encrypt(x, y)
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
print("*"*50,"TestMul Unsigned","*"*50)
TestMul("uint2ec",[(np.random.randint(0, 4),np.random.randint(0, 4)) for i in range(10000)], np.random.randint(0, 4), np.random.randint(0, 4), {"x": "encrypted", "y": "clear"})
TestMul("uint2ee",[(np.random.randint(0, 4),np.random.randint(0, 4)) for i in range(10000)], np.random.randint(0, 4), np.random.randint(0, 4), {"x": "encrypted", "y": "encrypted"})
TestMul("uint4ec",[(np.random.randint(0, 16),np.random.randint(0, 16)) for i in range(10000)], np.random.randint(0, 16), np.random.randint(0, 16), {"x": "encrypted", "y": "clear"})
TestMul("uint4ee",[(np.random.randint(0, 16),np.random.randint(0, 16)) for i in range(10000)], np.random.randint(0, 16), np.random.randint(0, 16), {"x": "encrypted", "y": "encrypted"})
TestMul("uint8ec",[(np.random.randint(0, 256),np.random.randint(0, 256)) for i in range(10000)], np.random.randint(0, 256), np.random.randint(0, 256), {"x": "encrypted", "y": "clear"})
TestMul("uint8ee",[(np.random.randint(0, 256),np.random.randint(0, 256)) for i in range(10000)], np.random.randint(0, 256), np.random.randint(0, 256), {"x": "encrypted", "y": "encrypted"})
# TestMul("uint16ec",[(np.random.randint(0, 65536),np.random.randint(0, 65536)) for i in range(10000)], np.random.randint(0, 65536), np.random.randint(0, 65536), {"x": "encrypted", "y": "clear"})
# TestMul("uint16ee",[(np.random.randint(0, 65536),np.random.randint(0, 65536)) for i in range(10000)], np.random.randint(0, 65536), np.random.randint(0, 65536), {"x": "encrypted", "y": "encrypted"})
# TestMul("uint32ec",[(np.random.randint(0, 4294967296),np.random.randint(0, 4294967296)) for i in range(10000)], np.random.randint(0, 4294967296), np.random.randint(0, 4294967296), {"x": "encrypted", "y": "clear"})
# TestMul("uint32ee",[(np.random.randint(0, 4294967296),np.random.randint(0, 4294967296)) for i in range(10000)], np.random.randint(0, 4294967296), np.random.randint(0, 4294967296), {"x": "encrypted", "y": "encrypted"})

print("*"*50,"TestMul signed","*"*50)
TestMul("int2ec",[(np.random.randint(-2, 2),np.random.randint(-2, 2)) for i in range(10000)], np.random.randint(-2, 2), np.random.randint(-2, 2), {"x": "encrypted", "y": "clear"})
TestMul("int2ee",[(np.random.randint(-2, 2),np.random.randint(-2, 2)) for i in range(10000)], np.random.randint(-2, 2), np.random.randint(-2, 2), {"x": "encrypted", "y": "encrypted"})
TestMul("int4ec",[(np.random.randint(-8, 8),np.random.randint(-8, 8)) for i in range(10000)], np.random.randint(-8, 8), np.random.randint(-8, 8), {"x": "encrypted", "y": "clear"})
TestMul("int4ee",[(np.random.randint(-8, 8),np.random.randint(-8, 8)) for i in range(10000)], np.random.randint(-8, 8), np.random.randint(-8, 8), {"x": "encrypted", "y": "encrypted"})
TestMul("int8ec",[(np.random.randint(-128, 128),np.random.randint(-128, 128)) for i in range(10000)], np.random.randint(-128, 128), np.random.randint(-128, 128), {"x": "encrypted", "y": "clear"})
TestMul("int8ee",[(np.random.randint(-128, 128),np.random.randint(-128, 128)) for i in range(10000)], np.random.randint(-128, 128), np.random.randint(-128, 128), {"x": "encrypted", "y": "encrypted"})
# TestMul("int16ec",[(np.random.randint(-32768, 32767),np.random.randint(-32768, 32767)) for i in range(10000)], np.random.randint(-32768, 32767), np.random.randint(-32768, 32767), {"x": "encrypted", "y": "clear"})
# TestMul("int16ee",[(np.random.randint(-32768, 32767),np.random.randint(-32768, 32767)) for i in range(10000)], np.random.randint(-32768, 32767), np.random.randint(-32768, 32767), {"x": "encrypted", "y": "encrypted"})
# TestMul("int32ec",[(np.random.randint(-2147483648, 2147483647),np.random.randint(-2147483648, 2147483647)) for i in range(10000)], np.random.randint(-2147483648, 2147483647), np.random.randint(-2147483648, 2147483647), {"x": "encrypted", "y": "clear"})
# TestMul("int32ee",[(np.random.randint(-2147483648, 2147483647),np.random.randint(-2147483648, 2147483647)) for i in range(10000)], np.random.randint(-2147483648, 2147483647), np.random.randint(-2147483648, 2147483647), {"x": "encrypted", "y": "encrypted"})
# 打印时间字典
print("时间记录：")
count = 0  # 计数器，用于跟踪打印的键值对数量
for key, value in timings.items():
    print(key, ":", value)  # 打印键值对，不换行
    count += 1
    if count % 4 == 0:  # 每隔四个键值对
        print("\n\n")  # 打印一个空行