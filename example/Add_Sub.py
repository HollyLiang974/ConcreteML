'''
-*- ecoding: utf-8 -*-
@Enviroment: ConcreteML
@ModuleName: Add_Sub
@Author: Sakura
@Time: 2023/9/10 17:12
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
def TestAdd(size,inputset, x, y,parameters: Mapping[str, Union[str, EncryptionStatus]]):
    @fhe.compiler(parameters)
    def add(x, y):
        return x + y

    # 生成电路时间
    start_time = time.time()
    circuit = add.compile(inputset, configuration=config)
    timings["生成电路时间",size] = time.time() - start_time

    # 加密时间
    start_time = time.time()
    enc = circuit.encrypt(x, y)
    timings["加密时间",size] = time.time() - start_time

    # 计算时间
    start_time = time.time()
    res_enc = circuit.run(enc)
    timings["计算时间",size] = time.time() - start_time

    # 解密时间
    start_time = time.time()
    plaintext = circuit.decrypt(res_enc)
    timings["解密时间",size] = time.time() - start_time
    return 0
print("*"*50,"TestAdd Unsigned","*"*50)
TestAdd("uint2ec",[(np.random.randint(0, 4),np.random.randint(0, 4)) for i in range(10000)], np.random.randint(0, 4), np.random.randint(0, 4), {"x": "encrypted", "y": "clear"})
TestAdd("uint2ee",[(np.random.randint(0, 4),np.random.randint(0, 4)) for i in range(10000)], np.random.randint(0, 4), np.random.randint(0, 4), {"x": "encrypted", "y": "encrypted"})
TestAdd("uint4ec",[(np.random.randint(0, 16),np.random.randint(0, 16)) for i in range(10000)], np.random.randint(0, 16), np.random.randint(0, 16), {"x": "encrypted", "y": "clear"})
TestAdd("uint4ee",[(np.random.randint(0, 16),np.random.randint(0, 16)) for i in range(10000)], np.random.randint(0, 16), np.random.randint(0, 16), {"x": "encrypted", "y": "encrypted"})
TestAdd("uint8ec",[(np.random.randint(0, 256),np.random.randint(0, 256)) for i in range(10000)], np.random.randint(0, 256), np.random.randint(0, 256), {"x": "encrypted", "y": "clear"})
TestAdd("uint8ee",[(np.random.randint(0, 256),np.random.randint(0, 256)) for i in range(10000)], np.random.randint(0, 256), np.random.randint(0, 256), {"x": "encrypted", "y": "encrypted"})
TestAdd("uint16ec",[(np.random.randint(0, 65536),np.random.randint(0, 65536)) for i in range(10000)], np.random.randint(0, 65536), np.random.randint(0, 65536), {"x": "encrypted", "y": "clear"})
TestAdd("uint16ee",[(np.random.randint(0, 65536),np.random.randint(0, 65536)) for i in range(10000)], np.random.randint(0, 65536), np.random.randint(0, 65536), {"x": "encrypted", "y": "encrypted"})
TestAdd("uint32ec",[(np.random.randint(0, 4294967296),np.random.randint(0, 4294967296)) for i in range(10000)], np.random.randint(0, 4294967296), np.random.randint(0, 4294967296), {"x": "encrypted", "y": "clear"})
TestAdd("uint32ee",[(np.random.randint(0, 4294967296),np.random.randint(0, 4294967296)) for i in range(10000)], np.random.randint(0, 4294967296), np.random.randint(0, 4294967296), {"x": "encrypted", "y": "encrypted"})
# TestAdd("uint64ec",[(np.random.randint(0, 18446744073709551616),np.random.randint(0, 18446744073709551616)) for i in range(10000)], np.random.randint(0, 18446744073709551616), np.random.randint(0, 18446744073709551616), {"x": "encrypted", "y": "clear"})
# TestAdd("uint64ee",[(np.random.randint(0, 18446744073709551616),np.random.randint(0, 18446744073709551616)) for i in range(10000)], np.random.randint(0, 18446744073709551616), np.random.randint(0, 18446744073709551616), {"x": "encrypted", "y": "encrypted"})

print("*"*50,"TestAdd signed","*"*50)
TestAdd("int2ec",[(np.random.randint(-2, 1),np.random.randint(-2, 1)) for i in range(10000)], np.random.randint(-2, 1), np.random.randint(-2, 1), {"x": "encrypted", "y": "clear"})
TestAdd("int2ee",[(np.random.randint(-2, 1),np.random.randint(-2, 1)) for i in range(10000)], np.random.randint(-2, 1), np.random.randint(-2, 1), {"x": "encrypted", "y": "encrypted"})
TestAdd("int4ec",[(np.random.randint(-8, 7),np.random.randint(-8, 7)) for i in range(10000)], np.random.randint(-8, 7), np.random.randint(-8, 7), {"x": "encrypted", "y": "clear"})
TestAdd("int4ee",[(np.random.randint(-8, 7),np.random.randint(-8, 7)) for i in range(10000)], np.random.randint(-8, 7), np.random.randint(-8, 7), {"x": "encrypted", "y": "encrypted"})
TestAdd("int8ec",[(np.random.randint(-128, 127),np.random.randint(-128, 127)) for i in range(10000)], np.random.randint(-128, 127), np.random.randint(-128, 127), {"x": "encrypted", "y": "clear"})
TestAdd("int8ee",[(np.random.randint(-128, 127),np.random.randint(-128, 127)) for i in range(10000)], np.random.randint(-128, 127), np.random.randint(-128, 127), {"x": "encrypted", "y": "encrypted"})
TestAdd("int16ec",[(np.random.randint(-32768, 32767),np.random.randint(-32768, 32767)) for i in range(10000)], np.random.randint(-32768, 32767), np.random.randint(-32768, 32767), {"x": "encrypted", "y": "clear"})
TestAdd("int16ee",[(np.random.randint(-32768, 32767),np.random.randint(-32768, 32767)) for i in range(10000)], np.random.randint(-32768, 32767), np.random.randint(-32768, 32767), {"x": "encrypted", "y": "encrypted"})
TestAdd("int32ec",[(np.random.randint(-2147483648, 2147483647),np.random.randint(-2147483648, 2147483647)) for i in range(10000)], np.random.randint(-2147483648, 2147483647), np.random.randint(-2147483648, 2147483647), {"x": "encrypted", "y": "clear"})
TestAdd("int32ee",[(np.random.randint(-2147483648, 2147483647),np.random.randint(-2147483648, 2147483647)) for i in range(10000)], np.random.randint(-2147483648, 2147483647), np.random.randint(-2147483648, 2147483647), {"x": "encrypted", "y": "encrypted"})
# TestAdd("int64ec",[(np.random.randint(-9223372036854775808, 9223372036854775807),np.random.randint(-9223372036854775808, 9223372036854775807)) for i in range(10000)], np.random.randint(-9223372036854775808, 9223372036854775807), np.random.randint(-9223372036854775808, 9223372036854775807), {"x": "encrypted", "y": "clear"})
# TestAdd("int64ee",[(np.random.randint(-9223372036854775808, 9223372036854775807),np.random.randint(-9223372036854775808, 9223372036854775807)) for i in range(10000)], np.random.randint(-9223372036854775808, 9223372036854775807), np.random.randint(-9223372036854775808, 9223372036854775807), {"x": "encrypted", "y": "encrypted"})
# 打印时间字典
print("时间记录：")
count = 0  # 计数器，用于跟踪打印的键值对数量
for key, value in timings.items():
    print(key, ":", value)  # 打印键值对，不换行
    count += 1
    if count % 4 == 0:  # 每隔四个键值对
        print("\n\n")  # 打印一个空行

    



def TestSub(size,inputset, x, y,parameters: Mapping[str, Union[str, EncryptionStatus]]):
    @fhe.compiler(parameters)
    def sub(x, y):
        return x - y
    # 生成电路时间
    start_time = time.time()
    circuit = sub.compile(inputset, configuration=config)
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


print("*"*50,"TestSub Unsigned","*"*50)
TestSub("uint2ec",[(np.random.randint(0, 4),np.random.randint(0, 4)) for i in range(10000)], np.random.randint(0, 4), np.random.randint(0, 4), {"x": "encrypted", "y": "clear"})
TestSub("uint2ee",[(np.random.randint(0, 4),np.random.randint(0, 4)) for i in range(10000)], np.random.randint(0, 4), np.random.randint(0, 4), {"x": "encrypted", "y": "encrypted"})
TestSub("uint4ec",[(np.random.randint(0, 16),np.random.randint(0, 16)) for i in range(10000)], np.random.randint(0, 16), np.random.randint(0, 16), {"x": "encrypted", "y": "clear"})
TestSub("uint4ee",[(np.random.randint(0, 16),np.random.randint(0, 16)) for i in range(10000)], np.random.randint(0, 16), np.random.randint(0, 16), {"x": "encrypted", "y": "encrypted"})
TestSub("uint8ec",[(np.random.randint(0, 256),np.random.randint(0, 256)) for i in range(10000)], np.random.randint(0, 256), np.random.randint(0, 256), {"x": "encrypted", "y": "clear"})
TestSub("uint8ee",[(np.random.randint(0, 256),np.random.randint(0, 256)) for i in range(10000)], np.random.randint(0, 256), np.random.randint(0, 256), {"x": "encrypted", "y": "encrypted"})
TestSub("uint16ec",[(np.random.randint(0, 65536),np.random.randint(0, 65536)) for i in range(10000)], np.random.randint(0, 65536), np.random.randint(0, 65536), {"x": "encrypted", "y": "clear"})
TestSub("uint16ee",[(np.random.randint(0, 65536),np.random.randint(0, 65536)) for i in range(10000)], np.random.randint(0, 65536), np.random.randint(0, 65536), {"x": "encrypted", "y": "encrypted"})
TestSub("uint32ec",[(np.random.randint(0, 4294967296),np.random.randint(0, 4294967296)) for i in range(10000)], np.random.randint(0, 4294967296), np.random.randint(0, 4294967296), {"x": "encrypted", "y": "clear"})
TestSub("uint32ee",[(np.random.randint(0, 4294967296),np.random.randint(0, 4294967296)) for i in range(10000)], np.random.randint(0, 4294967296), np.random.randint(0, 4294967296), {"x": "encrypted", "y": "encrypted"})

print("*"*50,"TestSub Signed","*"*50)
TestSub("int2ec",[(np.random.randint(-2, 1),np.random.randint(-2, 1)) for i in range(10000)], np.random.randint(-2, 1), np.random.randint(-2, 1), {"x": "encrypted", "y": "clear"})
TestSub("int2ee",[(np.random.randint(-2, 1),np.random.randint(-2, 1)) for i in range(10000)], np.random.randint(-2, 1), np.random.randint(-2, 1), {"x": "encrypted", "y": "encrypted"})
TestSub("int4ec",[(np.random.randint(-8, 7),np.random.randint(-8, 7)) for i in range(10000)], np.random.randint(-8, 7), np.random.randint(-8, 7), {"x": "encrypted", "y": "clear"})
TestSub("int4ee",[(np.random.randint(-8, 7),np.random.randint(-8, 7)) for i in range(10000)], np.random.randint(-8, 7), np.random.randint(-8, 7), {"x": "encrypted", "y": "encrypted"})
TestSub("int8ec",[(np.random.randint(-128, 127),np.random.randint(-128, 127)) for i in range(10000)], np.random.randint(-128, 127), np.random.randint(-128, 127), {"x": "encrypted", "y": "clear"})
TestSub("int8ee",[(np.random.randint(-128, 127),np.random.randint(-128, 127)) for i in range(10000)], np.random.randint(-128, 127), np.random.randint(-128, 127), {"x": "encrypted", "y": "encrypted"})
TestSub("int16ec",[(np.random.randint(-32768, 32767),np.random.randint(-32768, 32767)) for i in range(10000)], np.random.randint(-32768, 32767), np.random.randint(-32768, 32767), {"x": "encrypted", "y": "clear"})
TestSub("int16ee",[(np.random.randint(-32768, 32767),np.random.randint(-32768, 32767)) for i in range(10000)], np.random.randint(-32768, 32767), np.random.randint(-32768, 32767), {"x": "encrypted", "y": "encrypted"})
TestSub("int32ec",[(np.random.randint(-2147483648, 2147483647),np.random.randint(-2147483648, 2147483647)) for i in range(10000)], np.random.randint(-2147483648, 2147483647), np.random.randint(-2147483648, 2147483647), {"x": "encrypted", "y": "clear"})
TestSub("int32ee",[(np.random.randint(-2147483648, 2147483647),np.random.randint(-2147483648, 2147483647)) for i in range(10000)], np.random.randint(-2147483648, 2147483647), np.random.randint(-2147483648, 2147483647), {"x": "encrypted", "y": "encrypted"})
# 打印时间字典
print("时间记录：")
count = 0  # 计数器，用于跟踪打印的键值对数量
for key, value in timings.items():
    print(key, ":", value)  # 打印键值对，不换行
    count += 1
    if count % 4 == 0:  # 每隔四个键值对
        print("\n\n")  # 打印一个空行