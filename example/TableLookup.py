'''
-*- ecoding: utf-8 -*-
@Enviroment: ConcreteML
@ModuleName: TableLookup
@Author: Sakura
@Time: 2023/7/18 14:59
@Software: PyCharm
功能描述: 不同位宽的查表操作
实现步骤:
结果：
'''
import time

from concrete import fhe


def TestTableLookUp(inputset, table,x):
    @fhe.compiler({"x": "encrypted"})
    def f(x):
        return table[x]

    circuit = f.compile(inputset)
    print(circuit)
    start_time = time.time()
    print("结果:", circuit.encrypt_run_decrypt(x))
    end_time = time.time()
    print("时间:", end_time - start_time)
    return 0

TestTableLookUp(range(4),fhe.LookupTable(range(4)),0)
TestTableLookUp(range(16),fhe.LookupTable(range(16)),0)
TestTableLookUp(range(256),fhe.LookupTable(range(256)),0)
#TestTableLookUp(range(65536),fhe.LookupTable(range(65536)),0)


