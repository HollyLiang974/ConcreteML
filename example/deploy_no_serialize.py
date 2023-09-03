'''
-*- ecoding: utf-8 -*-
@Enviroment: ConcreteML
@ModuleName: test_deploy
@Author: Sakura
@Time: 2023/9/2 17:55
@Software: PyCharm
功能描述:
实现步骤:
结果：
'''
from concrete import fhe
# You can develop your circuit using the techniques discussed in previous chapters. Here is a simple example:
y = 42
@fhe.compiler({"x": "encrypted"})
def function(x):
    return x + y
inputset = range(10)
circuit = function.compile(inputset)
server=circuit.server
client=circuit.client
keys=client.keys.generate()
evaluation_keys=client.evaluation_keys
# 加密数据7
arg: fhe.Value = client.encrypt(7)

result: fhe.Value = server.run(arg, evaluation_keys=evaluation_keys)
decrypted_result = client.decrypt(result)
assert decrypted_result == 49