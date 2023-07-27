'''
-*- ecoding: utf-8 -*-
@Enviroment: ConcreteML
@ModuleName: Deploy
@Author: liang
@Time: 2023/7/23 19:54
@Software: PyCharm
功能描述: 使用Concrete的server和client部署电路
实现步骤:
结果：
'''
from concrete import fhe
# You can develop your circuit using the techniques discussed in previous chapters. Here is a simple example:
@fhe.compiler({"x": "encrypted"})
def function(x):
    return x + 42

inputset = range(10)
circuit = function.compile(inputset)
# Once you have your circuit, you can save everything the server needs:
circuit.server.save("server.zip")
# Then, send server.zip to your computation server.
# Setting up a server
server = fhe.Server.load("server.zip")
serialized_client_specs: str = server.client_specs.serialize()

# Setting up clients
client_specs = fhe.ClientSpecs.deserialize(serialized_client_specs)
client = fhe.Client(client_specs)
# Generating keys (on the client)
client.keys.generate()
# After serialization, send the evaluation keys to the server.
serialized_evaluation_keys: bytes = client.evaluation_keys.serialize()

# The next step is to encrypt your inputs and request the server to perform some computation.
# This can be done in the following way:
arg: fhe.Value = client.encrypt(7)
serialized_arg: bytes = arg.serialize()

# Performing computation (on the server)
deserialized_evaluation_keys = fhe.EvaluationKeys.deserialize(serialized_evaluation_keys)
deserialized_arg = fhe.Value.deserialize(serialized_arg)

result: fhe.Value = server.run(deserialized_arg, evaluation_keys=deserialized_evaluation_keys)
serialized_result: bytes = result.serialize()

# Decrypting the result (on the client)
deserialized_result = fhe.Value.deserialize(serialized_result)
decrypted_result = client.decrypt(deserialized_result)
assert decrypted_result == 49
print(decrypted_result)
