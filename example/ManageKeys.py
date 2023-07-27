'''
-*- ecoding: utf-8 -*-
@Enviroment: ConcreteML
@ModuleName: ManageKeys
@Author: liang
@Time: 2023/7/23 18:58
@Software: PyCharm
功能描述: generates encryption/decryption keys and evaluation keys
实现步骤:
结果：
'''
import time
from concrete import fhe

@fhe.compiler({"x": "encrypted"})
def f(x):
    return x ** 2

inputset = range(10)
circuit = f.compile(inputset)

# 为电路生成密钥
# 要为电路明确地生成密钥，您可以使用：
circuit.keys.generate()
# 而且可以设置自定义种子以实现可复现性。在生产环境中不要手动指定种子！
# circuit.keys.generate(seed=420)

# 要序列化密钥，例如在网络上传输：
# serialized_keys: bytes = circuit.keys.serialize()

# 要将密钥反序列化回来，在接收到序列化密钥后：
# keys: fhe.Keys = fhe.Keys.deserialize(serialized_keys)

# 一旦您拥有一个有效的 fhe.Keys 对象，您可以直接将其分配给电路：
# circuit.keys = keys

# 您还可以直接使用文件系统存储密钥，无需自行处理序列化和文件管理：
# 密钥不会以加密形式保存！请确保将它们存储在安全环境中，或在保存后手动加密。
circuit.keys.save("./key/keys")

# 在密钥保存到磁盘后，您可以通过以下方式将它们加载回来：
circuit.keys.load("./key/keys")

# 如果您想在第一次运行时生成密钥，并在后续运行中重复使用这些密钥：
# circuit.keys.load_if_exists_generate_and_save_otherwise("./keys")