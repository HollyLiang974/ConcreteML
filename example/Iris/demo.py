'''
-*- ecoding: utf-8 -*-
@Enviroment: ConcreteML
@ModuleName: demo
@Author: Sakura
@Time: 2023/9/7 21:49
@Software: PyCharm
功能描述:将权重转为二进制推理
实现步骤:
结果：
'''
import time
import numpy as np
from concrete import fhe


nfeatures, nclasses = 576, 10
w_quant_L = np.random.randint(0,16, (nclasses, nfeatures))#(10,576)
w_bits = np.unpackbits(w_quant_L.astype(np.uint8), axis=0)
print(w_bits.shape) # (80, 576)
w_bits2 = []
# extract the relevant 4 bits
for row in range(10):
    w_bits2.append(w_bits[8*row+4:8*row+8,:])
w_bits2 = np.array(w_bits2)
w_bits2 = w_bits2.transpose(1,0,-1) # (4, 10, 576)

#Then compute the subsums :
block = np.random.randint(0,2, (16,16))
npatches = 36
nfilters = 16


lk = []
for i,lk_filter in enumerate(block.transpose()):  # on each filter
    lk.extend([fhe.LookupTable(lk_filter)] * npatches)
    # lk.extend([fhe.LookupTable(lk_filter*np.random.randint(0,16,1))] * npatches)  # npatches per filter
tables = fhe.LookupTable(lk)


N = 12  # 18 12 = number of subsums
end = 48  # 32 48 = nfeatures // N
X_train = np.random.randint(0,16, (1000, nfeatures)).astype(np.uint8)


cfg = fhe.Configuration(p_error=0.1, show_graph=True)

@fhe.compiler({"x": "encrypted"})
def h(x):
    y = tables[x]
    for t in range(3):
        for num_bits in range(4):
            w = w_bits2[num_bits].transpose()
            res = np.expand_dims(y[int(N * 0):int(N + N * 0)] @ w[int(N * 0):int(N + N * 0), :],axis=0)  # + b
            start = 1
            for i in range(start, end):
                res = np.concatenate((res, np.expand_dims(y[int(N * i):int(N + N * i)] @ w[int(N * i):int(N + N * i)],axis=0)), axis=0)
           # 48, 10

            if t == 0 and num_bits == 0:
                res_f = np.expand_dims(res, axis=0)
            else:
                res_f = np.concatenate((res_f, np.expand_dims(res, axis=0)), axis = 0)

    print(res_f.shape)
    return res_f # np.greater(res,0)#res #(res>0)*1.0

inputset = X_train[0:4,:]#[np.zeros(576).astype(np.uint8), (16 * np.ones(576)).astype(np.uint8)] # X_train[0:4,:]
inpt = X_train[5,:]
print("compiling")
t = time.time()
circuit = h.compile(inputset, configuration=cfg)
print('compiled in ', time.time()-t)
t = time.time()
circuit.keygen()
print("Keygen done in ", time.time()-t)
enc = circuit.encrypt(inpt)
t = time.time()
res_enc = circuit.run(enc)
print(time.time()-t)

cloud_output = circuit.decrypt(res_enc)

#Then sum the results as W = 23 * w_bits2[0] + 22 * w_bits[1] + 21 * w_bits2[2] + 20 * w_bits[0]:
imgs_fin = np.sum(cloud_output,axis=1)
#print(imgs_fin.shape)
imgs_fin = 2**3 * imgs_fin[0] + 2**2 * imgs_fin[1] + 2**1 * imgs_fin[2] + 2**0 * imgs_fin[3]
#print(imgs_fin.shape)

cloud_pred = np.argmax(imgs_fin,axis=0)
print("cloud_pred",cloud_pred)
#So this takes approx 5s (which is even less than the function that calls only the lookup table).