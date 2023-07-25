# FHE基础操作的性能，密**+/***明、密**+/***密（**2/4/8/16**位）

> 总结：密文$+-*/$密文时间>密文$+-*/$明文
>
> 位宽越长，计算时间越长

```
/home/holly/anaconda3/envs/concreteML/bin/python /home/holly/concrete-ml/operations.py 
************************************************** TestAdd **************************************************
%0 = x                  # EncryptedScalar<int2>        ∈ [-2, 1]
%1 = y                  # ClearScalar<int2>            ∈ [-2, 1]
%2 = add(%0, %1)        # EncryptedScalar<int1>        ∈ [-1, -1]
return %2
结果: 1
时间: 0.0003695487976074219
%0 = x                  # EncryptedScalar<int2>        ∈ [-2, 1]
%1 = y                  # EncryptedScalar<int2>        ∈ [-2, 1]
%2 = add(%0, %1)        # EncryptedScalar<int1>        ∈ [-1, -1]
return %2
结果: 1
时间: 0.0003457069396972656
%0 = x                  # EncryptedScalar<int4>        ∈ [-8, 7]
%1 = y                  # ClearScalar<int4>            ∈ [-8, 7]
%2 = add(%0, %1)        # EncryptedScalar<int1>        ∈ [-1, -1]
return %2
结果: 1
时间: 0.0005078315734863281
%0 = x                  # EncryptedScalar<int4>        ∈ [-8, 7]
%1 = y                  # EncryptedScalar<int4>        ∈ [-8, 7]
%2 = add(%0, %1)        # EncryptedScalar<int1>        ∈ [-1, -1]
return %2
结果: 1
时间: 0.00043582916259765625
%0 = x                  # EncryptedScalar<int8>        ∈ [-128, 127]
%1 = y                  # ClearScalar<int8>            ∈ [-128, 127]
%2 = add(%0, %1)        # EncryptedScalar<int1>        ∈ [-1, -1]
return %2
结果: 1
时间: 0.0008387565612792969
%0 = x                  # EncryptedScalar<int8>        ∈ [-128, 127]
%1 = y                  # EncryptedScalar<int8>        ∈ [-128, 127]
%2 = add(%0, %1)        # EncryptedScalar<int1>        ∈ [-1, -1]
return %2
结果: 1
时间: 0.0005042552947998047
%0 = x                  # EncryptedScalar<int16>        ∈ [-32768, 32767]
%1 = y                  # ClearScalar<int16>            ∈ [-32768, 32767]
%2 = add(%0, %1)        # EncryptedScalar<int1>         ∈ [-1, -1]
return %2
结果: 1
时间: 0.00029468536376953125
%0 = x                  # EncryptedScalar<int16>        ∈ [-32768, 32767]
%1 = y                  # EncryptedScalar<int16>        ∈ [-32768, 32767]
%2 = add(%0, %1)        # EncryptedScalar<int1>         ∈ [-1, -1]
return %2
结果: 1
时间: 0.0003631114959716797
************************************************** TestSub **************************************************
%0 = x                       # EncryptedScalar<int2>        ∈ [-2, 1]
%1 = y                       # ClearScalar<int2>            ∈ [-2, 1]
%2 = subtract(%0, %1)        # EncryptedScalar<int3>        ∈ [-3, 3]
return %2
结果: -1
时间: 0.00041985511779785156
%0 = x                       # EncryptedScalar<int2>        ∈ [-2, 1]
%1 = y                       # EncryptedScalar<int2>        ∈ [-2, 1]
%2 = subtract(%0, %1)        # EncryptedScalar<int3>        ∈ [-3, 3]
return %2
结果: -1
时间: 0.0005280971527099609
%0 = x                       # EncryptedScalar<int4>        ∈ [-8, 7]
%1 = y                       # ClearScalar<int4>            ∈ [-8, 7]
%2 = subtract(%0, %1)        # EncryptedScalar<int5>        ∈ [-15, 15]
return %2
结果: -1
时间: 0.00041985511779785156
%0 = x                       # EncryptedScalar<int4>        ∈ [-8, 7]
%1 = y                       # EncryptedScalar<int4>        ∈ [-8, 7]
%2 = subtract(%0, %1)        # EncryptedScalar<int5>        ∈ [-15, 15]
return %2
结果: -1
时间: 0.0005004405975341797
%0 = x                       # EncryptedScalar<int8>        ∈ [-128, 127]
%1 = y                       # ClearScalar<int8>            ∈ [-128, 127]
%2 = subtract(%0, %1)        # EncryptedScalar<int9>        ∈ [-255, 255]
return %2
结果: -1
时间: 0.000308990478515625
%0 = x                       # EncryptedScalar<int8>        ∈ [-128, 127]
%1 = y                       # EncryptedScalar<int8>        ∈ [-128, 127]
%2 = subtract(%0, %1)        # EncryptedScalar<int9>        ∈ [-255, 255]
return %2
结果: -1
时间: 0.00036835670471191406
%0 = x                       # EncryptedScalar<int16>        ∈ [-32768, 32767]
%1 = y                       # ClearScalar<int16>            ∈ [-32768, 32767]
%2 = subtract(%0, %1)        # EncryptedScalar<int17>        ∈ [-65535, 65535]
return %2
结果: -1
时间: 0.0004363059997558594
%0 = x                       # EncryptedScalar<int16>        ∈ [-32768, 32767]
%1 = y                       # EncryptedScalar<int16>        ∈ [-32768, 32767]
%2 = subtract(%0, %1)        # EncryptedScalar<int17>        ∈ [-65535, 65535]
return %2
结果: -1
时间: 0.0003180503845214844
************************************************** TestMul **************************************************
%0 = x                       # EncryptedScalar<int2>        ∈ [-2, 1]
%1 = y                       # ClearScalar<int2>            ∈ [-2, 1]
%2 = multiply(%0, %1)        # EncryptedScalar<int2>        ∈ [-2, -2]
return %2
结果: 0
时间: 0.0005311965942382812
%0 = x                       # EncryptedScalar<int2>        ∈ [-2, 1]
%1 = y                       # EncryptedScalar<int2>        ∈ [-2, 1]
%2 = multiply(%0, %1)        # EncryptedScalar<int2>        ∈ [-2, -2]
return %2
结果: 0
时间: 1.3543510437011719
%0 = x                       # EncryptedScalar<int4>        ∈ [-8, 7]
%1 = y                       # ClearScalar<int4>            ∈ [-8, 7]
%2 = multiply(%0, %1)        # EncryptedScalar<int7>        ∈ [-56, -56]
return %2
结果: 0
时间: 0.00028324127197265625
%0 = x                       # EncryptedScalar<int4>        ∈ [-8, 7]
%1 = y                       # EncryptedScalar<int4>        ∈ [-8, 7]
%2 = multiply(%0, %1)        # EncryptedScalar<int7>        ∈ [-56, -56]
return %2
结果: 0
时间: 60.93878197669983
************************************************** TestDiv **************************************************
%0 = x                           # EncryptedScalar<int2>        ∈ [-2, -2]
%1 = 1                           # ClearScalar<uint1>           ∈ [1, 1]
%2 = floor_divide(%1, %0)        # EncryptedScalar<int1>        ∈ [-1, -1]
return %2
结果: 0
时间: 0.9732739925384521
%0 = x                           # EncryptedScalar<int4>        ∈ [-8, -8]
%1 = 1                           # ClearScalar<uint1>           ∈ [1, 1]
%2 = floor_divide(%1, %0)        # EncryptedScalar<int1>        ∈ [-1, -1]
return %2
结果: 0
时间: 2.5715441703796387
%0 = x                           # EncryptedScalar<int8>        ∈ [-128, -128]
%1 = 1                           # ClearScalar<uint1>           ∈ [1, 1]
%2 = floor_divide(%1, %0)        # EncryptedScalar<int1>        ∈ [-1, -1]
return %2
结果: 0
时间: 210.9334361553192

进程已结束,退出代码0

```

