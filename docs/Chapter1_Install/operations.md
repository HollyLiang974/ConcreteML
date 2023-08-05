# FHE基础操作的性能，密**+/***明、密**+/***密（**2/4/8/16**位）

> 总结：密文$+-*/$密文时间>密文$+-*/$明文
>
> 位宽越长，计算时间越长
> 

| 操作（明文&密文）  | 时间（秒）             | 操作（密文&密文）  | 时间（秒）             |
| ------------------ | ---------------------- | ------------------ | :--------------------- |
| 明文+密文（int2）  | 0.00046443939208984375 | 密文+密文（int2）  | 0.0003917217254638672  |
| 明文+密文（int4）  | 0.00030303001403808594 | 密文+密文（int4）  | 0.0003383159637451172  |
| 明文+密文（int8）  | 0.0004138946533203125  | 密文+密文（int8）  | 0.0003552436828613281  |
| 明文+密文（int16） | 0.00028228759765625    | 密文+密文（int16） | 0.0004153251647949219  |
| 明文-密文（int2）  | 0.00033164024353027344 | 密文-密文（int2）  | 0.00036215782165527344 |
| 明文-密文（int4）  | 0.0002846717834472656  | 密文-密文（int4）  | 0.0003123283386230469  |
| 明文-密文（int8）  | 0.00033473968505859375 | 密文-密文（int8）  | 0.0004589557647705078  |
| 明文-密文（int16） | 0.0004210472106933594  | 密文-密文（int16） | 0.0004055500030517578  |
| 明文*密文（int2）  | 0.0003108978271484375  | 密文*密文（int2）  | 1.0131540298461914     |
| 明文*密文（int4）  | 0.0002846717834472656  | 密文*密文（int4）  | 26.822704315185547     |
| 明文*密文（int8）  | -                      | 密文*密文（int8）  | -                      |
| 明文*密文（int16） | -                      | 密文*密文（int16） | -                      |
| 明文/密文（int2）  | 0.7009823322296143     | -                  | -                      |
| 明文/密文（int4）  | 1.552710771560669      | -                  | -                      |
| 明文/密文（int8）  | 115.94496941566467     | -                  | -                      |
| 明文/密文（int16） | -                      | -                  | -                      |
| 密文/明文（int2）  | 0.6185073852539062     | -                  | -                      |
| 密文/明文（int4）  | 1.5069186687469482     | -                  | -                      |
| 密文/明文（int8）  | 120.8521785736084      | -                  | -                      |
| 密文/明文（int16） | -                      | -                  | -                      |

```
/home/holly/anaconda3/envs/concrete-ml/bin/python /home/holly/concrete-ml/example/operations.py 
************************************************** TestAdd **************************************************
%0 = x                  # EncryptedScalar<int2>        ∈ [-2, 1]
%1 = y                  # ClearScalar<int2>            ∈ [-2, 1]
%2 = add(%0, %1)        # EncryptedScalar<int1>        ∈ [-1, -1]
return %2
结果: 1
时间: 0.00046443939208984375
%0 = x                  # EncryptedScalar<int2>        ∈ [-2, 1]
%1 = y                  # EncryptedScalar<int2>        ∈ [-2, 1]
%2 = add(%0, %1)        # EncryptedScalar<int1>        ∈ [-1, -1]
return %2
结果: 1
时间: 0.0003917217254638672
%0 = x                  # EncryptedScalar<int4>        ∈ [-8, 7]
%1 = y                  # ClearScalar<int4>            ∈ [-8, 7]
%2 = add(%0, %1)        # EncryptedScalar<int1>        ∈ [-1, -1]
return %2
结果: 1
时间: 0.00030303001403808594
%0 = x                  # EncryptedScalar<int4>        ∈ [-8, 7]
%1 = y                  # EncryptedScalar<int4>        ∈ [-8, 7]
%2 = add(%0, %1)        # EncryptedScalar<int1>        ∈ [-1, -1]
return %2
结果: 1
时间: 0.0003383159637451172
%0 = x                  # EncryptedScalar<int8>        ∈ [-128, 127]
%1 = y                  # ClearScalar<int8>            ∈ [-128, 127]
%2 = add(%0, %1)        # EncryptedScalar<int1>        ∈ [-1, -1]
return %2
结果: 1
时间: 0.0004138946533203125
%0 = x                  # EncryptedScalar<int8>        ∈ [-128, 127]
%1 = y                  # EncryptedScalar<int8>        ∈ [-128, 127]
%2 = add(%0, %1)        # EncryptedScalar<int1>        ∈ [-1, -1]
return %2
结果: 1
时间: 0.0003552436828613281
%0 = x                  # EncryptedScalar<int16>        ∈ [-32768, 32767]
%1 = y                  # ClearScalar<int16>            ∈ [-32768, 32767]
%2 = add(%0, %1)        # EncryptedScalar<int1>         ∈ [-1, -1]
return %2
结果: 1
时间: 0.00028228759765625
%0 = x                  # EncryptedScalar<int16>        ∈ [-32768, 32767]
%1 = y                  # EncryptedScalar<int16>        ∈ [-32768, 32767]
%2 = add(%0, %1)        # EncryptedScalar<int1>         ∈ [-1, -1]
return %2
结果: 1
时间: 0.0004055500030517578
************************************************** TestSub **************************************************
%0 = x                       # EncryptedScalar<int2>        ∈ [-2, 1]
%1 = y                       # ClearScalar<int2>            ∈ [-2, 1]
%2 = subtract(%0, %1)        # EncryptedScalar<int3>        ∈ [-3, 3]
return %2
结果: -1
时间: 0.00033164024353027344
%0 = x                       # EncryptedScalar<int2>        ∈ [-2, 1]
%1 = y                       # EncryptedScalar<int2>        ∈ [-2, 1]
%2 = subtract(%0, %1)        # EncryptedScalar<int3>        ∈ [-3, 3]
return %2
结果: -1
时间: 0.00036215782165527344
%0 = x                       # EncryptedScalar<int4>        ∈ [-8, 7]
%1 = y                       # ClearScalar<int4>            ∈ [-8, 7]
%2 = subtract(%0, %1)        # EncryptedScalar<int5>        ∈ [-15, 15]
return %2
结果: -1
时间: 0.00026154518127441406
%0 = x                       # EncryptedScalar<int4>        ∈ [-8, 7]
%1 = y                       # EncryptedScalar<int4>        ∈ [-8, 7]
%2 = subtract(%0, %1)        # EncryptedScalar<int5>        ∈ [-15, 15]
return %2
结果: -1
时间: 0.0003123283386230469
%0 = x                       # EncryptedScalar<int8>        ∈ [-128, 127]
%1 = y                       # ClearScalar<int8>            ∈ [-128, 127]
%2 = subtract(%0, %1)        # EncryptedScalar<int9>        ∈ [-255, 255]
return %2
结果: -1
时间: 0.00033473968505859375
%0 = x                       # EncryptedScalar<int8>        ∈ [-128, 127]
%1 = y                       # EncryptedScalar<int8>        ∈ [-128, 127]
%2 = subtract(%0, %1)        # EncryptedScalar<int9>        ∈ [-255, 255]
return %2
结果: -1
时间: 0.0004589557647705078
%0 = x                       # EncryptedScalar<int16>        ∈ [-32768, 32767]
%1 = y                       # ClearScalar<int16>            ∈ [-32768, 32767]
%2 = subtract(%0, %1)        # EncryptedScalar<int17>        ∈ [-65535, 65535]
return %2
结果: -1
时间: 0.0004210472106933594
%0 = x                       # EncryptedScalar<int16>        ∈ [-32768, 32767]
%1 = y                       # EncryptedScalar<int16>        ∈ [-32768, 32767]
%2 = subtract(%0, %1)        # EncryptedScalar<int17>        ∈ [-65535, 65535]
return %2
结果: -1
时间: 0.0004153251647949219
************************************************** TestMul **************************************************
%0 = x                       # EncryptedScalar<int2>        ∈ [-2, 1]
%1 = y                       # ClearScalar<int2>            ∈ [-2, 1]
%2 = multiply(%0, %1)        # EncryptedScalar<int2>        ∈ [-2, -2]
return %2
结果: 0
时间: 0.0003108978271484375
%0 = x                       # EncryptedScalar<int2>        ∈ [-2, 1]
%1 = y                       # EncryptedScalar<int2>        ∈ [-2, 1]
%2 = multiply(%0, %1)        # EncryptedScalar<int2>        ∈ [-2, -2]
return %2
结果: 0
时间: 1.0131540298461914
%0 = x                       # EncryptedScalar<int4>        ∈ [-8, 7]
%1 = y                       # ClearScalar<int4>            ∈ [-8, 7]
%2 = multiply(%0, %1)        # EncryptedScalar<int7>        ∈ [-56, -56]
return %2
结果: 0
时间: 0.0002846717834472656
%0 = x                       # EncryptedScalar<int4>        ∈ [-8, 7]
%1 = y                       # EncryptedScalar<int4>        ∈ [-8, 7]
%2 = multiply(%0, %1)        # EncryptedScalar<int7>        ∈ [-56, -56]
return %2
结果: 0
时间: 26.822704315185547
************************************************** TestDiv **************************************************
%0 = x                           # EncryptedScalar<int2>        ∈ [-2, -2]
%1 = 1                           # ClearScalar<uint1>           ∈ [1, 1]
%2 = floor_divide(%1, %0)        # EncryptedScalar<int1>        ∈ [-1, -1]
return %2
结果: 0
时间: 0.7009823322296143
%0 = x                           # EncryptedScalar<int4>        ∈ [-8, -8]
%1 = 1                           # ClearScalar<uint1>           ∈ [1, 1]
%2 = floor_divide(%1, %0)        # EncryptedScalar<int1>        ∈ [-1, -1]
return %2
结果: 0
时间: 1.552710771560669
%0 = x                           # EncryptedScalar<int8>        ∈ [-128, -128]
%1 = 1                           # ClearScalar<uint1>           ∈ [1, 1]
%2 = floor_divide(%1, %0)        # EncryptedScalar<int1>        ∈ [-1, -1]
return %2
结果: 0
时间: 115.94496941566467
%0 = x                           # EncryptedScalar<int2>        ∈ [-2, -2]
%1 = 1                           # ClearScalar<uint1>           ∈ [1, 1]
%2 = floor_divide(%0, %1)        # EncryptedScalar<int2>        ∈ [-2, -2]
return %2
结果: 0
时间: 0.6185073852539062
%0 = x                           # EncryptedScalar<int4>        ∈ [-8, -8]
%1 = 1                           # ClearScalar<uint1>           ∈ [1, 1]
%2 = floor_divide(%0, %1)        # EncryptedScalar<int4>        ∈ [-8, -8]
return %2
结果: 0
时间: 1.5069186687469482
%0 = x                           # EncryptedScalar<int8>        ∈ [-128, -128]
%1 = 1                           # ClearScalar<uint1>           ∈ [1, 1]
%2 = floor_divide(%0, %1)        # EncryptedScalar<int8>        ∈ [-128, -128]
return %2
结果: 0
时间: 120.8521785736084

进程已结束,退出代码0
```

