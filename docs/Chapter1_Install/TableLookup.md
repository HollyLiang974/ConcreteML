# TableLookup

> 查询列表的位宽越长，查询时间越长
> 

| 查表范围 | 时间（秒）         |
| -------- | ------------------ |
| uint2    | 0.953528881072998  |
| uint4    | 2.4814624786376953 |
| uint8    | 210.20805501937866 |



```
%0 = x                               # EncryptedScalar<uint2>        ∈ [0, 3]
%1 = tlu(%0, table=[0 1 2 3])        # EncryptedScalar<uint2>        ∈ [0, 3]
return %1
结果: 0
时间: 0.953528881072998
%0 = x                                               # EncryptedScalar<uint4>        ∈ [0, 15]
%1 = tlu(%0, table=[ 0  1  2  ...  13 14 15])        # EncryptedScalar<uint4>        ∈ [0, 15]
return %1
结果: 0
时间: 2.4814624786376953
%0 = x                                               # EncryptedScalar<uint8>        ∈ [0, 255]
%1 = tlu(%0, table=[  0   1   ... 3 254 255])        # EncryptedScalar<uint8>        ∈ [0, 255]
return %1
结果: 0
时间: 210.20805501937866
```

