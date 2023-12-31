# 线性操作性能

| PTQ操作                          | 时间（秒）            | QAT操作                       | 时间（秒）            |
| -------------------------------- | --------------------- | ----------------------------- | --------------------- |
| Linear（2）largest bitwidth(3)   | 0.0008285045623779297 | Linear(2) largest bitwidth(7) | 0.0012514591217041016 |
| Linear（4）largest bitwidth(8)   | 0.0007781982421875    | Linear(4)largest bitwidth(11) | 0.0010149478912353516 |
| Linear（8）largest bitwidth(16)  | 0.0006074905395507812 | Linear（8）                   | 0.0007307529449462891 |
| Linear（16）largest bitwidth(32) | 0.0007030963897705078 | Linear（16）                  | 0.0036840438842773438 |
| Conv2d（2）                      | 0.9543304443359375    | Conv2d（2）                   | 0.9430763721466064    |
| Conv2d（4）                      | 1.2422044277191162    | Conv2d（4）                   | 1.2589774131774902    |
| Conv2d（8）                      | 1.9727725982666016    | Conv2d（8）                   | 1.9957025051116943    |
| Conv2d（16）                     | 3.3297104835510254    | Conv2d（16）                  | 3.579220771789551     |



## 1. Linear

### 1.1 PTQLinear

```
编译时间 0.11677122116088867
量化位数： 2 最大量化位数 3
推理时间 0.0008285045623779297
输入值： [[-1.6186525  4.128488 ]] 输出值: [[-1.68564873]]
**************************************************
编译时间 0.06408810615539551
量化位数： 4 最大量化位数 8
推理时间 0.0007781982421875
输入值： [[-0.40512258  0.00630411]] 输出值: [[-0.45341152]]
**************************************************
编译时间 0.0670619010925293
量化位数： 8 最大量化位数 16
推理时间 0.0006074905395507812
输入值： [[ 0.70389926 -1.576674  ]] 输出值: [[0.89179501]]
**************************************************
编译时间 0.04557514190673828
量化位数： 16 最大量化位数 32
推理时间 0.0007030963897705078
输入值： [[0.03412776 1.2772576 ]] 输出值: [[-0.94799256]]
**************************************************
```

### 1.2 QATLinear

```
编译时间 0.10905814170837402
量化位数： 2 最大量化位数 7
推理时间 0.0012514591217041016
输入值： [[-0.7754536   0.07886824]] 输出值: [[0.73649624]]
**************************************************
编译时间 0.10269546508789062
量化位数： 4 最大量化位数 11
推理时间 0.0010149478912353516
输入值： [[ 0.611176  -1.8614094]] 输出值: [[-0.13759456]]
**************************************************
编译时间 0.09854841232299805
量化位数： 8 最大量化位数 16
推理时间 0.0007307529449462891
输入值： [[-0.40439552  1.155109  ]] 输出值: [[0.31131517]]
**************************************************
编译时间 0.09077978134155273
量化位数： 16 最大量化位数 24
推理时间 0.0036840438842773438
输入值： [[ 2.4878087 -0.5385189]] 输出值: [[1.27008193]]
**************************************************
```

## 2. Conv2d

### 2.1 PTQConv2d

```
编译时间 0.237898588180542
量化位数： 2 最大量化位数 6
推理时间 0.9543304443359375
**************************************************
编译时间 0.19905948638916016
量化位数： 4 最大量化位数 10
推理时间 1.2422044277191162
**************************************************
编译时间 0.18508386611938477
量化位数： 8 最大量化位数 18
推理时间 1.9727725982666016
**************************************************
编译时间 0.25738024711608887
量化位数： 16 最大量化位数 34
推理时间 3.3297104835510254
**************************************************
```

### 2.2 QATConv2d

```
编译时间 0.42847418785095215
量化位数： 2 最大量化位数 7
推理时间 0.9430763721466064
**************************************************
编译时间 0.2637662887573242
量化位数： 4 最大量化位数 11
推理时间 1.2589774131774902
**************************************************
编译时间 0.2489628791809082
量化位数： 8 最大量化位数 20
推理时间 1.9957025051116943
**************************************************
编译时间 0.2559230327606201
量化位数： 16 最大量化位数 36
推理时间 3.579220771789551
**************************************************
```

