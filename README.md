

<p align="center">
    <a href="https://github.com/HongfeiWan/DeepVibration" target="_blank">
        <img src="https://github.com/HongfeiWan/DeepVibration/blob/main/imgaes/Logo.png" width="1000">
    </a>
</p>
<p align="center">
    <a href="https://github.com/pytorch/pytorch">
        <img src="https://img.shields.io/badge/pytorch-2.8.0-brightgreen.svg">
    </a>
    <a href="https://github.com/NVIDIA/cuda-python">
        <img src="https://img.shields.io/badge/cudapython-13.0.1-brightgreen.svg">
    </a>
    <a href="https://github.com/matlab">
        <img src="https://img.shields.io/badge/Matlab-2024a-brightgreen.svg">
    </a>
</p>



# DeepVibration

## 项目概述
DeepVibration主要用于CEvNS（相干弹性中微子-原子核散射）实验的数据分析。

## 主要程序：makeTreeSimpleDZL_V1725.C

### 程序功能
该程序是一个C++程序，用于读取V1725数据采集卡的原始二进制数据文件，进行数据处理和拟合分析，并输出ROOT格式的数据树。

### 输入输出

#### 输入
- **输入文件**: 二进制格式的FADC原始数据文件
- **文件路径**: `/st0/share/data/raw_data/CEvNS_DZL_test_sanmen/20250516/`
- **文件命名格式**: `FADC_RAW_Data_{运行编号}.bin`

#### 输出
- **ROOT文件**: 包含处理后的数据树
- **保存路径**: `/st0/home/wanhf/Data`
- **文件命名格式**: `{日期}_{运行编号}.root`
- **文本文件**: `20250516_DZL_pulse_RT.txt` - 包含脉冲数据

### 可配置参数

#### 基本配置参数
```cpp
const unsigned int CHANNEL_NUMBER = 16;        // 通道数量
const unsigned int EVENT_NUMBER = 10000;       // 每个bin存储的事件数量
const unsigned int MAX_WINDOWS = 30000;        // 时间窗口大小 (120μs)
const unsigned int PED_RANGE = 1000;           // 基线计算范围（前1000个点）
```

#### 快放拟合参数
```cpp
const unsigned int FIT_RANGE = 5500;           // 快放拟合的拟合范围
const unsigned int fit_start = 6000;           // 快放拟合的开始时间
const unsigned int PED_RANGE_FP = 1000;        // 快放拟合的基线范围
const unsigned int TRY_NUMBER = 10;            // 快放拟合的尝试次数
```

#### 电荷量积分范围
```cpp
// 通道0的积分范围（10次尝试）
const unsigned int Qp0_RANDN[TRY_NUMBER] = {3800,3700,3600,3500,4000,4000,4000,3600,3700,3700};
const unsigned int Qp0_RANUP[TRY_NUMBER] = {6700,6700,6700,6700,6700,6600,6900,6900,6900,7000};

// 通道1的积分范围（10次尝试）
const unsigned int Qp1_RANDN[TRY_NUMBER] = {4300,4400,4500,4600,4600,4600,4700,4800,4900,5000};
const unsigned int Qp1_RANUP[TRY_NUMBER] = {8000,8500,8600,8700,8800,8900,9000,9000,9000,9000};

// 其他通道的固定积分范围
const unsigned int Qp_RANDN[CHANNEL_NUMBER] = {900,1800,500,3000,4990,4990,0,0,0,0,0,0,0,0};
const unsigned int Qp_RANUP[CHANNEL_NUMBER] = {1500,3000,4500,5500,5005,5005,0,0,0,0,0,0,0,0};
```

#### 阈值参数
```cpp
const unsigned int THRESHOLD_AC = 1450;        // 阈值设置
```

### 中间输出变量

#### 事件级别变量
- `idevt`: 事件编号
- `trig`: 触发记录
- `time`: 时间（相对时间，相对于RAND START TIME）
- `deadtime`: 死时间
- `TimeV1725`: V1725时间
- `TTTV1725`: 绝对时间

#### 通道级别变量
- `ped[CHANNEL_NUMBER]`: 前沿基线
- `pedt[CHANNEL_NUMBER]`: 后沿基线
- `q[CHANNEL_NUMBER]`: 全时间窗的积分值
- `max[CHANNEL_NUMBER]`: 最大值
- `maxpt[CHANNEL_NUMBER]`: 最大值时间点
- `min[CHANNEL_NUMBER]`: 最小值
- `minpt[CHANNEL_NUMBER]`: 最小值时间点
- `tb[CHANNEL_NUMBER]`: 时间窗
- `rms[CHANNEL_NUMBER]`: 均方根
- `Qp[CHANNEL_NUMBER]`: 积分值（用于快放拟合）
- `Ped_rms[CHANNEL_NUMBER]`: 基线均方根

#### 快放拟合变量（当NOFIT未定义时）
- `chi2`: 拟合的χ²值
- `famp`: 拟合幅度
- `eamp`: 幅度误差
- `fped`: 拟合基线
- `eped`: 基线误差
- `fcross`: 拟合交叉点
- `ecross`: 交叉点误差
- `fslope`: 拟合斜率
- `eslope`: 斜率误差
- `fmid`: 拟合中点
- `fit_flag`: 拟合标志

### 可修改参数说明

#### 1. 数据采集参数
- **CHANNEL_NUMBER**: 控制处理的通道数量
- **EVENT_NUMBER**: 控制每个运行处理的事件数量
- **MAX_WINDOWS**: 控制时间窗口大小，影响数据精度和内存使用

#### 2. 基线计算参数
- **PED_RANGE**: 控制基线计算使用的数据点数量，影响基线精度
- **PED_RANGE_FP**: 快放拟合的基线范围，针对特定通道优化

#### 3. 积分范围参数
- **Qp0_RANDN/Qp0_RANUP**: 通道0的积分范围，可针对不同实验条件调整
- **Qp1_RANDN/Qp1_RANUP**: 通道1的积分范围
- **Qp_RANDN/Qp_RANUP**: 其他通道的固定积分范围

#### 4. 拟合参数
- **FIT_RANGE**: 控制拟合使用的数据点范围
- **fit_start**: 控制拟合开始的时间点
- **TRY_NUMBER**: 控制拟合尝试次数，影响拟合成功率

#### 5. 阈值参数
- **THRESHOLD_AC**: 控制触发阈值，影响事件选择

### 使用方法

```bash
./makeTreeSimpleDZL_V1725 [输入文件名] [运行开始编号] [运行结束编号] [杂项参数]
```

### 依赖项
- ROOT框架（TFile, TTree, TF1, TH1F, TGraph）
- 自定义头文件：misc.h, tanh_fit.h

### 注意事项
1. 程序需要ROOT环境支持
2. 输入文件路径和输出路径需要根据实际环境调整
3. 拟合参数需要根据具体实验条件优化
4. 通道数量和相关参数需要与硬件配置匹配
