
<p align="center">
    <a href="https://github.com/HongfeiWan/DeepVibration" target="_blank">
        <img src="https://github.com/HongfeiWan/DeepVibration/blob/main/imgaes/Logo.svg" width="1000">
    </a>
</p>
<p align="center">
    <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/python-3.8+-blue.svg">
    </a>
    <a href="https://root.cern/">
        <img src="https://img.shields.io/badge/ROOT-latest-orange.svg">
    </a>
    <a href="https://github.com/matlab">
        <img src="https://img.shields.io/badge/Matlab-2024a-brightgreen.svg">
    </a>
</p>

# DeepVibration

## 项目概述

DeepVibration 是一个用于分析高纯锗（HPGe）探测器信号及其振动信号的数据处理和分析工具集。该项目主要用于 CEvNS（相干弹性中微子-原子核散射）实验的数据分析，处理来自 V1725 数据采集卡的原始二进制数据，并对 HPGe 探测器的信号和振动传感器信号进行联合分析。

## 项目结构

```
DeepVibration/
├── C/                          # C++ 数据处理程序
│   └── src/
│       └── makeTreeSimpleDZL_V1725.C   # 主要的数据处理和拟合程序
├── python/                     # Python 工具集
│   ├── data/
│   │   └── preprocessor.py     # 数据预处理：bin转HDF5
│   └── utils/
│       ├── save.py             # HDF5数据保存工具
│       └── visualize.py        # 数据可视化工具
├── matlab/                     # MATLAB 分析脚本
│   ├── Preprocessor/           # 数据预处理脚本
│   ├── HPGe_signal/            # HPGe信号分析
│   ├── Viberation_signal/      # 振动信号分析
│   └── Joint/                  # 联合分析脚本
├── data/                       # 数据目录
│   ├── bin/                    # 原始二进制数据文件
│   └── hdf5/                   # 处理后的HDF5格式数据
│       └── raw_pulse/
│           ├── CH0-3/          # HPGe探测器通道（CH0-CH3）
│           └── CH5/            # 随机触发通道（CH5）
└── design/                     # 设计文档
```

## 功能模块

### 1. 数据预处理 (Python)

#### `python/data/preprocessor.py`

将 V1725 数据采集卡的原始二进制数据（`.bin` 文件）转换为 HDF5 格式的原始脉冲数据。

**主要功能：**
- 读取 V1725 二进制格式的 FADC 原始数据
- 提取指定通道的原始波形数据
- 支持多进程并行处理，充分利用多核 CPU
- 将数据保存为 HDF5 格式（`.h5`），便于后续分析和 MATLAB 读取

**使用方法：**
```python
# 修改配置参数
AMP_CHANNEL_LIST = [0, 1, 2, 3]     # HPGe探测器通道
TRIGGER_CHANNEL_LIST = [5]          # 随机触发通道
RUN_Start_NUMBER = 281              # 起始运行编号
RUN_End_NUMBER = 281                # 结束运行编号

# 运行预处理
python python/data/preprocessor.py
```

**输出：**
- `data/hdf5/raw_pulse/CH0-3/`: HPGe信号通道的HDF5文件
- `data/hdf5/raw_pulse/CH5/`: 随机触发通道的HDF5文件

### 2. 数据可视化 (Python)

#### `python/utils/visualize.py`

提供原始脉冲数据的可视化和数据结构查看功能。

**主要功能：**
- 列出所有 HDF5 文件
- 显示 HDF5 文件的数据结构（形状、数据类型、统计信息等）
- 可视化单个波形（指定事件和通道）
- 可视化多个通道的波形对比（同一事件）

**使用方法：**
```python
from utils.visualize import *

# 列出所有文件
list_all_h5_files()

# 获取文件列表
h5_files = get_h5_files()

# 显示文件结构
show_h5_structure('path/to/file.h5')

# 可视化单个波形
visualize_waveform('path/to/file.h5', 
                   event_idx=0, 
                   channel_idx=0, 
                   time_unit='us')

# 可视化多通道波形对比
visualize_multiple_channels('path/to/file.h5',
                           event_idx=0,
                           channel_indices=[0, 1, 2, 3],
                           time_unit='us')
```

**命令行使用：**
```bash
python python/utils/visualize.py
```

### 3. 数据处理和拟合 (C++)

#### `C/src/makeTreeSimpleDZL_V1725.C`

使用 ROOT 框架进行数据处理、脉冲拟合和特征提取的程序。

**主要功能：**
- 读取原始二进制数据或 HDF5 格式数据
- 计算基线、积分值、峰值等脉冲特征
- 执行快放拟合（fast rise time fitting）
- 输出 ROOT 格式的数据树

**配置参数：**

```cpp
// 基本配置
const unsigned int CHANNEL_NUMBER = 16;    // 通道数量
const unsigned int EVENT_NUMBER = 10000;   // 每个bin文件的事件数
const unsigned int MAX_WINDOWS = 30000;    // 时间窗口大小 (120μs)
const unsigned int PED_RANGE = 1000;       // 基线计算范围

// 快放拟合参数
const unsigned int FIT_RANGE = 5500;       // 拟合范围
const unsigned int fit_start = 6000;       // 拟合开始时间
const unsigned int TRY_NUMBER = 10;        // 拟合尝试次数

// 阈值参数
const unsigned int THRESHOLD_AC = 1450;    // 触发阈值
```

**输出变量：**

- **事件级别**: 事件编号、触发记录、时间戳、死时间等
- **通道级别**: 基线、积分值、峰值、峰值时间点、均方根等
- **拟合变量**: 拟合幅度、基线、斜率、交叉点、χ²值等

**使用方法：**
```bash
./makeTreeSimpleDZL_V1725 [输入文件名] [运行开始编号] [运行结束编号] [杂项参数]
```

**依赖项：**
- ROOT 框架（TFile, TTree, TF1, TH1F, TGraph）
- 自定义头文件：`misc.h`, `tanh_fit.h`

### 4. MATLAB 分析脚本

#### 数据预处理 (`matlab/Preprocessor/`)
- `1.bin2rawpulse.m`: 二进制文件转原始脉冲数据
- `2.bin2Qtime.m`: 提取电荷量-时间关系
- `3.bin2Qmaxtime.m`: 提取电荷量-峰值时间关系
- `4.bin_to_RT.m`: 提取随机触发数据

#### HPGe信号分析 (`matlab/HPGe_signal/`)
- `1.HPGe_signal_Histogram.m`: HPGe信号直方图分析
- `2.HPGe_signal_Scatter.m`: HPGe信号散点图分析

#### 振动信号分析 (`matlab/Viberation_signal/`)
- `0.Import_from_txt.m`: 从文本文件导入振动数据
- `1.All_sensor_correlation.m`: 所有传感器相关性分析
- `2.All_sensor_information.m`: 传感器信息统计
- `3.Vibration_Amplitude_Temperature.m`: 振动幅度-温度关系分析

#### 联合分析 (`matlab/Joint/`)
- `1.HPGe_scatter_Viberation_amplitude.m`: HPGe信号与振动幅度的联合分析

## 数据处理流程

1. **数据采集**: V1725 数据采集卡采集原始二进制数据（`.bin` 文件）
2. **数据预处理**: 使用 Python `preprocessor.py` 将 `.bin` 文件转换为 HDF5 格式
3. **数据可视化**: 使用 Python `visualize.py` 检查数据质量和波形
4. **数据处理**: 使用 C++ `makeTreeSimpleDZL_V1725.C` 进行脉冲拟合和特征提取
5. **信号分析**: 使用 MATLAB 脚本进行 HPGe 信号和振动信号的统计分析
6. **联合分析**: 使用 MATLAB 脚本分析 HPGe 信号与振动信号的相关性

## 通道说明

- **CH0-CH3**: HPGe 探测器信号通道，用于记录核信号
- **CH5**: 随机触发通道，用于记录随机触发事件

## 数据格式

### 输入格式
- **二进制文件**: V1725 数据采集卡原始格式（`.bin`）
  - 文件结构：Run Header + Event Header + Channel Data
  - 采样率：250 MSPS（采样间隔 4 ns）
  - 时间窗口：120 μs（30000 个采样点）

### 输出格式
- **HDF5 格式** (`.h5`)
  - `channel_data`: 形状为 (时间采样点数, 通道数, 事件数)，数据类型为 uint16
  - `time_data`: 形状为 (事件数,)，数据类型为 float64，单位为秒

## 依赖环境

### Python
- Python 3.8+
- numpy
- h5py
- matplotlib

### C++
- ROOT 框架（推荐最新版本）
- C++ 编译器（支持 C++11 或更高版本）

### MATLAB
- MATLAB 2024a 或更高版本
- Statistics and Machine Learning Toolbox

## 安装和配置

1. **克隆仓库**
```bash
git clone https://github.com/HongfeiWan/DeepVibration.git
cd DeepVibration
```

2. **安装 Python 依赖**
```bash
pip install numpy h5py matplotlib
```

3. **配置 ROOT 环境**（如使用 C++ 程序）
```bash
source /path/to/root/bin/thisroot.sh
```

4. **编译 C++ 程序**（如需要）
```bash
cd C/src
g++ -o makeTreeSimpleDZL_V1725 makeTreeSimpleDZL_V1725.C `root-config --cflags --libs`
```

## 使用示例

### 示例1: 预处理数据
```python
# 修改 python/data/preprocessor.py 中的配置
RUN_Start_NUMBER = 281
RUN_End_NUMBER = 281

# 运行预处理
python python/data/preprocessor.py
```

### 示例2: 可视化数据
```python
from utils.visualize import *

# 获取CH5目录的文件
h5_files = get_h5_files()
ch5_file = h5_files['CH5'][0]

# 查看数据结构
show_h5_structure(ch5_file)

# 可视化波形
visualize_waveform(ch5_file, event_idx=0, channel_idx=0, time_unit='us')
```

## 注意事项

1. **文件路径**: 需要根据实际环境修改数据文件的输入输出路径
2. **参数配置**: 拟合参数和阈值参数需要根据具体实验条件优化
3. **内存使用**: 处理大文件时注意内存占用，建议使用多进程处理
4. **数据一致性**: 确保通道配置与实际硬件配置匹配
5. **ROOT环境**: 使用 C++ 程序前需要正确配置 ROOT 环境

## 许可证

详见 [LICENSE](LICENSE) 文件

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题或建议，请通过 GitHub Issues 联系。
