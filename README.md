
<p align="center">
    <a href="https://github.com/HongfeiWan/DeepVibration" target="_blank">
        <img src="https://github.com/HongfeiWan/DeepVibration/blob/main/images/logo.svg" width="500">
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
│   ├── data/                   # 数据分析模块
│   │   ├── preprocessor.py     # 数据预处理：bin转HDF5
│   │   ├── coincident/         # 符合事件分析
│   │   │   ├── process/        # 信号处理（FFT、小波、Hilbert、Lomb-Scargle）
│   │   │   │   ├── fft.py      # FFT频谱分析
│   │   │   │   ├── hilbert.py  # Hilbert变换分析
│   │   │   │   ├── lomb.py     # Lomb-Scargle周期图分析
│   │   │   │   └── wavelet.py  # 小波变换分析
│   │   │   └── randomtrigger&inhibit.py  # RT和Inhibit符合事件分析
│   │   ├── compressor/         # 制冷机数据分析
│   │   │   └── select.py       # 制冷机数据读取和筛选
│   │   ├── ge-self/            # HPGe探测器自身信号分析（物理事件）
│   │   │   ├── select.py       # 物理事件筛选（既非RT也非Inhibit）
│   │   │   ├── wavelet.py      # 物理事件小波分析（100kHz-25MHz）
│   │   │   └── cut/            # 物理事件筛选条件
│   │   │       ├── gmmpncut.py      # GMM pn cut（高斯混合模型筛选）
│   │   │       ├── gmmwavelet.py    # GMM小波分析（对两类GMM事件分别做小波变换）
│   │   │       ├── lsmpncut.py      # LSM pn cut（最小二乘法筛选）
│   │   │       ├── mincut.py        # 最小值筛选
│   │   │       ├── overthreshold.py # 过阈值筛选
│   │   │       ├── pedcut.py        # 基线（pedestal）筛选
│   │   │       ├── pncut.py         # pn cut（正负脉冲筛选）
│   │   │       └── physical/        # 物理事件筛选
│   │   │           └── select.py   # 物理事件筛选
│   │   ├── inhibit/            # Inhibit信号分析
│   │   │   └── select.py       # Inhibit信号筛选
│   │   ├── randomtrigger/      # 随机触发分析
│   │   │   ├── distribution.py # RT信号分布分析
│   │   │   └── select.py       # RT信号筛选
│   │   ├── sensor/             # 传感器数据分析
│   │   │   └── vibration/      # 振动传感器数据
│   │   │       ├── preprocess.py              # 振动数据预处理（TXT转HDF5）
│   │   │       ├── accelerate/                # 加速度分析
│   │   │       │   └── select.py              # 加速度数据读取和筛选
│   │   │       ├── displacement/              # 位移分析
│   │   │       │   └── select.py              # 位移数据读取和筛选
│   │   │       ├── frequency/                  # 频率分析
│   │   │       │   ├── select.py               # 频率数据读取和筛选
│   │   │       │   ├── correlationship.py     # 频率相关性分析
│   │   │       │   └── distribution.py         # 频率分布分析
│   │   │       ├── temperature/               # 振动传感器温度
│   │   │       │   ├── select.py               # 温度数据读取和筛选
│   │   │       │   └── animation.py            # 温度动画
│   │   │       ├── temperature&displacement/   # 温度和位移联合分析
│   │   │       │   └── temperature&sensor2.py  # 温度和传感器2位移联合分析
│   │   │       └── velocity/                   # 速度分析
│   │   │           └── select.py                # 速度数据读取和筛选
│   │   ├── temperature/        # 温度数据联合分析
│   │   │   ├── unite.py        # 振动传感器和制冷机温度联合绘制
│   │   │   ├── power/          # 温度和功率联合分析
│   │   │   │   └── unite.py    # 振动传感器温度、制冷机温度和功率联合绘制
│   │   │   └── countrate/      # 计数率和温度联合分析
│   │   │       ├── unite.py     # bin文件事件计数率和制冷机温度联合绘制
│   │   │       ├── inhibit/     # Inhibit计数率和温度
│   │   │       │   └── unite.py
│   │   │       ├── physics/     # 物理事件计数率和温度
│   │   │       │   └── unite.py
│   │   │       └── randomtrigger/  # RT计数率和温度
│   │   │           └── unite.py
│   │   └── wavelet/            # 小波分析工具
│   │       └── wavelet.py      # 通用小波分析（对所有event的CH0信号拼接后分析）
│   └── utils/                  # 工具模块
│       ├── save.py             # HDF5数据保存工具
│       ├── visualize.py        # 数据可视化工具
│       └── time.py             # 时间工具（bin文件时间读取）
├── matlab/                     # MATLAB 分析脚本
│   ├── Preprocessor/           # 数据预处理脚本
│   ├── HPGe_signal/            # HPGe信号分析
│   ├── Viberation_signal/      # 振动信号分析
│   └── Joint/                  # 联合分析脚本
├── data/                       # 数据目录
│   ├── bin/                    # 原始二进制数据文件（.bin）
│   ├── hdf5/                   # 处理后的HDF5格式数据
│   │   └── raw_pulse/
│   │       ├── CH0-3/          # HPGe探测器通道（CH0-CH3）
│   │       └── CH5/            # 随机触发通道（CH5）
│   ├── compressor/             # 制冷机数据
│   │   └── txt/                # 制冷机文本数据（.txt）
│   └── vibration/              # 振动传感器数据
│       ├── txt/                # 振动传感器原始文本数据
│       └── hdf5/               # 振动传感器处理后的HDF5数据
└── docs/                       # 文档目录
    └── (相关PDF文档)
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

### 2.1 信号分析模块 (Python)

#### `python/data/ge-self/`

**HPGe探测器自身信号分析**（物理事件，既非RT也非Inhibit）

- **物理事件筛选** (`select.py`): 筛选既非RT也非Inhibit的物理事件
- **小波分析** (`wavelet.py`): 对物理事件的CH0信号进行小波时频分析（频率范围：100kHz-25MHz）

**物理事件筛选条件** (`cut/`):
- **GMM pn cut** (`gmmpncut.py`): 使用高斯混合模型（GMM）对CH1和CH2的最大值分布进行2成分拟合，自动识别两类事件区域，并展示示例波形
- **GMM小波分析** (`gmmwavelet.py`): 对GMM识别出的两类事件分别进行小波变换，统计对比两类事件的功率谱差异（使用增量统计模式避免OOM）
- **LSM pn cut** (`lsmpncut.py`): 使用最小二乘法拟合CH1和CH2最大值的关系，筛选在直线±1σ带内的事件
- **最小值筛选** (`mincut.py`): 基于CH0最小值的筛选
- **过阈值筛选** (`overthreshold.py`): 过滤过阈值事件（max(CH0) > 16382）
- **基线筛选** (`pedcut.py`): 基于基线（pedestal）的筛选
- **pn cut** (`pncut.py`): 正负脉冲筛选
- **物理事件筛选** (`physical/select.py`): 物理事件筛选

#### `python/data/inhibit/`

**Inhibit信号分析** (`select.py`): 分析Inhibit信号（CH0最小值 == 0）

#### `python/data/randomtrigger/`

**RT信号分析**:
- `distribution.py`: 分析随机触发信号的分布
- `select.py`: RT信号筛选

#### `python/data/coincident/`

**符合事件分析** (`randomtrigger&inhibit.py`): 分析RT和Inhibit的符合事件

**信号处理** (`process/`):
- `fft.py`: FFT频谱分析
- `hilbert.py`: Hilbert变换分析
- `lomb.py`: Lomb-Scargle周期图分析
- `wavelet.py`: 小波变换分析

### 2.2 传感器数据分析 (Python)

#### `python/data/sensor/vibration/`

**振动数据预处理** (`preprocess.py`): 将振动传感器TXT数据转换为HDF5格式

**振动信号分析**:
- **加速度分析** (`accelerate/select.py`): 加速度数据读取和筛选
- **位移分析** (`displacement/select.py`): 位移数据读取和筛选
- **频率分析** (`frequency/`):
  - `select.py`: 频率数据读取和筛选
  - `correlationship.py`: 频率相关性分析
  - `distribution.py`: 频率分布分析
- **速度分析** (`velocity/select.py`): 速度数据读取和筛选

**温度数据** (`temperature/`):
- `select.py`: 温度数据读取和筛选
- `animation.py`: 温度动画

**联合分析** (`temperature&displacement/`):
- `temperature&sensor2.py`: 温度和传感器2位移联合分析

### 2.3 温度数据联合分析 (Python)

#### `python/data/temperature/`

**温度联合绘制** (`unite.py`): 在同一幅图中绘制振动传感器温度和制冷机温度

**温度和功率联合绘制** (`power/unite.py`): 绘制振动传感器温度、制冷机温度（多列）和功率

**计数率和温度联合分析** (`countrate/`):
- `unite.py`: 绘制bin文件事件计数率和制冷机Controller温度
- `inhibit/unite.py`: Inhibit计数率和温度联合分析
- `physics/unite.py`: 物理事件计数率和温度联合分析
- `randomtrigger/unite.py`: RT计数率和温度联合分析

### 2.4 工具模块 (Python)

#### `python/utils/time.py`

**时间工具**: 从bin文件读取时间信息并计算时间跨度

**主要功能：**
- 读取bin文件的起始和结束时间
- 支持从HDF5文件读取（如果已处理）
- 批量处理多个bin文件

**使用方法：**
```python
from utils.time import get_bin_file_time_span, process_bin_files_time_span

# 获取单个bin文件的时间跨度
start_time, end_time = get_bin_file_time_span('path/to/file.bin')

# 批量处理多个bin文件
date_list = process_bin_files_time_span(
    bin_dir='data/bin',
    filename_input='20250520_CEvNS_DZL_sm_...',
    run_start=0,
    run_end=999
)
```

### 3. 制冷机数据分析 (Python)

#### `python/data/compressor/select.py`

读取和分析制冷机数据（温度、功率等）。

**主要功能：**
- 从TXT文件读取制冷机数据
- 按日期范围筛选数据
- 支持多个温度列（Compressor temp、Controller temp、Coldhead temp等）

### 4. 数据处理和拟合 (C++)

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

### 5. MATLAB 分析脚本

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

1. **数据采集**: 
   - V1725 数据采集卡采集原始二进制数据（`.bin` 文件）
   - 振动传感器采集数据（`.txt` 文件）
   - 制冷机采集温度、功率等数据（`.txt` 文件）

2. **数据预处理**: 
   - 使用 Python `preprocessor.py` 将 `.bin` 文件转换为 HDF5 格式
   - 使用 Python `sensor/vibration/preprocess.py` 将振动传感器TXT数据转换为HDF5格式

3. **数据可视化**: 
   - 使用 Python `visualize.py` 检查数据质量和波形
   - 使用 Python `utils/time.py` 查看bin文件的时间跨度

4. **信号分析**: 
   - 使用 Python `ge-self/select.py` 筛选物理事件
   - 使用 Python `ge-self/wavelet.py` 对物理事件进行小波分析（100kHz-25MHz）
   - 使用 Python `ge-self/cut/` 模块进行各种筛选（GMM、LSM、pn cut等）
   - 使用 Python `ge-self/cut/gmmwavelet.py` 对GMM两类事件分别进行小波统计分析
   - 使用 Python `inhibit/select.py` 分析Inhibit信号
   - 使用 Python `randomtrigger/` 模块分析RT信号
   - 使用 Python `coincident/` 模块分析符合事件

5. **联合分析**: 
   - 使用 Python `temperature/unite.py` 分析温度数据
   - 使用 Python `temperature/countrate/unite.py` 分析计数率和温度关系
   - 使用 Python `coincident/process/` 进行信号处理（FFT、小波、Hilbert等）

6. **数据处理**: 使用 C++ `makeTreeSimpleDZL_V1725.C` 进行脉冲拟合和特征提取

7. **高级分析**: 使用 MATLAB 脚本进行更深入的统计分析和可视化

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
- pandas
- scipy (用于信号处理)
- pywt (用于小波变换)
- scikit-learn (用于GMM拟合，gmmpncut和gmmwavelet模块需要)
- joblib (用于并行计算)

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
pip install numpy h5py matplotlib pandas scipy pywt scikit-learn joblib
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

### 示例3: GMM筛选和小波分析
```python
# 运行GMM pn cut，识别两类事件区域
python python/data/ge-self/cut/gmmpncut.py

# 对GMM两类事件分别进行小波统计分析（100kHz-25MHz）
python python/data/ge-self/cut/gmmwavelet.py
```

### 示例4: 物理事件小波分析
```python
# 对物理事件进行小波时频分析（100kHz-25MHz）
python python/data/ge-self/wavelet.py
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
