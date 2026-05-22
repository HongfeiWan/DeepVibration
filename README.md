<p align="center">
  <a href="https://github.com/HongfeiWan/DeepVibration" target="_blank">
    <img src="https://github.com/HongfeiWan/DeepVibration/blob/main/images/logo.svg" width="500">
  </a>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.8+-blue.svg"></a>
  <a href="https://root.cern/"><img src="https://img.shields.io/badge/ROOT-latest-orange.svg"></a>
  <a href="https://github.com/matlab"><img src="https://img.shields.io/badge/Matlab-2024a-brightgreen.svg"></a>
</p>

# DeepVibration

DeepVibration 是用于 HPGe 探测器、NaI 反符合通道、随机触发通道与振动/温度环境量联合分析的工具仓库。当前 Python 部分已经整理为标准分析包 `analysis`，把物理 cut、HDF5 I/O、参数读取、绘图和批量流程从临时脚本中拆出来，方便在不同物理示例中复用。

## Python Layout

```text
python/
├── analysis/                  # 可复用分析包
│   ├── parallel.py             # 全 CPU 并行配置：--workers auto
│   ├── io/                     # HDF5、参数文件、run 文件配对
│   ├── features/               # 波形、pedestal、频域特征
│   ├── cuts/                   # RT / Inhibit / Physical / ACT / ACV / mincut / pncut
│   ├── pipelines/              # 多个 cut 的组合流程
│   ├── environment/            # 振动、温度、压缩机环境量
│   ├── ml/                     # PCA / UMAP / HDBSCAN / GMM / LedoitWolf
│   ├── plotting/               # 波形、能谱等绘图工具
│   ├── simulation/             # 模拟与动力学工具
│   └── legacy/                 # 已迁出 data 的历史研究脚本，统一蛇形命名
├── scripts/                    # 正式批处理入口
│   ├── preprocess_raw_pulse.py  # bin -> raw_pulse HDF5 + CH0/1/4/5 参数
│   ├── build_parameters.py      # CH2/CH3 拟合与频域参数
│   ├── run_cuts.py              # 批量 Physical / ACT / ACV cut
│   ├── analyze_signal.py        # FFT / Lomb / Hilbert / Wavelet
│   ├── run_ml.py                # 参数矩阵 ML 分析
│   ├── plot_environment.py      # 环境量总览图
│   └── plot_spectrum.py         # 参数文件能谱绘图
├── examples/                   # 独立物理示例
│   ├── rt_selection.py
│   ├── inhibit_selection.py
│   ├── physical_act_acv.py
│   └── pncut_efficiency.py
```

真实实验数据仍放在仓库根目录的 `data/` 下，并由 `.gitignore` 排除。

## Data Flow

1. `python/scripts/preprocess_raw_pulse.py`
   - 读取 V1725 `.bin`
   - 输出 `data/hdf5/raw_pulse/CH0-3`、`CH4`、`CH5`
   - 生成 `CH0_parameters`、`CH1_parameters`、`CH4_parameters`、`CH5_parameters`

2. `python/scripts/build_parameters.py`
   - 读取 `CH0-3/channel_data`
   - 对 CH2/CH3 做 tanh 拟合、频域特征和补写参数
   - 输出 `CH2_parameters`、`CH3_parameters`

3. `python/scripts/run_cuts.py`
   - 从参数文件批量执行 Physical、ACT 或 ACV 筛选
   - 可选输出每个 run 的 mask HDF5

4. `python/scripts/plot_spectrum.py`
   - 从 `CH0_parameters/max_ch0` 和可选 cut mask 生成能谱

5. `python/scripts/analyze_signal.py` / `python/scripts/run_ml.py`
   - 对单事件波形做 FFT/Lomb/Hilbert/Wavelet
   - 对参数矩阵做 PCA、UMAP、HDBSCAN、GMM 或 LedoitWolf

## Parallel Policy

所有新的批量入口统一使用 `analysis.parallel`：

```bash
python python/scripts/preprocess_raw_pulse.py --workers auto --chunk-size 1000
python python/scripts/build_parameters.py --workers auto --chunk-size 1000
python python/scripts/run_cuts.py --mode act --workers auto --chunk-size 1000
```

`--workers auto` 默认使用 `os.cpu_count()` 返回的全部逻辑 CPU。需要限制资源时可手动指定：

```bash
python python/scripts/build_parameters.py --workers 8
```

为了避免多进程和 NumPy/BLAS 底层线程互相抢核，`analysis.parallel` 会默认把每个 worker 进程内的 BLAS 线程数限制为 1，并在 worker 内阻止再开一层进程池。

## Cut Definitions

常用 cut 已集中到 `analysis.cuts`：

```python
from analysis.cuts import rt_mask, inhibit_mask, physical_mask, acv_mask, act_mask

rt = rt_mask(max_ch5, threshold=6000.0)
inhibit = inhibit_mask(ch0_min)
physical = physical_mask(max_ch5, ch0_min)
acv = acv_mask(max_ch4, tmax_ch4)
act = ~acv
```

组合流程在 `analysis.pipelines`：

```python
from analysis.pipelines import run_act_acv_selection

result = run_act_acv_selection(
    max_ch5=max_ch5,
    ch0_min=ch0_min,
    max_ch0=max_ch0,
    max_ch1=max_ch1,
    max_ch4=max_ch4,
    tmax_ch4=tmax_ch4,
    mode="act",
)
print(result.mask, result.stats)
```

## Examples

```bash
python python/examples/rt_selection.py data/hdf5/raw_pulse/CH5/<run>.h5
python python/examples/inhibit_selection.py data/hdf5/raw_pulse/CH0-3/<run>.h5
python python/examples/physical_act_acv.py \
  --ch0 data/hdf5/raw_pulse/CH0_parameters/<run>.h5 \
  --ch1 data/hdf5/raw_pulse/CH1_parameters/<run>.h5 \
  --ch4 data/hdf5/raw_pulse/CH4_parameters/<run>.h5 \
  --ch5 data/hdf5/raw_pulse/CH5_parameters/<run>.h5 \
  --mode act
```

## Verification

```bash
PYTHONPATH=python python -m unittest discover -s tests
PYTHONPATH=python python -m compileall python/analysis python/scripts python/examples
python python/scripts/preprocess_raw_pulse.py --help
python python/scripts/build_parameters.py --help
python python/scripts/run_cuts.py --help
python python/scripts/analyze_signal.py --help
python python/scripts/run_ml.py --help
python python/scripts/plot_environment.py --help
```

## Non-Python Code

- `C/src/makeTreeSimpleDZL_V1725.C`: ROOT/C++ 数据处理入口。
- `matlab/`: 早期 MATLAB 预处理、HPGe、振动和联合分析脚本。
- `3rdparty/`: 串口通信依赖。
