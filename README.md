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
│   ├── build_event_feature_umap.py # 全参数事件矩阵 + LedoitWolf/Mahalanobis/UMAP
│   ├── run_clean_remaining_umap.py # 排除基础异常后的 clean UMAP
│   ├── run_hdbscan_on_umap.py   # 对缓存 UMAP 做 HDBSCAN 密度聚类
│   ├── plot_hdbscan_cluster_waveforms.py # 抽样查看各 cluster 原始波形
│   ├── plot_hdbscan_cluster_feature_distributions.py # 解释 cluster 的参数分布
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

6. `python/scripts/build_event_feature_umap.py`
   - 扫描 `data/hdf5/raw_pulse/*_parameters`
   - 按 run 配对所有 CH 参数文件，拼成事件级 `(n_events, n_features)` 特征矩阵
   - 缓存 LedoitWolf 收缩协方差、精度矩阵、事件马氏距离和 UMAP 诊断图

7. `python/scripts/run_anomaly_umap_example.py`
   - 用全参数 Mahalanobis UMAP 做已知异常类验证图
   - 只给随机触发、过阈值、最小值 3σ 外、前沿基线 3σ 外等 basic-cut 异常着色，其余事件作为灰色背景
   - 默认异常优先抽样，避免稀有异常在随机抽样中消失

8. `python/scripts/run_clean_remaining_umap.py`
   - 在全参数缓存上排除拟合失败、Inhibit、RT、ACT、min/pedestal 3σ 外和过阈值事件
   - 对剩余 clean events 重新做 Mahalanobis UMAP，作为后续无监督聚类的输入

9. `python/scripts/run_hdbscan_on_umap.py`
   - 对缓存 UMAP embedding 做 HDBSCAN 密度聚类
   - 扫描 `min_cluster_size` / `min_samples`，按目标 cluster 数、噪声比例和聚类置信度自动选参数

10. `python/scripts/plot_hdbscan_cluster_waveforms.py` / `python/scripts/plot_hdbscan_cluster_feature_distributions.py`
    - 前者从每个 cluster 抽样原始 `CH0-3/channel_data` 波形
    - 后者从所有 sampled events 的参数分布解释 cluster 为什么能被分开

## Parallel Policy

所有新的批量入口统一使用 `analysis.parallel`：

```bash
python python/scripts/preprocess_raw_pulse.py --workers auto --chunk-size 1000
python python/scripts/build_parameters.py --workers auto --chunk-size 1000
python python/scripts/run_cuts.py --mode act --workers auto --chunk-size 1000
python python/scripts/build_event_feature_umap.py --stage all --workers auto --chunk-size 10000
```

`--workers auto` 默认使用 `os.cpu_count()` 返回的全部逻辑 CPU。需要限制资源时可手动指定：

```bash
python python/scripts/build_parameters.py --workers 8
```

为了避免多进程和 NumPy/BLAS 底层线程互相抢核，`analysis.parallel` 会默认把每个 worker 进程内的 BLAS 线程数限制为 1，并在 worker 内阻止再开一层进程池。

## Event Matrix + Mahalanobis UMAP

大规模全参数事件分析入口：

```bash
uv run --extra ml python python/scripts/build_event_feature_umap.py \
  --stage all \
  --workers auto \
  --chunk-size 10000 \
  --cov-fit-max-events 1000000 \
  --umap-max-events 200000
```

默认缓存目录为 `data/cache/event_feature_umap/event_feature_cache.h5`。缓存内容包括：

- `features`: float32 的 `(n_events, n_features)` 事件特征矩阵。
- `feature_names`, `run_names`, `run_index`, `event_index`, `event_time`: 事件映射和溯源信息。
- `masks/*`: RT、Inhibit、Physical、ACT、ACV、PN cut、time cut 和 clean 事件标签。
- `mahalanobis/*`: LedoitWolf 协方差、精度矩阵、标准化均值/尺度和拟合样本索引。
- `mahalanobis_distance`: 每个事件到 LedoitWolf 全局中心的马氏距离。
- `umap/*`: Mahalanobis metric UMAP 的抽样索引和二维嵌入。

`--cov-fit-max-events` 和 `--umap-max-events` 控制抽样规模；设为 `0` 表示全量拟合。对 512 x 10000 级别的事件数，不应缓存完整 `N x N` pairwise 马氏距离矩阵，因为它会远超 256GB 内存和常规磁盘容量；当前实现用 LedoitWolf 精度矩阵作为 UMAP 的 Mahalanobis metric，让 UMAP 在内部构建近邻图，同时保留可分批读取的事件矩阵、mask 和距离到中心诊断量。

只重建缓存、只拟合协方差、只画图也可以拆开跑：

```bash
python python/scripts/build_event_feature_umap.py --stage cache --workers auto --rebuild-cache
uv run --extra ml python python/scripts/build_event_feature_umap.py --stage fit --cov-fit-max-events 1000000
uv run --extra ml python python/scripts/build_event_feature_umap.py --stage umap --umap-max-events 200000
python python/scripts/build_event_feature_umap.py --stage plot --output-dir data/cache/event_feature_umap/plots
```

### Known-Anomaly UMAP Example

展示 UMAP 是否有物理区分能力时，推荐先用独立 cut 定义已知异常类，再只对这些异常着色：

```bash
uv run --extra ml python python/scripts/run_anomaly_umap_example.py \
  --stage all \
  --workers auto \
  --chunk-size 100000 \
  --umap-max-events 200000 \
  --umap-neighbors 200 \
  --umap-min-dist 0.002 \
  --max-events-per-anomaly 20000 \
  --output-dir data/cache/event_feature_umap/anomaly_umap_nn200_md0002_stratified
```

这个图的解释逻辑是：UMAP 只使用全参数特征矩阵和 LedoitWolf/Mahalanobis 距离，不使用事件类别标签；类别标签只在画图阶段用于着色。如果随机触发、饱和/过阈值、mincut 3σ 外、前沿基线 3σ 外事件在图上落在孤立簇、边界结构或明显不同的流形分支上，就说明这些低维结构和已有物理筛选具有一致性。

### Clean UMAP, DBSCAN-Style Clustering, And Cluster Interpretation

当 known-anomaly UMAP 已经证明基础异常能被区分后，可以进一步研究“排除这些异常后的剩余事件”内部是否还有自然结构。推荐流程是：

1. 先做裸 UMAP，只画 clean remaining events。
2. 再对 UMAP embedding 做 DBSCAN-style 密度聚类；当前脚本使用 HDBSCAN，因为它比固定 `eps` 的 DBSCAN 更适合密度不均匀的 UMAP 图。
3. 抽查每个 cluster 的原始波形。
4. 最后用参数分布解释 cluster 为什么分开。

#### 1. Clean Remaining UMAP

这一版 clean 定义为排除：

- `fit_failed`: CH2/CH3 拟合失败，定义为 `n_fit_points <= 0` 或 `tanh_p0` 为 `1e6` sentinel。
- `inhibit`: `ch0_min == 0`。
- `random_trigger` / RT: `max_ch5 > 6000`。
- `act`: NaI 符合事件，`max_ch4 >= 7060` 且 `40us - tmax_ch4 * 4ns` 落在 `[1, 16] us`。
- `min_3sigma_outlier`: CH0/CH1 最小值落在 3σ 外。
- `pedestal_3sigma_outlier`: CH0/CH1 前沿基线落在 3σ 外。
- `over_threshold`: `max_ch0 > 16382` 或 `max_ch1 > 16382`。

示例命令：

```bash
uv run --extra ml python python/scripts/run_clean_remaining_umap.py \
  --cache-dir data/cache/event_feature_umap \
  --output-dir data/cache/clean_remaining_umap_remote_exclude_fit_failed_nn400_md01 \
  --clean-mask-name clean_remaining_no_fit_failed \
  --exclude-masks fit_failed inhibit random_trigger min_3sigma_outlier pedestal_3sigma_outlier act over_threshold \
  --umap-neighbors 400 \
  --umap-min-dist 0.1 \
  --umap-max-events 200000 \
  --workers auto \
  --chunk-size 100000
```

一次 5.1M 事件样本的结果中，排除 union 后剩余 clean events 为 `2,518,722`，UMAP 抽样 `200,000` 个事件。输出包括：

- `clean_remaining_umap.png`: 没有聚类着色的裸 UMAP。
- `clean_remaining_umap_summary.csv`: 各类排除事件数和 UMAP 参数。

#### 2. HDBSCAN Density Clustering

对裸 UMAP 的二维 embedding 做 HDBSCAN。这里的 HDBSCAN 可以理解为 DBSCAN 的层次密度聚类版本；它不需要预先设定 eps，并能自然给出 noise 点。为了得到适合物理诊断的 cluster 数，可以设置目标 cluster 数范围。

示例命令，目标约 6 个 cluster：

```bash
uv run --extra ml python python/scripts/run_hdbscan_on_umap.py \
  --cache-dir data/cache/event_feature_umap \
  --output-dir data/cache/hdbscan_clean_remaining_umap_exclude_fit_failed_6clusters \
  --min-cluster-sizes 1000,1500,2000,2500,3000,4000,5000,7000,10000 \
  --min-samples 25,50,100,200,400 \
  --target-min-clusters 6 \
  --target-max-clusters 6 \
  --point-size 2.0
```

本次 6-cluster 版本自动选择：

```text
min_cluster_size = 1000
min_samples = 50
clusters = 6
noise_fraction = 0.8345%
```

输出包括：

- `hdbscan_umap_clusters.png`: UMAP 上按 cluster 着色。
- `hdbscan_cluster_summary.csv`: 每个 cluster 的事件数、比例、平均 membership probability。
- `hdbscan_parameter_scan.csv`: 所有扫描参数的 cluster 数、噪声比例、最大簇比例和评分。
- `hdbscan_umap_clusters.h5`: `sample_indices`、`embedding`、`labels`、`probabilities`，可供后续画波形和参数分布。

#### 3. Cluster Waveform Checks

聚类后先抽查原始波形，确认某些 cluster 是否对应肉眼可见的波形族。对于 `CH0-3/channel_data`，可以每个 cluster 抽 9 个事件，绘制 CH0、CH1、CH2、CH3 四列：

```bash
python python/scripts/plot_hdbscan_cluster_waveforms.py \
  --cluster-h5 data/cache/hdbscan_clean_remaining_umap_exclude_fit_failed_6clusters/hdbscan_umap_clusters.h5 \
  --feature-cache data/cache/event_feature_umap/event_feature_cache.h5 \
  --output-dir data/cache/hdbscan_cluster_waveforms_exclude_fit_failed_6clusters_random_ch0_3 \
  --events-per-cluster 9 \
  --selection random-local \
  --channels 0 1 2 3 \
  --random-seed 2026
```

`--selection random-local` 的含义是：每个 cluster 内随机选一个 anchor event，再取同一 run 中离它最近的同簇事件。这样保留随机抽样味道，同时避免 HDF5 对 event 维度做大量离散读取。

输出包括每个 cluster 一张图：

- `cluster_noise_waveforms_ch0_3.png`
- `cluster_00_waveforms_ch0_3.png`
- ...
- `cluster_05_waveforms_ch0_3.png`
- `selected_cluster_waveform_events.csv`: 每张图对应的 run、event index、global index、UMAP 坐标和 membership probability。

#### 4. Parameter Distribution Explanation

如果波形肉眼看不出清楚差别，应看 cluster 的参数分布。这个阶段使用 HDBSCAN 结果中的全部 sampled events，而不是只看抽出来的 63 个波形事件。

```bash
python python/scripts/plot_hdbscan_cluster_feature_distributions.py \
  --cluster-h5 data/cache/hdbscan_clean_remaining_umap_exclude_fit_failed_6clusters/hdbscan_umap_clusters.h5 \
  --feature-cache data/cache/event_feature_umap/event_feature_cache.h5 \
  --output-dir data/cache/hdbscan_cluster_feature_distributions_exclude_fit_failed_6clusters \
  --top-n 30 \
  --hist-top-n 12 \
  --umap-feature-top-n 9 \
  --box-features-per-page 12 \
  --cluster-top-n 12
```

输出包括：

- `feature_separation_scores.csv`: 每个参数区分 cluster 的排序，核心指标是按 cluster 分组解释的方差比例 `eta2_no_noise`。
- `cluster_feature_summary.csv`: 每个 cluster、每个参数的均值、分位数和 robust z-score median。
- `cluster_top_features.csv`: 每个 cluster 最偏离全局分布的参数。
- `cluster_median_heatmap_top_features.png`: cluster 中位数相对全局中位数/IQR 的热图。
- `feature_boxplots_top_page_*.png`: 最能区分 cluster 的参数 boxplot。
- `feature_histograms_top.png`: 顶部区分参数的密度分布。
- `umap_top_feature_gradients.png`: UMAP 上用关键参数连续值着色。
- `top_two_feature_scatter.png`: 最强两个区分参数的二维散点图。

本次 6-cluster 版本中，自动排序最靠前的区分参数是：

```text
ch2_tmin_ch2
ch3_tmin_ch3
ch2_max_ch2
ch3_min_ch3
ch3_tanh_rms
ch1_tmax_ch1
ch1_ch1pedt_mean
ch2_tanh_rms
ch2_spectral_centroid_mhz
ch0_ch0_min
```

因此这批 cluster 的差别不一定都能在 CH0/CH3 波形上直接看出来；很多分离来自 CH2/CH3 的极值时间、幅度、拟合 RMS、频域 centroid，以及 CH1 的时间/基线参数。物理解释时建议按这个顺序检查：先看 `feature_separation_scores.csv` 找全局最能分离的变量，再看 `cluster_top_features.csv` 找每个 cluster 的特征签名，最后回到原始波形确认这些参数差异是否对应真实波形形态。

## Cut Definitions

常用 cut 已集中到 `analysis.cuts`：

```python
from analysis.cuts import (
    acv_mask,
    act_mask,
    inhibit_mask,
    pedestal_3sigma_mask,
    rt_mask,
    saturation_mask,
)

rt = rt_mask(max_ch5, threshold=6000.0)
inhibit = inhibit_mask(ch0_min)
pedestal_ok = pedestal_3sigma_mask(ch0ped_mean, ch1ped_mean, reference_mask=rt)
not_saturated = saturation_mask(max_ch0, max_ch1, max_adc=16382.0)
acv = acv_mask(max_ch4, tmax_ch4)
act = act_mask(max_ch4, tmax_ch4)
physical = (~rt) & (~inhibit)
basic = physical & pedestal_ok & not_saturated
```

基础筛选按物理流程理解为：先识别 `inhibit` 事例（`ch0_min == 0`）和随机触发事例（`max_ch5 > 6000`），用随机触发样本给 CH0/CH1 前沿基线 pedestal 建立 3σ 带并保留带内事件，再剔除 CH0/CH1 过阈值事件（`max_ch0/max_ch1 > 16382`）。NaI 符合定义为 `max_ch4 >= 7060` 且 `delta_t = 40us - tmax_ch4*4ns` 落在 `[1, 16] us` 内，记为 ACT；未触发 NaI 或落在窗口外的事件记为 ACV。

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
python python/scripts/build_event_feature_umap.py --help
python python/scripts/run_anomaly_umap_example.py --help
python python/scripts/run_clean_remaining_umap.py --help
python python/scripts/run_hdbscan_on_umap.py --help
python python/scripts/plot_hdbscan_cluster_waveforms.py --help
python python/scripts/plot_hdbscan_cluster_feature_distributions.py --help
python python/scripts/plot_environment.py --help
```

## Non-Python Code

- `C/src/makeTreeSimpleDZL_V1725.C`: ROOT/C++ 数据处理入口。
- `matlab/`: 早期 MATLAB 预处理、HPGe、振动和联合分析脚本。
- `3rdparty/`: 串口通信依赖。
