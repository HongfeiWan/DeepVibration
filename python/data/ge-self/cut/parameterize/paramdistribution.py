#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对符合 lsmpncut PN-cut ±1σ 带内的 Physical 事件，计算并展示所有参数的分布图。

参数包括：
- Amax, Amin: 波形幅度的最大值和最小值
- Tmax, Tmin: 达到最大值和最小值的时间
- Q: 峰值两侧首次回到 Pedt 高度之间的积分
- Qpre: Tmax 之前，波形和 Pedt 围合区域的积分
- Qprev: Tmax 之后，波形和 Pedt 围合区域的积分
- ped, pedt: 前沿和尾部基线
- abs(pedt-ped): 基线差值
"""

import os
import sys
from typing import Optional, Tuple, List, Dict
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
import matplotlib.pyplot as plt
import importlib.util
from tqdm import tqdm


# -----------------------------------------------------------------------------
# 导入必要的函数
# -----------------------------------------------------------------------------

current_dir = os.path.dirname(os.path.abspath(__file__))          # .../cut/parameterize
cut_dir = os.path.dirname(current_dir)                            # .../cut

# 导入 overthreshold
overthreshold_path = os.path.join(cut_dir, "overthreshold.py")
spec_over = importlib.util.spec_from_file_location("overthreshold_module", overthreshold_path)
overthreshold_module = importlib.util.module_from_spec(spec_over)
assert spec_over.loader is not None
spec_over.loader.exec_module(overthreshold_module)
select_physical_events_no_overthreshold = overthreshold_module.select_physical_events_no_overthreshold

# 导入 visualization
visualization_path = os.path.join(current_dir, "visualization.py")
spec_vis = importlib.util.spec_from_file_location("pulse_visualization_module", visualization_path)
pulse_vis_module = importlib.util.module_from_spec(spec_vis)
assert spec_vis.loader is not None
spec_vis.loader.exec_module(pulse_vis_module)
compute_pulse_parameters = pulse_vis_module.compute_pulse_parameters

# 导入 lsmpncut
lsmpncut_path = os.path.join(cut_dir, "lsmpncut.py")
spec_ls = importlib.util.spec_from_file_location("lsmpncut_module", lsmpncut_path)
lsmpncut_module = importlib.util.module_from_spec(spec_ls)
assert spec_ls.loader is not None
spec_ls.loader.exec_module(lsmpncut_module)
fit_single_line_in_range = lsmpncut_module.fit_single_line_in_range

# 导入中值滤波
python_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))  # .../python
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)
from utils.filter import median_filter


def _select_events_in_1sigma_band(
    ch0_3_file: Optional[str] = None,
    ch5_file: Optional[str] = None,
    rt_cut: float = 6000.0,
    ch0_threshold: float = 16382.0,
    ch0_idx: int = 0,
    ch1_idx: int = 1,
    x_min: float = 2000.0,
    x_max: float = 14000.0,
    sigma_factor: float = 1.0,
) -> Tuple[np.ndarray, str, str, np.ndarray]:
    """
    在不过阈值 Physical 事件中，使用 lsmpncut 的 PN-cut 逻辑选出落在 ±1σ 线性带内的事件。
    
    返回：
        event_ranks, ch0_3_file_sel, ch5_file_sel, selected_indices
    """
    print("=" * 70)
    print("PN-cut 参数分布：使用 lsmpncut 逻辑选择 ±1σ 带内的事件")
    print("=" * 70)

    sel = select_physical_events_no_overthreshold(
        ch0_3_file=ch0_3_file,
        ch5_file=ch5_file,
        rt_cut=rt_cut,
        ch0_threshold=ch0_threshold,
        ch0_idx=ch0_idx,
        ch5_idx=0,
    )

    ch0_3_file_sel: str = sel["ch0_3_file"]
    ch5_file_sel: str = sel["ch5_file"]
    selected_indices: np.ndarray = sel["selected_indices"]
    final_physical_count: int = int(sel["final_physical_count"])

    if final_physical_count == 0 or selected_indices.size == 0:
        raise RuntimeError("未发现不过阈值的 Physical 信号，无法进行 PN-cut 选择。")

    print(f"不过阈值 Physical 事件数: {final_physical_count}")

    # 读取这些事件的波形，并计算 CH0/CH1 的最大值（用于 PN-cut）
    with h5py.File(ch0_3_file_sel, "r") as f_ch0:
        channel_data = f_ch0["channel_data"]
        time_samples, num_channels, num_events = channel_data.shape

        if ch0_idx >= num_channels or ch1_idx >= num_channels:
            raise ValueError(
                f"通道索引超出范围：ch0_idx={ch0_idx}, ch1_idx={ch1_idx}, num_channels={num_channels}"
            )

        phys_ch0 = channel_data[:, ch0_idx, selected_indices].astype(np.float64)
        phys_ch1 = channel_data[:, ch1_idx, selected_indices].astype(np.float64)

    max_ch0 = phys_ch0.max(axis=0)
    max_ch1 = phys_ch1.max(axis=0)

    # 使用 lsmpncut 进行拟合
    print(f"\n在 {x_min} < max_ch0 < {x_max} 范围内使用 lsmpncut 进行两步最小二乘拟合...")
    a, b = fit_single_line_in_range(max_ch0, max_ch1, x_min=x_min, x_max=x_max)

    # 计算残差的标准差 σ
    mask_range = (max_ch0 > x_min) & (max_ch0 < x_max)
    x_fit = max_ch0[mask_range]
    y_fit = max_ch1[mask_range]
    y_fit_pred = a * x_fit + b
    residuals = y_fit - y_fit_pred
    sigma = residuals.std(ddof=1) if residuals.size > 1 else 0.0

    print(f"拟合直线: max_ch1 ≈ {a:.6f} * max_ch0 + {b:.3f}")
    print(f"拟合范围内点数: {x_fit.size}，残差标准差 σ = {sigma:.3f}")

    if sigma <= 0.0:
        raise RuntimeError("σ ≤ 0，无法定义 ±1σ 带。")

    # 选出落在 ±sigma_factor*σ 带内的事件
    predicted_all = a * max_ch0 + b
    all_residuals = max_ch1 - predicted_all
    sigma_mask = np.abs(all_residuals) <= sigma_factor * sigma

    event_ranks = np.where(sigma_mask)[0]

    print(f"\n落在 ±{sigma_factor}σ 带内的 Physical 事件数: {event_ranks.size}")

    if event_ranks.size == 0:
        raise RuntimeError("没有事件落在 ±σ 带内。")

    return event_ranks, ch0_3_file_sel, ch5_file_sel, selected_indices


def compute_qpre_qprev(
    waveform: np.ndarray,
    pedt: float,
    tmax_us: float,
    sampling_interval_ns: float = 4.0,
) -> Tuple[float, float]:
    """
    计算 Qpre 和 Qprev。
    
    Qpre: Tmax 之前，波形和 Pedt 围合的所有面积（包括上方和下方）
    Qprev: Tmax 之后，波形和 Pedt 围合的所有面积（包括上方和下方）
    
    参数：
        waveform: 波形数组
        pedt: 尾部基线值
        tmax_us: Tmax 时间（微秒）
        sampling_interval_ns: 采样间隔（纳秒）
    
    返回：
        (qpre, qprev)
    """
    wf = np.asarray(waveform, dtype=np.float64).ravel()
    n_samples = wf.shape[0]
    dt_s = sampling_interval_ns * 1e-9
    
    time_axis_us = np.arange(n_samples) * sampling_interval_ns / 1000.0
    idx_tmax = int(np.argmin(np.abs(time_axis_us - tmax_us)))
    
    # Qpre: Tmax 之前，波形和 pedt 围合的所有面积（|waveform - pedt| 的积分）
    before_tmax_mask = np.arange(n_samples) < idx_tmax
    qpre = float(np.sum(np.abs(wf[before_tmax_mask] - pedt) * dt_s))
    
    # Qprev: Tmax 之后，波形和 pedt 围合的所有面积（|waveform - pedt| 的积分）
    after_tmax_mask = np.arange(n_samples) > idx_tmax
    qprev = float(np.sum(np.abs(wf[after_tmax_mask] - pedt) * dt_s))
    
    return qpre, qprev


def _process_single_event_worker(args: Tuple) -> Tuple[int, Dict[str, float]]:
    """
    工作函数：处理单个事件的参数计算。
    
    参数：
        args: (rank, waveform, sampling_interval_ns, baseline_window_us)
    
    返回：
        (rank, 参数字典)
    """
    rank, waveform, sampling_interval_ns, baseline_window_us = args
    
    # 中值滤波
    waveform = median_filter(waveform, kernel_size=3)
    
    # 计算基本参数
    params = compute_pulse_parameters(
        waveform,
        sampling_interval_ns=sampling_interval_ns,
        baseline_window_us=baseline_window_us
    )
    
    # 计算 Tmin
    idx_min = int(np.argmin(waveform))
    tmin_us = idx_min * sampling_interval_ns / 1000.0
    
    # 计算 Qpre 和 Qprev
    qpre, qprev = compute_qpre_qprev(
        waveform,
        params["pedt"],
        params["tmax_us"],
        sampling_interval_ns=sampling_interval_ns
    )
    
    # 计算基线差值
    ped_diff = abs(params["pedt"] - params["ped"])
    
    result = {
        "amax": params["amax"],
        "amin": params["ammin"],
        "tmax": params["tmax_us"],
        "tmin": tmin_us,
        "q": params["q"],
        "qpre": qpre,
        "qprev": qprev,
        "ped": params["ped"],
        "pedt": params["pedt"],
        "ped_diff": ped_diff,
    }
    
    return rank, result


def compute_all_parameters_for_events(
    ch0_3_file: str,
    event_ranks: np.ndarray,
    selected_indices: np.ndarray,
    ch0_idx: int = 0,
    baseline_window_us: float = 2.0,
    max_workers: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    对选中的事件计算所有参数（使用多进程并行计算）。
    
    参数：
        ch0_3_file: CH0-3 HDF5 文件路径
        event_ranks: 事件在 selected_indices 中的索引
        selected_indices: 全局事件索引
        ch0_idx: CH0 通道索引
        baseline_window_us: 基线窗口长度（微秒）
        max_workers: 最大工作进程数，None 表示使用所有CPU核心
    
    返回：
        包含所有参数的字典
    """
    print("\n正在计算所有事件的参数...")
    
    sampling_interval_ns = 4.0
    n_events = event_ranks.size
    
    # 初始化参数数组
    amax_arr = np.zeros(n_events, dtype=np.float64)
    amin_arr = np.zeros(n_events, dtype=np.float64)
    tmax_arr = np.zeros(n_events, dtype=np.float64)
    tmin_arr = np.zeros(n_events, dtype=np.float64)
    q_arr = np.zeros(n_events, dtype=np.float64)
    qpre_arr = np.zeros(n_events, dtype=np.float64)
    qprev_arr = np.zeros(n_events, dtype=np.float64)
    ped_arr = np.zeros(n_events, dtype=np.float64)
    pedt_arr = np.zeros(n_events, dtype=np.float64)
    ped_diff_arr = np.zeros(n_events, dtype=np.float64)
    
    # 读取所有波形数据到内存（避免多进程间共享HDF5文件）
    print("正在读取波形数据到内存...")
    with h5py.File(ch0_3_file, "r") as f_ch0:
        ch0_channel_data = f_ch0["channel_data"]
        time_samples, num_channels, num_events_total = ch0_channel_data.shape
        
        # 获取所有需要读取的全局索引
        global_indices = selected_indices[event_ranks]
        
        # 使用高级索引一次性批量读取所有波形（比逐个读取快得多）
        # shape: (time_samples, n_events)
        all_waveforms = ch0_channel_data[:, ch0_idx, global_indices].astype(np.float64)
        
        # 转换为列表，每个元素是一个波形（转置以便按事件访问）
        # all_waveforms.T shape: (n_events, time_samples)
        waveforms_list = [all_waveforms[:, i] for i in range(n_events)]
    
    print(f"已读取 {len(waveforms_list)} 个波形，准备并行处理...")
    
    # 准备任务参数
    task_args = [
        (rank, waveforms_list[i], sampling_interval_ns, baseline_window_us)
        for i, rank in enumerate(event_ranks)
    ]
    
    # 确定工作进程数
    if max_workers is None:
        max_workers = os.cpu_count() or 1
    print(f"使用 {max_workers} 个CPU核心进行并行计算")
    
    # 使用多进程并行处理
    completed_count = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_rank = {
            executor.submit(_process_single_event_worker, args): args[0]
            for args in task_args
        }
        
        # 使用 tqdm 显示进度条
        with tqdm(total=n_events, desc="计算参数", unit="事件", ncols=100) as pbar:
            # 收集结果
            for future in as_completed(future_to_rank):
                try:
                    rank, result = future.result()
                    
                    # 存储结果
                    idx = np.where(event_ranks == rank)[0][0]
                    amax_arr[idx] = result["amax"]
                    amin_arr[idx] = result["amin"]
                    tmax_arr[idx] = result["tmax"]
                    tmin_arr[idx] = result["tmin"]
                    q_arr[idx] = result["q"]
                    qpre_arr[idx] = result["qpre"]
                    qprev_arr[idx] = result["qprev"]
                    ped_arr[idx] = result["ped"]
                    pedt_arr[idx] = result["pedt"]
                    ped_diff_arr[idx] = result["ped_diff"]
                    
                    completed_count += 1
                    pbar.update(1)
                except Exception as e:
                    print(f"\n警告: 处理事件 rank={future_to_rank[future]} 时出错: {e}")
                    pbar.update(1)
    
    print(f"\n完成: 成功处理 {completed_count}/{n_events} 个事件")
    
    return {
        "Amax": amax_arr,
        "Amin": amin_arr,
        "Tmax": tmax_arr,
        "Tmin": tmin_arr,
        "Q": q_arr,
        "Qpre": qpre_arr,
        "Qprev": qprev_arr,
        "ped": ped_arr,
        "pedt": pedt_arr,
        "abs(pedt-ped)": ped_diff_arr,
    }


def plot_parameter_distributions(
    parameters: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> str:
    """
    绘制所有参数的分布图。
    
    参数：
        parameters: 参数字典
        save_path: 保存路径，None 时自动生成
        show_plot: 是否显示图片
    
    返回：
        保存的图片路径
    """
    print("\n正在绘制参数分布图...")
    
    # 设置字体
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
    })
    
    # 创建子图：2行5列
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("Parameter Distributions for Events within ±1σ of lsmpncut Line", 
                  fontsize=16, fontweight="bold")
    
    param_names = [
        "Amax", "Amin", "Tmax", "Tmin", "Q",
        "Qpre", "Qprev", "ped", "pedt", "abs(pedt-ped)"
    ]
    
    for idx, param_name in enumerate(param_names):
        row = idx // 5
        col = idx % 5
        ax = axes[row, col]
        
        data = parameters[param_name]
        
        # 绘制直方图
        ax.hist(data, bins=50, color="tab:blue", alpha=0.7, edgecolor="black", linewidth=0.5)
        
        # 添加统计信息
        mean_val = np.mean(data)
        std_val = np.std(data)
        median_val = np.median(data)
        
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.2f}")
        ax.axvline(median_val, color="green", linestyle="--", linewidth=1.5, label=f"Median: {median_val:.2f}")
        
        ax.set_xlabel(param_name, fontsize=12, fontweight="bold")
        ax.set_ylabel("Counts", fontsize=12, fontweight="bold")
        ax.set_title(f"{param_name} Distribution\n(σ={std_val:.2f})", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 坐标轴数字加粗
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontweight("bold")
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontweight("bold")
    
    plt.tight_layout()
    
    # 保存图片
    if save_path is None:
        ge_self_dir = os.path.dirname(cut_dir)
        data_dir = os.path.dirname(ge_self_dir)
        python_dir = os.path.dirname(data_dir)
        project_root = os.path.dirname(python_dir)
        
        output_dir = os.path.join(project_root, "images", "presentation")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_png = f"parameter_distributions_{timestamp}.png"
        save_path = os.path.join(output_dir, filename_png)
    
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\n参数分布图已保存至: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return save_path


def plot_parameter_distributions_for_lsmpncut_1sigma(
    ch0_3_file: Optional[str] = None,
    ch5_file: Optional[str] = None,
    rt_cut: float = 6000.0,
    ch0_threshold: float = 16382.0,
    ch0_idx: int = 0,
    ch1_idx: int = 1,
    x_min: float = 2000.0,
    x_max: float = 14000.0,
    sigma_factor: float = 1.0,
    baseline_window_us: float = 2.0,
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> str:
    """
    主函数：选择符合 ±1σ 带内的事件，计算所有参数，并绘制分布图。
    
    返回：
        保存的图片路径
    """
    # 1. 选择事件
    event_ranks, ch0_3_file_used, ch5_file_used, selected_indices = _select_events_in_1sigma_band(
        ch0_3_file=ch0_3_file,
        ch5_file=ch5_file,
        rt_cut=rt_cut,
        ch0_threshold=ch0_threshold,
        ch0_idx=ch0_idx,
        ch1_idx=ch1_idx,
        x_min=x_min,
        x_max=x_max,
        sigma_factor=sigma_factor,
    )
    
    # 2. 计算所有参数（使用所有CPU核心）
    parameters = compute_all_parameters_for_events(
        ch0_3_file=ch0_3_file_used,
        event_ranks=event_ranks,
        selected_indices=selected_indices,
        ch0_idx=ch0_idx,
        baseline_window_us=baseline_window_us,
        max_workers=None,  # None 表示使用所有CPU核心
    )
    
    # 3. 绘制分布图
    save_path = plot_parameter_distributions(
        parameters=parameters,
        save_path=save_path,
        show_plot=show_plot,
    )
    
    return save_path


if __name__ == "__main__":
    try:
        plot_parameter_distributions_for_lsmpncut_1sigma(
            ch0_3_file=None,
            ch5_file=None,
            rt_cut=6000.0,
            ch0_threshold=16382.0,
            ch0_idx=0,
            ch1_idx=1,
            x_min=2000.0,
            x_max=14000.0,
            sigma_factor=1.0,
            baseline_window_us=2.0,
            save_path=None,
            show_plot=True,
        )
    except Exception as e:
        print(f"\n参数分布图生成失败: {e}")
        import traceback
        traceback.print_exc()
