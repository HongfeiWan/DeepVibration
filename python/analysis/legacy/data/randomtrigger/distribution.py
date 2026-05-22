#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
随机触发通道（CH5）最大值分布分析
分析CH5目录中所有事件的波形最大值分布。
支持批量汇总同一目录下所有 CH5 hdf5 文件，并优先从 CH5_parameters 直接读取 max_ch5，避免读取原始波形。
"""
import os
import sys
import matplotlib

# 以脚本方式运行时必须在 import pyplot 之前选择后端；Agg 等后端下 plt.show() 不会弹出窗口
if __name__ == '__main__' and not os.environ.get('MPLBACKEND'):
    for _bk in ('TkAgg', 'Qt5Agg', 'QtAgg'):
        try:
            matplotlib.use(_bk, force=True)
            break
        except Exception:
            continue

import matplotlib.pyplot as plt
import h5py
import numpy as np
from typing import Optional, Tuple, List, Dict

# 添加路径以便导入 utils 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
python_dir = os.path.dirname(os.path.dirname(current_dir))
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from analysis.io.inspect import get_h5_files

# 与 python/data/ge-self/cut/parameterize/tradition/tradition.py、plotstyle.md 一致
_PLOT_TICK = 12
_PLOT_AXIS = 16
_PLOT_TITLE = 18
_PLOT_LEGEND = 12
_PLOT_XLIM = (0.0, 8000.0)
# 默认将落在该 ADC 区间内的直方 bin 计数在图上置零（仅影响绘图与返回的 hist counts，不改动原始 max 数组）
_DEFAULT_MASK_ADC_RANGE = (2000.0, 6000.0)


def _apply_plotstyle_font() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
        }
    )


def _mask_histogram_counts_in_adc_range(
    counts: np.ndarray,
    bin_edges: np.ndarray,
    adc_lo: float,
    adc_hi: float,
) -> np.ndarray:
    """
    将 bin 与 [adc_lo, adc_hi] 在数轴上有交集的计数置为 0（bin 为 [edge[i], edge[i+1]) 半开区间）。
    """
    out = np.asarray(counts, dtype=np.float64).copy()
    for i in range(out.shape[0]):
        bl = float(bin_edges[i])
        br = float(bin_edges[i + 1])
        if br > adc_lo and bl < adc_hi:
            out[i] = 0.0
    return out


def _plot_ch5_hist_log(
    ax: plt.Axes,
    values: np.ndarray,
    bins: int,
    mask_mid_adc_bins: bool,
    mask_adc_range: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """绘制对数 y 直方图；可选将 mask_adc_range 内 bin 计数置零。返回 (counts, bin_edges)。"""
    n, bin_edges = np.histogram(values, bins=bins)
    n = n.astype(np.float64, copy=False)
    if mask_mid_adc_bins:
        n = _mask_histogram_counts_in_adc_range(
            n, bin_edges, mask_adc_range[0], mask_adc_range[1]
        )
    ax.bar(
        bin_edges[:-1],
        n,
        width=np.diff(bin_edges),
        align="edge",
        edgecolor="black",
        alpha=0.7,
    )
    ax.set_yscale("log")
    return n, bin_edges


def _try_load_max_ch5_from_parameters(ch5_h5_file: str) -> Optional[np.ndarray]:
    """
    对于 raw_pulse/CH5 下的原始 h5 文件，尝试从对应的 raw_pulse/CH5_parameters 读取 max_ch5。

    返回:
        - 成功：max_ch5 (shape: [n_events])
        - 失败：None
    """
    param_file = ch5_h5_file.replace(
        os.path.join('raw_pulse', 'CH5'),
        os.path.join('raw_pulse', 'CH5_parameters'),
    )
    if not os.path.exists(param_file):
        return None

    try:
        with h5py.File(param_file, 'r') as f:
            if 'max_ch5' not in f:
                return None
            return np.asarray(f['max_ch5'][...], dtype=np.float64)
    except Exception:
        return None


def _compute_max_ch5_from_raw_waveform(
    ch5_h5_file: str,
    channel_idx: int = 0,
    chunk_size: int = 1024,
) -> np.ndarray:
    """
    从 raw_pulse/CH5 的原始波形 h5 中，流式计算每个事件的最大值，避免一次性读入全部三维数组。
    """
    with h5py.File(ch5_h5_file, 'r') as f:
        if 'channel_data' not in f:
            raise KeyError('文件中没有找到 channel_data 数据集')
        dset = f['channel_data']
        time_samples, num_channels, num_events = dset.shape
        if channel_idx < 0 or channel_idx >= num_channels:
            raise IndexError(f'通道索引 {channel_idx} 超出范围 [0, {num_channels-1}]')

        max_ch5 = np.full(num_events, -np.inf, dtype=np.float64)
        for start in range(0, time_samples, chunk_size):
            end = min(start + chunk_size, time_samples)
            chunk = dset[start:end, channel_idx, :]  # (chunk_len, n_events)
            chunk_max = np.max(chunk, axis=0).astype(np.float64)
            max_ch5 = np.maximum(max_ch5, chunk_max)
        return max_ch5

def analyze_ch5_max_distribution(h5_file: str = None,
                                 channel_idx: int = 0,
                                 bins: int = 100,
                                 cut: Optional[float] = None,
                                 save_path: Optional[str] = None,
                                 show_plot: bool = True,
                                 figsize: Tuple[int, int] = (10, 6),
                                 prefer_parameters: bool = True,
                                 mask_mid_adc_bins: bool = True,
                                 mask_adc_range: Tuple[float, float] = _DEFAULT_MASK_ADC_RANGE,
                                 ) -> Tuple[np.ndarray, dict]:
    """
    分析CH5文件中所有事件波形最大值的分布
    
    参数:
        h5_file: HDF5 文件路径，如果为None则自动获取CH5目录中的第一个文件
        channel_idx: 通道索引（CH5目录中只有通道0）
        bins: 直方图的bins数量
        cut: 截断阈值，cut线右侧（>cut）的事件为RT触发事件，如果为None则不显示cut线
        save_path: 保存图片的路径，如果为None则不保存
        show_plot: 是否显示图片
        figsize: 图片大小 (宽度, 高度)
        mask_mid_adc_bins: 为 True 时，将 mask_adc_range 与 bin 有交集的计数在图中置为 0（默认开启）
        mask_adc_range: 与 mask_mid_adc_bins 配合，默认 (2000, 6000)

    返回:
        (max_values数组, histogram统计结果) 的元组
    """
    # 如果没有指定文件，自动获取CH5目录中的第一个文件
    if h5_file is None:
        h5_files = get_h5_files()
        if 'CH5' not in h5_files or not h5_files['CH5']:
            raise FileNotFoundError('在 data/hdf5/raw_pulse/CH5 目录中未找到 h5 文件')
        h5_file = h5_files['CH5'][0]
        print(f'自动选择文件: {os.path.basename(h5_file)}')
    if not os.path.exists(h5_file):
        raise FileNotFoundError(f'文件不存在: {h5_file}')
    
    print('=' * 70)
    print(f'分析文件: {os.path.basename(h5_file)}')
    print(f'文件路径: {h5_file}')
    print('=' * 70)
    
    try:
        max_values: Optional[np.ndarray] = None
        if prefer_parameters:
            max_values = _try_load_max_ch5_from_parameters(h5_file)
            if max_values is not None:
                print('\n已从 CH5_parameters 读取 max_ch5（未读取原始波形）。')

        if max_values is None:
            print('\n未找到可用的 CH5_parameters，回退为从原始波形计算 max_ch5（流式）。')
            max_values = _compute_max_ch5_from_raw_waveform(h5_file, channel_idx=channel_idx, chunk_size=1024)

        num_events = int(max_values.shape[0])

        # 统计信息
        print(f'\n统计信息:')
        print(f'  总事件数: {num_events}')
        print(f'  最大值范围: [{np.min(max_values):.2f}, {np.max(max_values):.2f}]')
        print(f'  平均值: {np.mean(max_values):.2f}')
        print(f'  中位数: {np.median(max_values):.2f}')
        print(f'  标准差: {np.std(max_values):.2f}')

        # 如果指定了 cut 值，显示 RT 事件统计
        if cut is not None:
            rt_event_count_tmp = int(np.sum(max_values > cut))
            rt_rate = rt_event_count_tmp / num_events * 100
            print(f'\n截断分析 (Cut = {cut:.2f}):')
            print(f'  RT触发事件数 (> cut): {rt_event_count_tmp}')
            print(f'  RT触发率: {rt_rate:.2f}%')

        # 绘制分布图
        _apply_plotstyle_font()
        plt.rcParams.setdefault("axes.unicode_minus", False)
        fig, ax = plt.subplots(figsize=figsize)

        # 绘制直方图（y 轴对数；x 轴显示 0–8000）；可选将中段 ADC 区间内 bin 计数置零
        n, bins_edges = _plot_ch5_hist_log(
            ax,
            max_values,
            bins=bins,
            mask_mid_adc_bins=mask_mid_adc_bins,
            mask_adc_range=mask_adc_range,
        )

        # 添加统计信息
        mean_val = float(np.mean(max_values))
        median_val = float(np.median(max_values))
        std_val = float(np.std(max_values))

        # 计算 cut 线右侧的事件数量（RT触发事件）
        rt_event_count = 0
        if cut is not None:
            rt_event_count = int(np.sum(max_values > cut))

        # 添加 cut 截断线（如果指定了 cut）
        if cut is not None:
            ax.axvline(
                cut,
                color="red",
                linestyle="-",
                linewidth=2,
                label="Cut",
            )

        ax.set_xlabel(r"max(CH5) (ADC counts)", fontsize=_PLOT_AXIS)
        ax.set_ylabel("Count", fontsize=_PLOT_AXIS)
        ax.set_title(
            f"CH5 Maximum Amplitude Distribution\nFile: {os.path.basename(h5_file)}",
            fontsize=_PLOT_TITLE,
        )
        ax.set_xlim(_PLOT_XLIM)
        if cut is not None:
            ax.legend(loc="best", fontsize=_PLOT_LEGEND)
        ax.tick_params(axis="both", which="major", labelsize=_PLOT_TICK)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'\n图片已保存至: {save_path}')

        if show_plot:
            plt.show(block=True)
        else:
            plt.close()

        # 返回统计结果
        hist_stats = {
            'counts': n,
            'bins': bins_edges,
            'mean': mean_val,
            'median': median_val,
            'std': std_val,
            'min': float(np.min(max_values)),
            'max': float(np.max(max_values)),
            'cut': cut,
            'rt_event_count': rt_event_count if cut is not None else None,
            'rt_rate': (rt_event_count/num_events*100) if cut is not None else None
        }

        return max_values, hist_stats
    
    except Exception as e:
        print(f'分析过程中出错: {e}')
        raise


def analyze_ch5_max_distribution_all_files(
    base_dir: Optional[str] = None,
    bins: int = 100,
    cut: Optional[float] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    prefer_parameters: bool = True,
    mask_mid_adc_bins: bool = True,
    mask_adc_range: Tuple[float, float] = _DEFAULT_MASK_ADC_RANGE,
) -> Tuple[np.ndarray, Dict[str, dict]]:
    """
    汇总 raw_pulse/CH5 目录下所有 h5 文件的 max_ch5，并绘制总体分布。

    参数:
        base_dir: raw_pulse 目录路径；None 则自动定位到 <project_root>/data/hdf5/raw_pulse
        bins/cut/save_path/show_plot/figsize: 同 analyze_ch5_max_distribution
        prefer_parameters: 优先从 CH5_parameters 读取 max_ch5；失败则回退到原始波形流式计算
        mask_mid_adc_bins / mask_adc_range: 同 analyze_ch5_max_distribution

    返回:
        (all_max_values, per_file_stats)
    """
    h5_files = get_h5_files(base_dir)
    if 'CH5' not in h5_files or not h5_files['CH5']:
        raise FileNotFoundError('在 data/hdf5/raw_pulse/CH5 目录中未找到 h5 文件')

    ch5_list: List[str] = h5_files['CH5']
    print('=' * 70)
    print(f'批量分析 CH5 文件数: {len(ch5_list)}')
    print('=' * 70)

    all_values: List[np.ndarray] = []
    per_file_stats: Dict[str, dict] = {}

    for idx, fp in enumerate(ch5_list, 1):
        print(f'\n[{idx}/{len(ch5_list)}] {os.path.basename(fp)}')
        vals = None
        if prefer_parameters:
            vals = _try_load_max_ch5_from_parameters(fp)
            if vals is not None:
                print('  使用 CH5_parameters/max_ch5')
        if vals is None:
            print('  回退：从原始波形流式计算 max_ch5')
            vals = _compute_max_ch5_from_raw_waveform(fp, channel_idx=0, chunk_size=1024)

        vals = np.asarray(vals, dtype=np.float64)
        all_values.append(vals)
        per_file_stats[os.path.basename(fp)] = {
            'n_events': int(vals.shape[0]),
            'min': float(np.min(vals)) if vals.size else float('nan'),
            'max': float(np.max(vals)) if vals.size else float('nan'),
            'mean': float(np.mean(vals)) if vals.size else float('nan'),
            'median': float(np.median(vals)) if vals.size else float('nan'),
            'std': float(np.std(vals)) if vals.size else float('nan'),
        }

    all_max_values = np.concatenate(all_values, axis=0) if all_values else np.array([], dtype=np.float64)

    print('\n' + '=' * 70)
    print('总体统计信息:')
    print(f'  总文件数: {len(ch5_list)}')
    print(f'  总事件数: {all_max_values.shape[0]}')
    if all_max_values.size > 0:
        print(f'  最大值范围: [{np.min(all_max_values):.2f}, {np.max(all_max_values):.2f}]')
        print(f'  平均值: {np.mean(all_max_values):.2f}')
        print(f'  中位数: {np.median(all_max_values):.2f}')
        print(f'  标准差: {np.std(all_max_values):.2f}')

    # 绘图
    _apply_plotstyle_font()
    plt.rcParams.setdefault("axes.unicode_minus", False)
    fig, ax = plt.subplots(figsize=figsize)
    n, bins_edges = _plot_ch5_hist_log(
        ax,
        all_max_values,
        bins=bins,
        mask_mid_adc_bins=mask_mid_adc_bins,
        mask_adc_range=mask_adc_range,
    )

    if cut is not None and all_max_values.size > 0:
        ax.axvline(
            cut,
            color="red",
            linestyle="-",
            linewidth=2,
            label="Cut",
        )
        ax.legend(loc="best", fontsize=_PLOT_LEGEND)

    ax.set_xlabel(r"max(CH5) (ADC counts)", fontsize=_PLOT_AXIS)
    ax.set_ylabel("Count", fontsize=_PLOT_AXIS)
    #ax.set_title("CH5 Maximum Amplitude Distribution (All Files)", fontsize=_PLOT_TITLE)
    ax.set_xlim(_PLOT_XLIM)
    ax.tick_params(axis="both", which="major", labelsize=_PLOT_TICK)
    #ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'\n图片已保存至: {save_path}')

    if show_plot:
        plt.show(block=True)
    else:
        plt.close()

    return all_max_values, per_file_stats

# 示例使用
if __name__ == '__main__':
    print(f'matplotlib 后端: {matplotlib.get_backend()}')
    if str(matplotlib.get_backend()).lower() == 'agg':
        print(
            '提示: 当前为 Agg 后端，无法弹出窗口。可在运行前执行: export MPLBACKEND=TkAgg\n'
            '      或使用带图形界面的本地终端，并确保 DISPLAY 已设置（SSH 需开启 X11 转发）。'
        )
    print('=' * 70)
    print('分析所有文件（优先参数文件）')
    print('=' * 70)
    try:
        # 批量分析并显示 cut 线
        all_values, per_file_stats = analyze_ch5_max_distribution_all_files(
            base_dir=None,  # 自动定位 raw_pulse
            bins=1000,
            cut=6000,  # 设置截断阈值，cut线右侧（>cut）的事件为RT触发事件
            show_plot=True,
            prefer_parameters=True,
        )
        print(f'\n分析完成！总事件数: {all_values.shape[0]}，文件数: {len(per_file_stats)}')
    except Exception as e:
        print(f'分析失败: {e}')


