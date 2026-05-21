#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inhibit信号分析
从 raw_pulse/CH{k}_parameters 中读取每事件最小值（与 preprocessor 写入的 ch{k}_min 一致），
判断是否为 inhibit 信号。
Inhibit信号判断标准：对应通道 min == 0（最小值严格等于0）

注：preprocessor 仅对 CH0、CH1 写入 ch0_min / ch1_min；CH2/CH3 无对应参数文件。
"""
import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple

# 添加路径以便导入 utils 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
python_dir = os.path.dirname(os.path.dirname(current_dir))
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from utils.visualize import get_h5_files

# 仓库根目录（python/data/inhibit -> .../DeepVibration）
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

# inhibit 波形图：较小画布 + 较大字号（英寸）
_INHIBIT_FIGSIZE_INCH = (7.0, 4.5)
_INHIBIT_FS_AXIS_LABEL = 20
_INHIBIT_FS_TICK = 16
_INHIBIT_FS_LEGEND = 16


def _parameters_h5_path(pulse_h5_path: str, channel_idx: int) -> str:
    """与 preprocessor 一致：CH{k}_parameters/<与 CH0-3 文件同名>.h5"""
    if channel_idx not in (0, 1):
        raise ValueError(
            'preprocessor 仅在 CH0_parameters、CH1_parameters 中写入 ch0_min/ch1_min；'
            '请将 channel_idx 设为 0 或 1。'
        )
    subdir = f'CH{channel_idx}_parameters'
    return os.path.join(
        _REPO_ROOT, 'data', 'hdf5', 'raw_pulse', subdir, os.path.basename(pulse_h5_path)
    )


def _min_dataset_name(channel_idx: int) -> str:
    return f'ch{channel_idx}_min'


def _apply_plotstyle_rc():
    """与 python/utils/plotstyle.md 一致的全局字体与默认样式"""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
    })


def analyze_inhibit_signals(h5_file: str = None,
                            channel_idx: int = 0) -> Dict:
    """
    分析文件中 CH0（或 CH1）信号的 inhibit 信号；最小值从 CH{k}_parameters 读取。
    
    参数:
        h5_file: HDF5 文件路径，如果为None则自动获取CH0-3目录中的第一个文件
        channel_idx: 通道索引，0 或 1（分别对应 ch0_min / ch1_min）
    
    返回:
        包含统计信息的字典
    """
    # 如果没有指定文件，自动获取CH0-3目录中的第一个文件
    if h5_file is None:
        h5_files = get_h5_files()
        if 'CH0-3' not in h5_files or not h5_files['CH0-3']:
            raise FileNotFoundError('在 data/hdf5/raw_pulse/CH0-3 目录中未找到 h5 文件')
        h5_file = h5_files['CH0-3'][0]
        print(f'自动选择文件: {os.path.basename(h5_file)}')
    
    if not os.path.exists(h5_file):
        raise FileNotFoundError(f'文件不存在: {h5_file}')
    
    print('=' * 70)
    print(f'分析文件: {os.path.basename(h5_file)}')
    print(f'文件路径: {h5_file}')
    print('=' * 70)
    
    params_path = _parameters_h5_path(h5_file, channel_idx)
    ds_name = _min_dataset_name(channel_idx)
    if not os.path.exists(params_path):
        raise FileNotFoundError(
            f'未找到参数文件（需先经 preprocessor 生成）: {params_path}'
        )

    try:
        with h5py.File(params_path, 'r') as f_params:
            if ds_name not in f_params:
                raise KeyError(f'参数文件中缺少数据集 {ds_name}（预期为 preprocessor 写入的每事件最小值）')
            min_values = np.asarray(f_params[ds_name][:], dtype=np.float64)

        num_events = min_values.size
        print(f'\n已从 {os.path.basename(params_path)} 读取 {ds_name}，事件数={num_events}')

        # 与原始脉冲文件核对事件数与通道索引
        with h5py.File(h5_file, 'r') as f:
            if 'channel_data' not in f:
                raise KeyError('脉冲文件中没有找到 channel_data 数据集')
            channel_data = f['channel_data']
            time_samples, num_channels, n_ev_pulse = channel_data.shape
            print(
                f'脉冲文件维度: (时间采样点数={time_samples}, 通道数={num_channels}, 事件数={n_ev_pulse})'
            )
            if channel_idx < 0 or channel_idx >= num_channels:
                raise IndexError(f'通道索引 {channel_idx} 超出范围 [0, {num_channels-1}]')
            if n_ev_pulse != num_events:
                raise ValueError(
                    f'事件数不一致: {ds_name} 长度={num_events}，channel_data 第三维={n_ev_pulse}'
                )

        # 判断inhibit信号：min == 0（最小值严格等于0）
        inhibit_mask = min_values == 0
        inhibit_count = np.sum(inhibit_mask)
        normal_count = num_events - inhibit_count
        inhibit_rate = inhibit_count / num_events * 100

        # 统计接近0但不等于0的情况（用于参考）
        near_zero_mask = (min_values > -1e-6) & (min_values < 0)  # 接近0但不等于0
        near_zero_count = np.sum(near_zero_mask)

        # 打印统计信息
        print(f'\n' + '=' * 70)
        print(f'Inhibit信号分析结果:')
        print(f'=' * 70)
        print(f'总事件数: {num_events}')
        print(f'Inhibit信号数量 (CH{channel_idx} min == 0): {inhibit_count}')
        print(f'正常信号数量 (CH{channel_idx} min != 0): {normal_count}')
        print(f'Inhibit信号比例: {inhibit_rate:.2f}%')
        print(f'正常信号比例: {100 - inhibit_rate:.2f}%')
        if near_zero_count > 0:
            print(f'\n参考信息: 有 {near_zero_count} 个事件的最小值接近0但不等于0 (范围: (-1e-6, 0))')

        if inhibit_count > 0:
            inhibit_indices = np.where(inhibit_mask)[0]
            print(f'\n前10个Inhibit事件索引: {inhibit_indices[:10].tolist()}')
        else:
            print(f'\n未发现Inhibit信号（没有事件的最小值严格等于0）！')
            # 显示最小值最接近0的事件
            abs_min_values = np.abs(min_values)
            closest_to_zero_indices = np.argsort(abs_min_values)[:10]
            print(f'\n最小值最接近0的前10个事件索引: {closest_to_zero_indices.tolist()}')
            print(f'对应的最小值: {min_values[closest_to_zero_indices].tolist()}')

        print('=' * 70)

        # 返回统计结果
        stats = {
            'total_events': num_events,
            'inhibit_count': int(inhibit_count),
            'normal_count': int(normal_count),
            'inhibit_rate': float(inhibit_rate),
            'normal_rate': float(100 - inhibit_rate),
            'min_values': min_values,
            'inhibit_mask': inhibit_mask,
            'inhibit_indices': np.where(inhibit_mask)[0].tolist() if inhibit_count > 0 else []
        }

        return stats
    
    except Exception as e:
        print(f'分析过程中出错: {e}')
        raise

def plot_inhibit_signals(h5_file: str = None,
                         channel_idx: int = 0,
                         save_path: Optional[str] = None,
                         show_plot: bool = True,
                         figsize: Optional[Tuple[float, float]] = None) -> None:
    """
    可视化 inhibit 信号对应通道的波形（最小值标注与 CH{k}_parameters 中一致）。
    每个 inhibit 事件单独一张图（仅一个子图），按事件顺序依次绘制；保存时多张图自动加 _p001、_p002…
    
    参数:
        h5_file: HDF5 文件路径，如果为None则自动获取CH0-3目录中的第一个文件
        channel_idx: 通道索引，0 或 1（与 analyze_inhibit_signals 一致）
        save_path: 保存图片路径，如果为None则不保存；多张图时自动加后缀 _p001、_p002…
        show_plot: 是否显示图片（多图时逐张弹出）
        figsize: 单张图尺寸（宽、高，英寸）；默认较小幅面，见 _INHIBIT_FIGSIZE_INCH
    """
    # 如果没有指定文件，自动获取CH0-3目录中的第一个文件
    if h5_file is None:
        h5_files = get_h5_files()
        if 'CH0-3' not in h5_files or not h5_files['CH0-3']:
            raise FileNotFoundError('在 data/hdf5/raw_pulse/CH0-3 目录中未找到 h5 文件')
        h5_file = h5_files['CH0-3'][0]
    
    if not os.path.exists(h5_file):
        raise FileNotFoundError(f'文件不存在: {h5_file}')
    
    print('=' * 70)
    print(f'可视化Inhibit信号波形')
    print(f'文件: {os.path.basename(h5_file)}')
    print('=' * 70)
    
    try:
        # 先分析inhibit信号
        stats = analyze_inhibit_signals(h5_file, channel_idx)
        
        inhibit_count = stats['inhibit_count']
        inhibit_indices = stats['inhibit_indices']
        
        if inhibit_count == 0:
            print('未发现Inhibit信号，无法绘制')
            return

        n_figs = inhibit_count

        _apply_plotstyle_rc()
        if figsize is None:
            figsize = _INHIBIT_FIGSIZE_INCH

        with h5py.File(h5_file, 'r') as f:
            channel_data = f['channel_data']
            time_samples, num_channels, num_events = channel_data.shape

            sampling_interval_ns = 4.0  # 4ns per sample
            time_axis_us = np.arange(time_samples) * sampling_interval_ns / 1000.0  # μs

            for fig_idx, event_idx in enumerate(inhibit_indices):
                fig, ax = plt.subplots(1, 1, figsize=figsize)

                waveform = channel_data[:, channel_idx, event_idx]

                ax.plot(time_axis_us, waveform, 'b-', linewidth=2.0, alpha=0.8)


                ax.set_xlabel('Time (μs)', fontsize=_INHIBIT_FS_AXIS_LABEL)
                ax.set_ylabel('Amplitude (ADC counts)', fontsize=_INHIBIT_FS_AXIS_LABEL)
                ax.tick_params(axis='both', which='major', labelsize=_INHIBIT_FS_TICK)

                ax.set_xlim(0, 120)

                fig.tight_layout()

                if save_path:
                    if n_figs > 1:
                        root, ext = os.path.splitext(save_path)
                        out_path = f'{root}_p{fig_idx + 1:03d}{ext}'
                    else:
                        out_path = save_path
                    fig.savefig(out_path, dpi=150, bbox_inches='tight')
                    print(f'\n图片已保存至: {out_path}')

                if show_plot:
                    plt.show()
                else:
                    plt.close(fig)

        print(f'\n绘制完成！')
    
    except Exception as e:
        print(f'可视化过程中出错: {e}')
        raise

# 示例使用
if __name__ == '__main__':
    # 方法1: 分析单个文件
    print('=' * 70)
    print('方法1: 分析单个文件')
    print('=' * 70)
    
    try:
        stats = analyze_inhibit_signals(
            h5_file=None,  # 自动选择CH0-3目录中的第一个文件
            channel_idx=0  # CH0通道
        )
        
        # 可视化inhibit信号波形
        print(f'\n' + '=' * 70)
        print('可视化Inhibit信号波形')
        print('=' * 70)
        
        plot_inhibit_signals(
            h5_file=None,  # 自动选择CH0-3目录中的第一个文件
            channel_idx=0,  # CH0通道
            show_plot=True
        )

    except Exception as e:
        print(f'分析失败: {e}')
    

