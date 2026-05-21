#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
随机触发通道（CH5）最大值分布分析
分析CH5目录中所有事件的波形最大值分布
"""
import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

# 添加路径以便导入 utils 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
python_dir = os.path.dirname(os.path.dirname(current_dir))
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from utils.visualize import get_h5_files

# 与 python/data/inhibit/select.py 中波形图一致：幅面 + 字号（英寸）
_FIGSIZE_INCH = (7.0, 4.5)
_FS_AXIS_LABEL = 20
_FS_TICK = 16
_FS_LEGEND = 16
_FS_TITLE = 18


def _apply_plotstyle_rc():
    """与 inhibit/select.py 相同：Arial sans-serif"""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
    })


def analyze_ch5_max_distribution(h5_file: str = None,
                                 channel_idx: int = 0,
                                 bins: int = 100,
                                 cut: Optional[float] = None,
                                 save_path: Optional[str] = None,
                                 show_plot: bool = True,
                                 figsize: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, dict]:
    """
    分析CH5文件中所有事件波形最大值的分布
    
    参数:
        h5_file: HDF5 文件路径，如果为None则自动获取CH5目录中的第一个文件
        channel_idx: 通道索引（CH5目录中只有通道0）
        bins: 直方图的bins数量
        cut: 截断阈值，cut线右侧（>cut）的事件为RT触发事件，如果为None则不显示cut线
        save_path: 保存图片的路径，如果为None则不保存
        show_plot: 是否显示图片
        figsize: 图片大小（宽、高，英寸）；默认与 inhibit/select.py 一致
    
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
    print('=' * 70)

    try:
        with h5py.File(h5_file, 'r') as f:
            # 检查数据集是否存在
            if 'channel_data' not in f:
                raise KeyError('文件中没有找到 channel_data 数据集')

            channel_data = f['channel_data']
            time_samples, num_channels, num_events = channel_data.shape

            if channel_idx < 0 or channel_idx >= num_channels:
                raise IndexError(f'通道索引 {channel_idx} 超出范围 [0, {num_channels-1}]')

            max_values = np.zeros(num_events, dtype=np.float64)

            batch_size = 1000
            for i in range(0, num_events, batch_size):
                end_idx = min(i + batch_size, num_events)
                batch_data = channel_data[:, channel_idx, i:end_idx]
                batch_max = np.max(batch_data, axis=0)
                max_values[i:end_idx] = batch_max

            mean_val = np.mean(max_values)
            median_val = np.median(max_values)
            std_val = np.std(max_values)

            rt_event_count = 0
            if cut is not None:
                rt_event_count = int(np.sum(max_values > cut))

            _apply_plotstyle_rc()
            if figsize is None:
                figsize = _FIGSIZE_INCH

            fig, ax = plt.subplots(figsize=figsize)

            n, bins_edges, _ = ax.hist(
                max_values,
                bins=bins,
                color='C0',
                alpha=0.8,
                edgecolor='black',
            )

            if cut is not None:
                ax.axvline(
                    cut,
                    color='red',
                    linestyle='--',
                    linewidth=2,
                    label=f'Cut = {cut:.0f}',
                )

            ax.set_xlabel('Maximum CH5 (ADC counts)', fontsize=_FS_AXIS_LABEL)
            ax.set_ylabel('Count', fontsize=_FS_AXIS_LABEL)
            # ax.set_title(
            #     f'CH5 max amplitude distribution (N={num_events})',
            #     fontsize=_FS_TITLE,
            # )
            ax.tick_params(axis='both', which='major', labelsize=_FS_TICK)
            if cut is not None:
                ax.legend(loc='upper right', fontsize=_FS_LEGEND)
            fig.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f'\n图片已保存至: {save_path}')

            if show_plot:
                plt.show()
            else:
                plt.close(fig)
            
            # 返回统计结果
            hist_stats = {
                'counts': n,
                'bins': bins_edges,
                'mean': mean_val,
                'median': median_val,
                'std': std_val,
                'min': np.min(max_values),
                'max': np.max(max_values),
                'cut': cut,
                'rt_event_count': rt_event_count if cut is not None else None,
                'rt_rate': rt_event_count/num_events*100 if cut is not None else None
            }
            
            return max_values, hist_stats
    
    except Exception as e:
        print(f'分析过程中出错: {e}')
        raise

# 示例使用
if __name__ == '__main__':
    print('=' * 70)
    print('分析单个文件')
    print('=' * 70)
    try:
        # 分析文件并显示 cut 线
        max_values, stats = analyze_ch5_max_distribution(
            h5_file=None,  # 自动选择CH5目录中的第一个文件
            bins=10,
            cut=6000,  # 设置截断阈值，cut线右侧（>cut）的事件为RT触发事件
            show_plot=True
        )
        print(f'\n分析完成！RT触发事件数: {stats["rt_event_count"]}')
    except Exception as e:
        print(f'分析失败: {e}')


