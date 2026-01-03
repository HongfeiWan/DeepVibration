#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inhibit信号分析
分析CH0-CH3文件中每个事件的CH0信号最小值，判断是否为inhibit信号
Inhibit信号判断标准：CH0信号的min <= 0
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

def analyze_inhibit_signals(h5_file: str = None,
                            channel_idx: int = 0) -> Dict:
    """
    分析文件中CH0信号的inhibit信号
    
    参数:
        h5_file: HDF5 文件路径，如果为None则自动获取CH0-3目录中的第一个文件
        channel_idx: 通道索引，默认0表示CH0通道
    
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
    
    try:
        with h5py.File(h5_file, 'r') as f:
            # 检查数据集是否存在
            if 'channel_data' not in f:
                raise KeyError('文件中没有找到 channel_data 数据集')
            
            channel_data = f['channel_data']
            time_samples, num_channels, num_events = channel_data.shape
            
            print(f'\n数据维度: (时间采样点数={time_samples}, 通道数={num_channels}, 事件数={num_events})')
            
            if channel_idx < 0 or channel_idx >= num_channels:
                raise IndexError(f'通道索引 {channel_idx} 超出范围 [0, {num_channels-1}]')
            
            # 提取所有事件的CH0波形最小值
            print(f'\n正在计算所有事件的CH{channel_idx}信号最小值...')
            min_values = np.zeros(num_events, dtype=np.float64)
            
            # 批量读取以提高效率
            batch_size = 1000  # 每次处理1000个事件
            for i in range(0, num_events, batch_size):
                end_idx = min(i + batch_size, num_events)
                batch_data = channel_data[:, channel_idx, i:end_idx]  # shape: (time_samples, batch_size)
                batch_min = np.min(batch_data, axis=0)  # 沿时间轴取最小值
                min_values[i:end_idx] = batch_min
                
                if (i // batch_size + 1) % 10 == 0 or end_idx == num_events:
                    print(f'  已处理 {end_idx}/{num_events} 个事件 ({end_idx/num_events*100:.1f}%)')
            
            # 判断inhibit信号：min <= 0
            inhibit_mask = min_values <= 0
            inhibit_count = np.sum(inhibit_mask)
            normal_count = num_events - inhibit_count
            inhibit_rate = inhibit_count / num_events * 100
            
            # 打印统计信息
            print(f'\n' + '=' * 70)
            print(f'Inhibit信号分析结果:')
            print(f'=' * 70)
            print(f'总事件数: {num_events}')
            print(f'Inhibit信号数量 (CH{channel_idx} min <= 0): {inhibit_count}')
            print(f'正常信号数量 (CH{channel_idx} min > 0): {normal_count}')
            print(f'Inhibit信号比例: {inhibit_rate:.2f}%')
            print(f'正常信号比例: {100 - inhibit_rate:.2f}%')
            print(f'\nCH{channel_idx}信号最小值统计:')
            print(f'  最小值范围: [{np.min(min_values):.2f}, {np.max(min_values):.2f}]')
            print(f'  平均值: {np.mean(min_values):.2f}')
            print(f'  中位数: {np.median(min_values):.2f}')
            
            # 显示inhibit信号的详细信息
            if inhibit_count > 0:
                inhibit_min_values = min_values[inhibit_mask]
                print(f'\nInhibit信号的最小值统计:')
                print(f'  最小值: {np.min(inhibit_min_values):.2f}')
                print(f'  最大值: {np.max(inhibit_min_values):.2f}')
                print(f'  平均值: {np.mean(inhibit_min_values):.2f}')
                print(f'  中位数: {np.median(inhibit_min_values):.2f}')
                
                # 找出inhibit事件的索引（前10个）
                inhibit_indices = np.where(inhibit_mask)[0]
                print(f'\n前10个Inhibit事件索引: {inhibit_indices[:10].tolist()}')
            else:
                print(f'\n未发现Inhibit信号！')
            
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
                         max_events_to_plot: Optional[int] = 10,
                         save_path: Optional[str] = None,
                         show_plot: bool = True,
                         figsize: Tuple[int, int] = (16, 10)) -> None:
    """
    可视化inhibit信号对应的CH0波形
    
    参数:
        h5_file: HDF5 文件路径，如果为None则自动获取CH0-3目录中的第一个文件
        channel_idx: 通道索引，默认0表示CH0通道
        max_events_to_plot: 最多绘制的inhibit事件数量，None表示绘制所有
        save_path: 保存图片路径，如果为None则不保存
        show_plot: 是否显示图片
        figsize: 图片大小
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
        
        # 确定要绘制的事件数量
        if max_events_to_plot is None:
            num_to_plot = inhibit_count
        else:
            num_to_plot = min(max_events_to_plot, inhibit_count)
        
        selected_indices = inhibit_indices[:num_to_plot]
        print(f'\n将绘制 {num_to_plot} 个inhibit信号的波形')
        
        # 读取文件获取波形数据
        with h5py.File(h5_file, 'r') as f:
            channel_data = f['channel_data']
            time_samples, num_channels, num_events = channel_data.shape
            
            # 参数设置
            sampling_interval_ns = 4.0  # 4ns per sample
            sampling_interval_s = sampling_interval_ns * 1e-9
            time_axis_us = np.arange(time_samples) * sampling_interval_ns / 1000.0  # 转换为微秒
            
            # 创建图形
            n_cols = 3
            n_rows = (num_to_plot + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            
            # 如果只有一个subplot，确保axes是数组
            if num_to_plot == 1:
                axes = np.array([[axes]])
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            
            # 绘制每个inhibit信号的波形
            for plot_idx, event_idx in enumerate(selected_indices):
                row = plot_idx // n_cols
                col = plot_idx % n_cols
                ax = axes[row, col]
                
                # 获取波形数据
                waveform = channel_data[:, channel_idx, event_idx]
                min_val = np.min(waveform)
                max_val = np.max(waveform)
                mean_val = np.mean(waveform)
                
                # 绘制波形
                ax.plot(time_axis_us, waveform, 'b-', linewidth=0.8, alpha=0.8)
                
                # 标注最小值（inhibit信号的关键特征）
                min_idx = np.argmin(waveform)
                ax.plot(time_axis_us[min_idx], min_val, 'ro', markersize=6, label=f'Min: {min_val:.1f}')
                
                # 设置标题和标签
                ax.set_xlabel('Time (μs)', fontsize=9)
                ax.set_ylabel('Amplitude (ADC counts)', fontsize=9)
                ax.set_title(f'Event #{event_idx} (Inhibit Signal)\n'
                           f'Min: {min_val:.1f}, Max: {max_val:.1f}, Mean: {mean_val:.1f}',
                           fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
                
                # 添加0线作为参考
                ax.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
            
            # 隐藏多余的subplot
            for plot_idx in range(num_to_plot, n_rows * n_cols):
                row = plot_idx // n_cols
                col = plot_idx % n_cols
                axes[row, col].axis('off')
            
            plt.suptitle(f'Inhibit Signal Waveforms (CH{channel_idx})\n'
                        f'Total inhibit events: {inhibit_count}, Showing: {num_to_plot}',
                        fontsize=12, y=0.995)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f'\n图片已保存至: {save_path}')
            
            if show_plot:
                plt.show()
            else:
                plt.close()
        
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
            max_events_to_plot=10,  # 最多绘制10个inhibit信号
            show_plot=True
        )

    except Exception as e:
        print(f'分析失败: {e}')
    

