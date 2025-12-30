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

def analyze_ch5_max_distribution(h5_file: str = None,
                                 channel_idx: int = 0,
                                 bins: int = 100,
                                 cut: Optional[float] = None,
                                 save_path: Optional[str] = None,
                                 show_plot: bool = True,
                                 figsize: Tuple[int, int] = (10, 6)) -> Tuple[np.ndarray, dict]:
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
        with h5py.File(h5_file, 'r') as f:
            # 检查数据集是否存在
            if 'channel_data' not in f:
                raise KeyError('文件中没有找到 channel_data 数据集')
            
            channel_data = f['channel_data']
            time_samples, num_channels, num_events = channel_data.shape
            
            print(f'\n数据维度: (时间采样点数={time_samples}, 通道数={num_channels}, 事件数={num_events})')
            
            if channel_idx < 0 or channel_idx >= num_channels:
                raise IndexError(f'通道索引 {channel_idx} 超出范围 [0, {num_channels-1}]')
            
            # 提取所有事件的波形最大值
            print(f'\n正在计算所有事件的最大值...')
            max_values = np.zeros(num_events, dtype=np.float64)
            
            # 批量读取以提高效率（避免一次性加载所有数据到内存）
            batch_size = 1000  # 每次处理1000个事件
            for i in range(0, num_events, batch_size):
                end_idx = min(i + batch_size, num_events)
                batch_data = channel_data[:, channel_idx, i:end_idx]  # shape: (time_samples, batch_size)
                batch_max = np.max(batch_data, axis=0)  # 沿时间轴取最大值
                max_values[i:end_idx] = batch_max
                if (i // batch_size + 1) % 10 == 0 or end_idx == num_events:
                    print(f'  已处理 {end_idx}/{num_events} 个事件 ({end_idx/num_events*100:.1f}%)')
            # 统计信息
            print(f'\n统计信息:')
            print(f'  总事件数: {num_events}')
            print(f'  最大值范围: [{np.min(max_values):.2f}, {np.max(max_values):.2f}]')
            print(f'  平均值: {np.mean(max_values):.2f}')
            print(f'  中位数: {np.median(max_values):.2f}')
            print(f'  标准差: {np.std(max_values):.2f}')
            
            # 如果指定了 cut 值，显示 RT 事件统计
            if cut is not None:
                rt_event_count = np.sum(max_values > cut)
                rt_rate = rt_event_count / num_events * 100
                print(f'\n截断分析 (Cut = {cut:.2f}):')
                print(f'  RT触发事件数 (> cut): {rt_event_count}')
                print(f'  RT触发率: {rt_rate:.2f}%')
            
            # 绘制分布图
            fig, ax = plt.subplots(figsize=figsize)
            
            # 绘制直方图
            n, bins_edges, patches = ax.hist(max_values, bins=bins, edgecolor='black', alpha=0.7)
            
            # 添加统计信息
            mean_val = np.mean(max_values)
            median_val = np.median(max_values)
            std_val = np.std(max_values)
            
            # 计算 cut 线右侧的事件数量（RT触发事件）
            rt_event_count = 0
            if cut is not None:
                rt_event_count = np.sum(max_values > cut)
            
            # 在图上标注统计信息
            stats_text = (f'Total Events: {num_events}\n'
                         f'Mean: {mean_val:.2f}\n'
                         f'Median: {median_val:.2f}\n'
                         f'Std: {std_val:.2f}\n'
                         f'Min: {np.min(max_values):.2f}\n'
                         f'Max: {np.max(max_values):.2f}')
            
            if cut is not None:
                stats_text += f'\n\nCut: {cut:.2f}\nRT Events (>cut): {rt_event_count}\nRT Rate: {rt_event_count/num_events*100:.2f}%'
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   fontsize=10)
            
            # 添加 cut 截断线（如果指定了cut值）
            if cut is not None:
                ax.axvline(cut, color='red', linestyle='-', linewidth=2, 
                          label=f'Cut: {cut:.2f} (RT Events: {rt_event_count})')
                
                # 在 cut 线上方添加标注（在绘制后获取正确的 y 范围）
                y_max = ax.get_ylim()[1]
                ax.text(cut, y_max * 0.95, f'Cut={cut:.2f}\nRT Events: {rt_event_count}',
                       ha='center', va='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            ax.set_xlabel('Maximum Amplitude (ADC counts)', fontsize=12)
            ax.set_ylabel('Number of Events', fontsize=12)
            ax.set_title(f'CH5 Maximum Amplitude Distribution\nFile: {os.path.basename(h5_file)}', fontsize=11)
            if cut is not None:
                ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f'\n图片已保存至: {save_path}')
            
            if show_plot:
                plt.show()
            else:
                plt.close()
            
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
            bins=100,
            cut=6000,  # 设置截断阈值，cut线右侧（>cut）的事件为RT触发事件
            show_plot=True
        )
        print(f'\n分析完成！RT触发事件数: {stats["rt_event_count"]}')
    except Exception as e:
        print(f'分析失败: {e}')


