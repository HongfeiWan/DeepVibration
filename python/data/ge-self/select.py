#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
筛选并可视化Physical信号（既非RT也非Inhibit）的四个通道原始波形
筛选条件：既非RT（CH5 max <= rt_cut）也非Inhibit（CH0 min != 0）
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

def select_physical_events(ch0_3_file: str = None,
                          ch5_file: str = None,
                          rt_cut: float = 6000.0,
                          ch0_idx: int = 0,
                          ch5_idx: int = 0) -> Dict:
    """
    筛选既非RT也非Inhibit的Physical信号
    
    参数:
        ch0_3_file: CH0-3文件路径，如果为None则自动获取
        ch5_file: CH5文件路径，如果为None则自动获取
        rt_cut: RT信号截断阈值（CH5最大值 > rt_cut 为RT信号）
        ch0_idx: CH0通道索引
        ch5_idx: CH5通道索引
    
    返回:
        包含筛选结果的字典
    """
    print('=' * 70)
    print('筛选Physical信号（既非RT也非Inhibit）')
    print('=' * 70)
    
    # 如果没有指定文件，自动获取
    if ch0_3_file is None or ch5_file is None:
        h5_files = get_h5_files()
        if 'CH0-3' not in h5_files or not h5_files['CH0-3']:
            raise FileNotFoundError('在 data/hdf5/raw_pulse/CH0-3 目录中未找到 h5 文件')
        if 'CH5' not in h5_files or not h5_files['CH5']:
            raise FileNotFoundError('在 data/hdf5/raw_pulse/CH5 目录中未找到 h5 文件')
        
        # 查找匹配的文件对
        ch0_3_files = h5_files['CH0-3']
        ch5_files = h5_files['CH5']
        ch0_3_dict = {os.path.basename(f): f for f in ch0_3_files}
        ch5_dict = {os.path.basename(f): f for f in ch5_files}
        
        # 查找匹配的文件名
        matched = False
        for filename in ch0_3_dict.keys():
            if filename in ch5_dict:
                ch0_3_file = ch0_3_dict[filename] if ch0_3_file is None else ch0_3_file
                ch5_file = ch5_dict[filename] if ch5_file is None else ch5_file
                matched = True
                break
        
        if not matched:
            raise ValueError('未找到匹配的CH0-3和CH5文件对')
    
    if not os.path.exists(ch0_3_file):
        raise FileNotFoundError(f'文件不存在: {ch0_3_file}')
    if not os.path.exists(ch5_file):
        raise FileNotFoundError(f'文件不存在: {ch5_file}')
    
    print(f'\n文件:')
    print(f'  CH0-3: {os.path.basename(ch0_3_file)}')
    print(f'  CH5:   {os.path.basename(ch5_file)}')
    
    # 筛选既非RT也非Inhibit的events
    print(f'\n正在筛选既非RT也非Inhibit的events...')
    

    batch_size = 1000
    
    with h5py.File(ch0_3_file, 'r') as f_ch0:
        ch0_channel_data = f_ch0['channel_data']
        ch0_time_samples, ch0_num_channels, ch0_num_events = ch0_channel_data.shape
        ch0_min_values = np.zeros(ch0_num_events, dtype=np.float64)
        for i in range(0, ch0_num_events, batch_size):
            end_idx = min(i + batch_size, ch0_num_events)
            batch_data = ch0_channel_data[:, ch0_idx, i:end_idx]
            ch0_min_values[i:end_idx] = np.min(batch_data, axis=0)
    
    with h5py.File(ch5_file, 'r') as f_ch5:
        ch5_channel_data = f_ch5['channel_data']
        ch5_time_samples, ch5_num_channels, ch5_num_events = ch5_channel_data.shape
        ch5_max_values = np.zeros(ch5_num_events, dtype=np.float64)
        for i in range(0, ch5_num_events, batch_size):
            end_idx = min(i + batch_size, ch5_num_events)
            batch_data = ch5_channel_data[:, ch5_idx, i:end_idx]
            ch5_max_values[i:end_idx] = np.max(batch_data, axis=0)
        
        # 判断信号类型
        rt_mask = ch5_max_values > rt_cut
        inhibit_mask = ch0_min_values == 0  # 严格等于0
        neither_mask = ~rt_mask & ~inhibit_mask
        selected_indices = np.where(neither_mask)[0]
        trigger_times = None
        
        rt_count = np.sum(rt_mask)
        inhibit_count = np.sum(inhibit_mask)
        physical_count = len(selected_indices)
    
    num_events = len(selected_indices)
    print(f'  筛选完成:')
    print(f'    RT信号: {rt_count} 个')
    print(f'    Inhibit信号: {inhibit_count} 个')
    print(f'    Physical信号: {physical_count} 个')
    
    # 读取触发时间
    with h5py.File(ch0_3_file, 'r') as f_ch0:
        ch0_time_data = f_ch0['time_data']
        trigger_times = ch0_time_data[selected_indices]
    
    return {
        'ch0_3_file': ch0_3_file,
        'ch5_file': ch5_file,
        'selected_indices': selected_indices,
        'trigger_times': trigger_times,
        'rt_count': rt_count,
        'inhibit_count': inhibit_count,
        'physical_count': physical_count,
        'total_events': rt_count + inhibit_count + physical_count
    }

def plot_physical_waveforms(ch0_3_file: str = None,
                           ch5_file: str = None,
                           rt_cut: float = 6000.0,
                           ch0_idx: int = 0,
                           ch5_idx: int = 0,
                           max_events_to_plot: Optional[int] = 10,
                           save_path: Optional[str] = None,
                           show_plot: bool = True,
                           figsize: Tuple[int, int] = (16, 12)) -> None:
    """
    筛选并可视化Physical信号（既非RT也非Inhibit）的四个通道原始波形
    
    参数:
        ch0_3_file: CH0-3文件路径，如果为None则自动获取
        ch5_file: CH5文件路径，如果为None则自动获取
        rt_cut: RT信号截断阈值（CH5最大值 > rt_cut 为RT信号）
        ch0_idx: CH0通道索引
        ch5_idx: CH5通道索引
        max_events_to_plot: 最多绘制的Physical事件数量，None表示绘制所有
        save_path: 保存图片路径，如果为None则不保存
        show_plot: 是否显示图片
        figsize: 图片大小
    """
    # 1. 筛选Physical信号
    selection_result = select_physical_events(ch0_3_file, ch5_file, rt_cut, ch0_idx, ch5_idx)
    
    ch0_3_file = selection_result['ch0_3_file']
    selected_indices = selection_result['selected_indices']
    physical_count = selection_result['physical_count']
    
    if physical_count == 0:
        print('未发现Physical信号，无法绘制')
        return
    
    # 确定要绘制的事件数量
    if max_events_to_plot is None:
        num_to_plot = physical_count
    else:
        num_to_plot = min(max_events_to_plot, physical_count)
    
    selected_indices = selected_indices[:num_to_plot]
    print(f'\n将绘制 {num_to_plot} 个Physical信号的CH0-3和CH5通道波形')
    
    # 获取CH5文件路径
    ch5_file = selection_result['ch5_file']
    
    # 2. 读取文件获取波形数据
    with h5py.File(ch0_3_file, 'r') as f_ch0:
        ch0_channel_data = f_ch0['channel_data']
        time_samples_ch0, num_channels_ch0, num_events_ch0 = ch0_channel_data.shape
        
        # 确保有4个通道（CH0-3）
        if num_channels_ch0 < 4:
            print(f'警告: CH0-3文件只有 {num_channels_ch0} 个通道，将只绘制前 {num_channels_ch0} 个通道')
            num_channels_to_plot = num_channels_ch0
        else:
            num_channels_to_plot = 4
        
        # 读取CH5数据
        with h5py.File(ch5_file, 'r') as f_ch5:
            ch5_channel_data = f_ch5['channel_data']
            time_samples_ch5, num_channels_ch5, num_events_ch5 = ch5_channel_data.shape
            
            # 参数设置
            sampling_interval_ns = 4.0  # 4ns per sample
            sampling_interval_s = sampling_interval_ns * 1e-9
            time_axis_us_ch0 = np.arange(time_samples_ch0) * sampling_interval_ns / 1000.0  # 转换为微秒
            time_axis_us_ch5 = np.arange(time_samples_ch5) * sampling_interval_ns / 1000.0  # 转换为微秒
            
            # 通道名称（CH0-3 + CH5）
            channel_names = ['CH0', 'CH1', 'CH2', 'CH3', 'CH5']
            channel_colors = ['b', 'g', 'r', 'm', 'orange']  # 蓝色、绿色、红色、品红色、橙色
            
            # 3. 创建图形
            # 每个事件一行，每行5个子图（CH0-3 + CH5）
            n_cols = 5  # 4个CH0-3通道 + 1个CH5通道
            n_rows = num_to_plot  # 每个事件一行
            
            # 调整图形大小以适应5列
            fig_width = figsize[0] * (5 / 4)  # 按比例增加宽度
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, figsize[1]))
            
            # 如果只有一个事件，确保axes是二维数组
            if num_to_plot == 1:
                axes = axes.reshape(1, -1)
            
            # 4. 绘制每个Physical信号的CH0-3和CH5通道波形
            for plot_idx, event_idx in enumerate(selected_indices):
                # 绘制CH0-3的4个通道
                for ch_idx in range(num_channels_to_plot):
                    ax = axes[plot_idx, ch_idx] if n_rows > 1 else axes[ch_idx]
                    
                    # 获取CH0-3波形数据
                    waveform = ch0_channel_data[:, ch_idx, event_idx].astype(np.float64)
                    min_val = np.min(waveform)
                    max_val = np.max(waveform)
                    mean_val = np.mean(waveform)
                    std_val = np.std(waveform)
                    
                    # 绘制波形
                    color = channel_colors[ch_idx] if ch_idx < len(channel_colors) else 'b'
                    ax.plot(time_axis_us_ch0, waveform, color=color, linewidth=0.8, alpha=0.8)
                    
                    # 标注关键统计信息
                    ax.axhline(mean_val, color='gray', linestyle='--', linewidth=0.5, alpha=0.5, label=f'Mean: {mean_val:.1f}')
                    
                    # 设置标题和标签
                    if plot_idx == 0:
                        # 第一行显示通道名称
                        ax.set_title(f'{channel_names[ch_idx]}', fontsize=11, fontweight='bold')
                    else:
                        ax.set_title('', fontsize=9)
                    
                    if ch_idx == 0:
                        # 第一列显示事件编号
                        ax.set_ylabel(f'Event #{event_idx}\nAmplitude (ADC)', fontsize=9)
                    else:
                        ax.set_ylabel('Amplitude (ADC)', fontsize=9)
                    
                    if plot_idx == n_rows - 1:
                        # 最后一行显示时间轴标签
                        ax.set_xlabel('Time (μs)', fontsize=9)
                    else:
                        ax.set_xlabel('', fontsize=9)
                    
                    # 在图上显示统计信息
                    info_text = f'Min: {min_val:.1f}\nMax: {max_val:.1f}\nMean: {mean_val:.1f}\nStd: {std_val:.1f}'
                    ax.text(0.98, 0.98, info_text, transform=ax.transAxes,
                           verticalalignment='top', horizontalalignment='right',
                           fontsize=7, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    
                    ax.grid(True, alpha=0.3)
                
                # 绘制CH5通道（第5列）
                ch5_col_idx = 4
                ax = axes[plot_idx, ch5_col_idx] if n_rows > 1 else axes[ch5_col_idx]
                
                # 获取CH5波形数据
                ch5_waveform = ch5_channel_data[:, ch5_idx, event_idx].astype(np.float64)
                ch5_min_val = np.min(ch5_waveform)
                ch5_max_val = np.max(ch5_waveform)
                ch5_mean_val = np.mean(ch5_waveform)
                ch5_std_val = np.std(ch5_waveform)
                
                # 绘制CH5波形
                ch5_color = channel_colors[ch5_col_idx] if ch5_col_idx < len(channel_colors) else 'orange'
                ax.plot(time_axis_us_ch5, ch5_waveform, color=ch5_color, linewidth=0.8, alpha=0.8)
                
                # 标注关键统计信息
                ax.axhline(ch5_mean_val, color='gray', linestyle='--', linewidth=0.5, alpha=0.5, label=f'Mean: {ch5_mean_val:.1f}')
                
                # 设置标题和标签
                if plot_idx == 0:
                    # 第一行显示通道名称
                    ax.set_title(f'{channel_names[ch5_col_idx]}', fontsize=11, fontweight='bold')
                else:
                    ax.set_title('', fontsize=9)
                
                if plot_idx == n_rows - 1:
                    # 最后一行显示时间轴标签
                    ax.set_xlabel('Time (μs)', fontsize=9)
                else:
                    ax.set_xlabel('', fontsize=9)
                
                ax.set_ylabel('Amplitude (ADC)', fontsize=9)
                
                # 在图上显示统计信息
                ch5_info_text = f'Min: {ch5_min_val:.1f}\nMax: {ch5_max_val:.1f}\nMean: {ch5_mean_val:.1f}\nStd: {ch5_std_val:.1f}'
                ax.text(0.98, 0.98, ch5_info_text, transform=ax.transAxes,
                       verticalalignment='top', horizontalalignment='right',
                       fontsize=7, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                ax.grid(True, alpha=0.3)
        
        # 设置整体标题
        fig.suptitle(f'Physical Signals (Neither RT nor Inhibit) - CH0-3 and CH5 Waveforms\n'
                    f'Total: {physical_count} events, Displaying: {num_to_plot} events',
                    fontsize=13, fontweight='bold', y=0.995)
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'\n图片已保存: {save_path}')
        
        if show_plot:
            plt.show()
        else:
            plt.close()

if __name__ == '__main__':
    try:
        plot_physical_waveforms(
            ch0_3_file=None,  # 自动选择匹配的文件对
            ch5_file=None,
            rt_cut=6000.0,
            ch0_idx=0,
            ch5_idx=0,
            max_events_to_plot=3,  # 绘制前5个Physical信号
            show_plot=True
        )
    except Exception as e:
        print(f'分析失败: {e}')
        import traceback
        traceback.print_exc()

