#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Random Trigger 和 Inhibit 信号的 Coincident 分析
分析CH0-3和CH5目录中对应文件的coincident事件
"""
import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple, List

# 添加路径以便导入 utils 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
python_dir = os.path.dirname(os.path.dirname(current_dir))
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)
from utils.visualize import get_h5_files

def find_matched_file_pairs(ch0_3_file: str = None, ch5_file: str = None) -> List[Tuple[str, str]]:
    """
    查找CH0-3和CH5目录中文件名匹配的文件对
    参数:
        ch0_3_file: CH0-3文件的路径，如果为None则查找所有匹配的文件对
        ch5_file: CH5文件的路径，如果为None则查找所有匹配的文件对
    
    返回:
        文件对列表 [(ch0_3_path, ch5_path), ...]
    """
    h5_files = get_h5_files()
    
    if ch0_3_file is not None and ch5_file is not None:
        # 直接使用指定的文件对
        return [(ch0_3_file, ch5_file)]
    
    ch0_3_files = h5_files.get('CH0-3', [])
    ch5_files = h5_files.get('CH5', [])
    
    if not ch0_3_files or not ch5_files:
        raise FileNotFoundError('CH0-3或CH5目录中未找到文件')
    
    # 匹配文件名（去除路径，只比较文件名）
    matched_pairs = []
    
    # 创建文件名到路径的映射
    ch0_3_dict = {os.path.basename(f): f for f in ch0_3_files}
    ch5_dict = {os.path.basename(f): f for f in ch5_files}
    
    # 查找匹配的文件名
    for filename in ch0_3_dict.keys():
        if filename in ch5_dict:
            matched_pairs.append((ch0_3_dict[filename], ch5_dict[filename]))
    
    if not matched_pairs:
        raise ValueError('未找到匹配的文件对')
    
    return matched_pairs

def analyze_coincident_events(ch0_3_file: str,
                              ch5_file: str,
                              rt_cut: float = 6000.0,
                              ch0_idx: int = 0,
                              ch5_idx: int = 0,
                              verbose: bool = True) -> Dict:
    """
    分析一对文件的coincident事件
    参数:
        ch0_3_file: CH0-3文件路径
        ch5_file: CH5文件路径
        rt_cut: RT信号的截断阈值（CH5最大值 > rt_cut）
        ch0_idx: CH0-3文件中的CH0通道索引（默认0）
        ch5_idx: CH5文件中的通道索引（默认0）
        verbose: 是否打印详细进度信息
    返回:
        包含统计信息的字典
    """
    if not os.path.exists(ch0_3_file):
        raise FileNotFoundError(f'文件不存在: {ch0_3_file}')
    if not os.path.exists(ch5_file):
        raise FileNotFoundError(f'文件不存在: {ch5_file}')
    
    if verbose:
        print('=' * 70)
        print(f'分析文件对:')
        print(f'  CH0-3: {os.path.basename(ch0_3_file)}')
        print(f'  CH5:   {os.path.basename(ch5_file)}')
        print('=' * 70)
    
    try:
        # 读取CH0-3文件（分析CH0的最小值）
        with h5py.File(ch0_3_file, 'r') as f_ch0:
            ch0_channel_data = f_ch0['channel_data']
            ch0_time_samples, ch0_num_channels, ch0_num_events = ch0_channel_data.shape
            
            if verbose:
                print(f'\nCH0-3文件维度: (时间点={ch0_time_samples}, 通道数={ch0_num_channels}, 事件数={ch0_num_events})')
            
            if ch0_idx < 0 or ch0_idx >= ch0_num_channels:
                raise IndexError(f'CH0通道索引 {ch0_idx} 超出范围 [0, {ch0_num_channels-1}]')
            
            # 提取所有事件的CH0最小值
            if verbose:
                print(f'\n正在计算CH0信号最小值...')
            ch0_min_values = np.zeros(ch0_num_events, dtype=np.float64)
            batch_size = 1000
            
            for i in range(0, ch0_num_events, batch_size):
                end_idx = min(i + batch_size, ch0_num_events)
                batch_data = ch0_channel_data[:, ch0_idx, i:end_idx]
                batch_min = np.min(batch_data, axis=0)
                ch0_min_values[i:end_idx] = batch_min
                
                if verbose and ((i // batch_size + 1) % 10 == 0 or end_idx == ch0_num_events):
                    print(f'  已处理 {end_idx}/{ch0_num_events} 个事件 ({end_idx/ch0_num_events*100:.1f}%)')
        
        # 读取CH5文件（分析CH5的最大值）
        with h5py.File(ch5_file, 'r') as f_ch5:
            ch5_channel_data = f_ch5['channel_data']
            ch5_time_samples, ch5_num_channels, ch5_num_events = ch5_channel_data.shape
            
            if verbose:
                print(f'\nCH5文件维度: (时间点={ch5_time_samples}, 通道数={ch5_num_channels}, 事件数={ch5_num_events})')
            
            if ch5_idx < 0 or ch5_idx >= ch5_num_channels:
                raise IndexError(f'CH5通道索引 {ch5_idx} 超出范围 [0, {ch5_num_channels-1}]')
            
            # 检查事件数是否匹配
            if ch0_num_events != ch5_num_events:
                raise ValueError(f'事件数不匹配！CH0-3: {ch0_num_events}, CH5: {ch5_num_events}')
            
            # 提取所有事件的CH5最大值
            if verbose:
                print(f'\n正在计算CH5信号最大值...')
            ch5_max_values = np.zeros(ch5_num_events, dtype=np.float64)
            
            for i in range(0, ch5_num_events, batch_size):
                end_idx = min(i + batch_size, ch5_num_events)
                batch_data = ch5_channel_data[:, ch5_idx, i:end_idx]
                batch_max = np.max(batch_data, axis=0)
                ch5_max_values[i:end_idx] = batch_max
                
                if verbose and ((i // batch_size + 1) % 10 == 0 or end_idx == ch5_num_events):
                    print(f'  已处理 {end_idx}/{ch5_num_events} 个事件 ({end_idx/ch5_num_events*100:.1f}%)')
        
        # 判断信号类型
        # RT信号：CH5最大值 > rt_cut
        rt_mask = ch5_max_values > rt_cut
        rt_count = np.sum(rt_mask)
        
        # Inhibit信号：CH0最小值 <= 0
        inhibit_mask = ch0_min_values <= 0
        inhibit_count = np.sum(inhibit_mask)
        
        # Coincident信号：同时满足RT和Inhibit条件
        coincident_mask = rt_mask & inhibit_mask
        coincident_count = np.sum(coincident_mask)
        
        total_events = ch0_num_events
        
        # 打印统计信息
        if verbose:
            print(f'\n' + '=' * 70)
            print(f'Coincident 分析结果:')
            print('=' * 70)
            print(f'总事件数: {total_events}')
            print(f'\nRandom Trigger 信号数量 (CH5 max > {rt_cut:.2f}): {rt_count}')
            print(f'Random Trigger 信号比例: {rt_count/total_events*100:.2f}%')
            print(f'\nInhibit 信号数量 (CH0 min <= 0): {inhibit_count}')
            print(f'Inhibit 信号比例: {inhibit_count/total_events*100:.2f}%')
            print(f'\nCoincident 信号数量 (同时满足RT和Inhibit): {coincident_count}')
            print(f'Coincident 信号比例: {coincident_count/total_events*100:.2f}%')
            print(f'\n信号分类统计:')
            print(f'  仅RT信号 (非Inhibit): {rt_count - coincident_count}')
            print(f'  仅Inhibit信号 (非RT): {inhibit_count - coincident_count}')
            print(f'  既非RT也非Inhibit: {total_events - rt_count - inhibit_count + coincident_count}')
            print('=' * 70)
        
        # 返回统计结果
        stats = {
            'total_events': int(total_events),
            'rt_count': int(rt_count),
            'rt_rate': float(rt_count / total_events * 100),
            'inhibit_count': int(inhibit_count),
            'inhibit_rate': float(inhibit_count / total_events * 100),
            'coincident_count': int(coincident_count),
            'coincident_rate': float(coincident_count / total_events * 100),
            'rt_only_count': int(rt_count - coincident_count),
            'inhibit_only_count': int(inhibit_count - coincident_count),
            'neither_count': int(total_events - rt_count - inhibit_count + coincident_count),
            'rt_mask': rt_mask,
            'inhibit_mask': inhibit_mask,
            'coincident_mask': coincident_mask,
            'rt_cut': float(rt_cut),
            'ch0_min_values': ch0_min_values,
            'ch5_max_values': ch5_max_values
        }
        
        return stats
    
    except Exception as e:
        print(f'分析过程中出错: {e}')
        raise

def plot_single_rt_only_event(ch0_3_file: str,
                              ch5_file: str,
                              event_index: int = 0,
                              rt_cut: float = 6000.0,
                              ch0_idx: int = 0,
                              ch5_idx: int = 0,
                              save_path: Optional[str] = None,
                              show_plot: bool = True,
                              figsize: Tuple[int, int] = (14, 8)) -> None:
    """
    可视化单个符合条件的仅RT信号（非Inhibit）的event的CH0信号形状
    
    参数:
        ch0_3_file: CH0-3文件路径
        ch5_file: CH5文件路径
        event_index: 要查看的event在符合条件的events中的索引（默认0，表示第一个）
        rt_cut: RT信号的截断阈值（CH5最大值 > rt_cut）
        ch0_idx: CH0通道索引
        ch5_idx: CH5通道索引
        save_path: 保存图片路径，如果为None则不保存
        show_plot: 是否显示图片
        figsize: 图片大小
    """
    if not os.path.exists(ch0_3_file):
        raise FileNotFoundError(f'文件不存在: {ch0_3_file}')
    if not os.path.exists(ch5_file):
        raise FileNotFoundError(f'文件不存在: {ch5_file}')
    
    print('=' * 70)
    print(f'可视化单个仅RT信号（非Inhibit）的event')
    print('=' * 70)
    
    # 先进行分析，获取筛选结果
    stats = analyze_coincident_events(
        ch0_3_file, ch5_file, rt_cut, ch0_idx, ch5_idx, verbose=False
    )
    
    # 仅RT信号（非Inhibit）：RT信号且非Inhibit信号
    rt_only_mask = stats['rt_mask'] & ~stats['inhibit_mask']
    rt_only_indices = np.where(rt_only_mask)[0]
    
    rt_only_count = len(rt_only_indices)
    print(f'\n仅RT信号（非Inhibit）数量: {rt_only_count}')
    
    if rt_only_count == 0:
        print('没有仅RT信号（非Inhibit）的events，无法绘制')
        return
    
    # 检查索引是否有效
    if event_index < 0 or event_index >= rt_only_count:
        raise IndexError(f'event_index {event_index} 超出范围 [0, {rt_only_count-1}]')
    
    # 获取要查看的event的实际索引
    actual_event_idx = rt_only_indices[event_index]
    print(f'查看第 {event_index} 个符合条件的event（实际索引: {actual_event_idx}）')
    
    # 读取文件获取波形数据和时间戳
    with h5py.File(ch0_3_file, 'r') as f_ch0:
        ch0_channel_data = f_ch0['channel_data']
        ch0_time_data = f_ch0['time_data']
        ch0_time_samples, ch0_num_channels, ch0_num_events = ch0_channel_data.shape
        
        # 参数设置
        sampling_interval_ns = 4.0  # 4ns per sample
        sampling_interval_s = sampling_interval_ns * 1e-9  # 转换为秒
        
        # 获取event的触发时间
        event_trigger_time = ch0_time_data[actual_event_idx]
        
        # 获取CH0的波形数据
        ch0_waveform = ch0_channel_data[:, ch0_idx, actual_event_idx]
        
        # 计算时间轴（相对于event开始的时间，单位：微秒）
        time_axis_us = np.arange(ch0_time_samples) * sampling_interval_ns / 1000.0  # 转换为微秒
        
        # 计算统计信息
        ch0_min = np.min(ch0_waveform)
        ch0_max = np.max(ch0_waveform)
        ch0_mean = np.mean(ch0_waveform)
        ch0_std = np.std(ch0_waveform)
        min_idx = np.argmin(ch0_waveform)
        max_idx = np.argmax(ch0_waveform)
        
        print(f'\nEvent信息:')
        print(f'  实际event索引: {actual_event_idx}')
        print(f'  触发时间: {event_trigger_time:.6f} s')
        print(f'  采样点数: {ch0_time_samples}')
        print(f'  采样间隔: {sampling_interval_ns} ns')
        print(f'  Event时长: {ch0_time_samples * sampling_interval_s * 1e6:.2f} μs')
        print(f'\nCH0信号统计:')
        print(f'  最小值: {ch0_min:.2f} (位置: {min_idx}, 时间: {time_axis_us[min_idx]:.2f} μs)')
        print(f'  最大值: {ch0_max:.2f} (位置: {max_idx}, 时间: {time_axis_us[max_idx]:.2f} μs)')
        print(f'  平均值: {ch0_mean:.2f}')
        print(f'  标准差: {ch0_std:.2f}')
        print(f'  最小值 > 0: {ch0_min > 0} (非Inhibit)')
        
        # 读取CH5文件获取CH5最大值（用于确认RT条件）
        with h5py.File(ch5_file, 'r') as f_ch5:
            ch5_channel_data = f_ch5['channel_data']
            ch5_max = np.max(ch5_channel_data[:, ch5_idx, actual_event_idx])
            print(f'\nCH5信号统计:')
            print(f'  最大值: {ch5_max:.2f}')
            print(f'  CH5 max > {rt_cut}: {ch5_max > rt_cut} (RT信号)')
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. 完整波形（时间轴：微秒）
        ax = axes[0, 0]
        ax.plot(time_axis_us, ch0_waveform, 'b-', linewidth=1.0, alpha=0.8)
        ax.set_xlabel('Time (μs)', fontsize=10)
        ax.set_ylabel('Amplitude (ADC counts)', fontsize=10)
        ax.set_title(f'CH0 Waveform - Event #{actual_event_idx}\n'
                    f'Trigger Time: {event_trigger_time:.6f} s',
                    fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 标注最小值和最大值点
        ax.plot(time_axis_us[min_idx], ch0_min, 'ro', markersize=8, label=f'Min: {ch0_min:.1f}')
        ax.plot(time_axis_us[max_idx], ch0_max, 'go', markersize=8, label=f'Max: {ch0_max:.1f}')
        ax.legend(fontsize=9)
        
        # 2. 完整波形（时间轴：采样点）
        ax = axes[0, 1]
        ax.plot(np.arange(ch0_time_samples), ch0_waveform, 'b-', linewidth=1.0, alpha=0.8)
        ax.set_xlabel('Sample Index', fontsize=10)
        ax.set_ylabel('Amplitude (ADC counts)', fontsize=10)
        ax.set_title(f'CH0 Waveform - Event #{actual_event_idx}\n'
                    f'Total Samples: {ch0_time_samples}',
                    fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 标注最小值和最大值点
        ax.plot(min_idx, ch0_min, 'ro', markersize=8, label=f'Min: {ch0_min:.1f}')
        ax.plot(max_idx, ch0_max, 'go', markersize=8, label=f'Max: {ch0_max:.1f}')
        ax.legend(fontsize=9)
        
        # 3. 波形的前一部分（放大显示）
        ax = axes[1, 0]
        zoom_samples = min(5000, ch0_time_samples)
        ax.plot(time_axis_us[:zoom_samples], ch0_waveform[:zoom_samples], 
                'b-', linewidth=1.2, alpha=0.9)
        ax.set_xlabel('Time (μs)', fontsize=10)
        ax.set_ylabel('Amplitude (ADC counts)', fontsize=10)
        ax.set_title(f'Zoomed View (First {zoom_samples} samples)', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 4. 统计信息文本
        ax = axes[1, 1]
        ax.axis('off')  # 关闭坐标轴，只显示文本
        
        info_text = f'Event Information:\n'
        info_text += f'  Event Index: {actual_event_idx}\n'
        info_text += f'  Trigger Time: {event_trigger_time:.6f} s\n'
        info_text += f'  Duration: {ch0_time_samples * sampling_interval_s * 1e6:.2f} μs\n'
        info_text += f'  Sampling Rate: {1.0/sampling_interval_s/1e6:.2f} MSPS\n'
        info_text += f'\nCH0 Signal Statistics:\n'
        info_text += f'  Min: {ch0_min:.2f} (at {time_axis_us[min_idx]:.2f} μs)\n'
        info_text += f'  Max: {ch0_max:.2f} (at {time_axis_us[max_idx]:.2f} μs)\n'
        info_text += f'  Mean: {ch0_mean:.2f}\n'
        info_text += f'  Std: {ch0_std:.2f}\n'
        info_text += f'  Min > 0: {ch0_min > 0} ✓ (Non-Inhibit)\n'
        info_text += f'\nCH5 Signal Statistics:\n'
        info_text += f'  Max: {ch5_max:.2f}\n'
        info_text += f'  Max > {rt_cut}: {ch5_max > rt_cut} ✓ (RT Signal)\n'
        info_text += f'\nClassification:\n'
        info_text += f'  RT Only (Non-Inhibit): ✓'
        
        ax.text(0.1, 0.5, info_text, fontsize=10, family='monospace',
               verticalalignment='center', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'RT Only (Non-Inhibit) Event #{actual_event_idx} - CH0 Waveform',
                    fontsize=12, y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'\n图片已保存至: {save_path}')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    print(f'\n绘制完成！')

def plot_rt_only_waveforms(ch0_3_file: str,
                           ch5_file: str,
                           rt_cut: float = 6000.0,
                           ch0_idx: int = 0,
                           ch5_idx: int = 0,
                           max_events_to_plot: Optional[int] = None,
                           save_path: Optional[str] = None,
                           show_plot: bool = True,
                           figsize: Tuple[int, int] = (16, 10)) -> None:
    """
    在同一个坐标系下绘制所有仅RT信号（非Inhibit）的波形
    使用绝对时间戳，将所有event的波形按时间顺序绘制在一起
    
    参数:
        ch0_3_file: CH0-3文件路径
        ch5_file: CH5文件路径
        rt_cut: RT信号的截断阈值（CH5最大值 > rt_cut）
        ch0_idx: CH0通道索引
        ch5_idx: CH5通道索引
        max_events_to_plot: 最多绘制的event数量（默认None表示绘制所有）
        save_path: 保存图片路径，如果为None则不保存
        show_plot: 是否显示图片
        figsize: 图片大小
    """
    if not os.path.exists(ch0_3_file):
        raise FileNotFoundError(f'文件不存在: {ch0_3_file}')
    if not os.path.exists(ch5_file):
        raise FileNotFoundError(f'文件不存在: {ch5_file}')
    
    print('=' * 70)
    print(f'绘制仅RT信号（非Inhibit）的波形（同一坐标系）')
    print('=' * 70)
    
    # 先进行分析，获取筛选结果
    stats = analyze_coincident_events(
        ch0_3_file, ch5_file, rt_cut, ch0_idx, ch5_idx, verbose=False
    )
    
    # 仅RT信号（非Inhibit）：RT信号且非Inhibit信号
    rt_only_mask = stats['rt_mask'] & ~stats['inhibit_mask']
    rt_only_indices = np.where(rt_only_mask)[0]
    
    rt_only_count = len(rt_only_indices)
    print(f'\n仅RT信号（非Inhibit）数量: {rt_only_count}')
    
    if rt_only_count == 0:
        print('没有仅RT信号（非Inhibit）的events，无法绘制')
        return
    
    # 限制绘制的event数量（如果指定）
    if max_events_to_plot is not None:
        num_events_to_plot = min(max_events_to_plot, rt_only_count)
        selected_indices = rt_only_indices[:num_events_to_plot]
        print(f'绘制前 {num_events_to_plot} 个仅RT信号（非Inhibit）的波形')
    else:
        selected_indices = rt_only_indices
        num_events_to_plot = rt_only_count
        print(f'绘制所有 {num_events_to_plot} 个仅RT信号（非Inhibit）的波形')
    
    # 读取文件获取波形数据和时间戳
    with h5py.File(ch0_3_file, 'r') as f_ch0:
        ch0_channel_data = f_ch0['channel_data']
        ch0_time_data = f_ch0['time_data']
        ch0_time_samples, ch0_num_channels, ch0_num_events = ch0_channel_data.shape
        
        # 参数设置
        sampling_interval_ns = 4.0  # 4ns per sample
        sampling_interval_s = sampling_interval_ns * 1e-9  # 转换为秒
        event_duration_s = ch0_time_samples * sampling_interval_s  # 每个event的时长
        
        print(f'\n正在收集波形数据...')
        print(f'每个event参数:')
        print(f'  采样点数: {ch0_time_samples}')
        print(f'  采样间隔: {sampling_interval_ns} ns = {sampling_interval_s} s')
        print(f'  Event时长: {event_duration_s*1e6:.2f} μs')
        
        # 收集所有event的波形点和对应的时间戳
        # 注意：event之间不是首尾相连的，它们有各自独立的触发时间
        all_waveform_values = []
        all_time_stamps = []
        
        event_info = []  # 存储每个event的信息，用于标注
        
        for i, event_idx in enumerate(selected_indices):
            # 获取event的触发时间（这是event开始的时间戳）
            event_trigger_time = ch0_time_data[event_idx]
            
            # 获取CH0的波形数据
            ch0_waveform = ch0_channel_data[:, ch0_idx, event_idx]
            
            # 计算这个event内每个采样点的绝对时间戳
            # 绝对时间 = 触发时间（event开始时间） + 采样点索引 * 采样间隔
            event_times = event_trigger_time + np.arange(ch0_time_samples) * sampling_interval_s
            
            # 添加到总数组中
            all_waveform_values.extend(ch0_waveform)
            all_time_stamps.extend(event_times)
            
            # 记录event信息
            event_info.append({
                'event_idx': event_idx,
                'trigger_time': event_trigger_time,
                'start_time': event_times[0],
                'end_time': event_times[-1],
                'duration': event_duration_s
            })
        
        # 转换为numpy数组
        all_waveform_values = np.array(all_waveform_values)
        all_time_stamps = np.array(all_time_stamps)
        
        # 排序（按时间戳排序，确保时间顺序正确）
        sort_indices = np.argsort(all_time_stamps)
        all_waveform_values = all_waveform_values[sort_indices]
        all_time_stamps = all_time_stamps[sort_indices]
        
        print(f'\n收集完成: {len(all_waveform_values)} 个数据点')
        print(f'时间跨度: {all_time_stamps[0]:.6f} s 到 {all_time_stamps[-1]:.6f} s')
        print(f'总时间长度: {all_time_stamps[-1] - all_time_stamps[0]:.6f} s')
        
        # 统计event之间的间隔
        if len(event_info) > 1:
            intervals = []
            for i in range(1, len(event_info)):
                interval = event_info[i]['trigger_time'] - event_info[i-1]['trigger_time']
                intervals.append(interval)
            intervals = np.array(intervals)
            print(f'\nEvent间隔统计:')
            print(f'  平均间隔: {np.mean(intervals):.6f} s ({np.mean(intervals)*1e3:.2f} ms)')
            print(f'  最小间隔: {np.min(intervals):.6f} s ({np.min(intervals)*1e3:.2f} ms)')
            print(f'  最大间隔: {np.max(intervals):.6f} s ({np.max(intervals)*1e3:.2f} ms)')
            print(f'  标准差: {np.std(intervals):.6f} s ({np.std(intervals)*1e3:.2f} ms)')
        
        # 创建图形
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        # 1. 完整波形图（分别绘制每个event，保持event之间的间隔）
        ax = axes[0]
        # 分别绘制每个event，这样event之间不会连线
        for i, event_idx in enumerate(selected_indices):
            # 获取event的触发时间
            event_trigger_time = ch0_time_data[event_idx]
            
            # 获取CH0的波形数据
            ch0_waveform = ch0_channel_data[:, ch0_idx, event_idx]
            
            # 计算这个event内每个采样点的绝对时间戳
            event_times = event_trigger_time + np.arange(ch0_time_samples) * sampling_interval_s
            
            # 分别绘制每个event（这样event之间不会连线）
            ax.plot(event_times, ch0_waveform, 'b-', linewidth=0.5, alpha=0.7)
        
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Amplitude (ADC counts)', fontsize=10)
        ax.set_title(f'RT Only (Non-Inhibit) Waveforms - All Events\n'
                    f'Total events: {num_events_to_plot}, Each event is drawn separately\n'
                    f'Note: Events are NOT connected - they have independent trigger times with intervals',
                    fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 标记每个event的触发时间（用竖线，最多标记20个避免太拥挤）
        for i, info in enumerate(event_info[:20]):
            ax.axvline(info['trigger_time'], color='red', linestyle='--', 
                      linewidth=0.5, alpha=0.5)
            if i == 0:
                ax.text(info['trigger_time'], ax.get_ylim()[1] * 0.9, 'Event Start', 
                       rotation=90, ha='right', va='top', fontsize=8, color='red')
        
        # 2. 放大视图（前几个event，清楚显示间隔）
        ax = axes[1]
        # 显示前几个event的波形（分别绘制，不连线）
        num_events_to_show = min(5, len(selected_indices))
        for i in range(num_events_to_show):
            event_idx = selected_indices[i]
            # 获取event的触发时间
            event_trigger_time = ch0_time_data[event_idx]
            
            # 获取CH0的波形数据
            ch0_waveform = ch0_channel_data[:, ch0_idx, event_idx]
            
            # 计算这个event内每个采样点的绝对时间戳
            event_times = event_trigger_time + np.arange(ch0_time_samples) * sampling_interval_s
            
            # 分别绘制每个event
            ax.plot(event_times, ch0_waveform, 'b-', linewidth=0.8, alpha=0.8)
        
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Amplitude (ADC counts)', fontsize=10)
        ax.set_title(f'Zoomed View (First {num_events_to_show} events, showing event intervals)\n'
                    f'Each event is drawn separately - gaps between events are visible',
                    fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 标记event边界（显示event之间的间隔）
        for i, info in enumerate(event_info[:num_events_to_show]):
            ax.axvline(info['trigger_time'], color='red', linestyle='--', 
                      linewidth=1, alpha=0.6, 
                      label='Event Start' if i == 0 else '')
            ax.axvline(info['end_time'], color='green', linestyle='--', 
                      linewidth=1, alpha=0.6, 
                      label='Event End' if i == 0 else '')
        if len(event_info) > 0:
            ax.legend(fontsize=8, loc='upper right')
        
        # 3. 时间轴转换为相对时间（从第一个event开始，方便查看间隔）
        ax = axes[2]
        # 分别绘制每个event（相对时间）
        first_event_time = all_time_stamps[0]
        for i, event_idx in enumerate(selected_indices[:10]):  # 只显示前10个event
            # 获取event的触发时间
            event_trigger_time = ch0_time_data[event_idx]
            
            # 获取CH0的波形数据
            ch0_waveform = ch0_channel_data[:, ch0_idx, event_idx]
            
            # 计算这个event内每个采样点的绝对时间戳
            event_times = event_trigger_time + np.arange(ch0_time_samples) * sampling_interval_s
            
            # 转换为相对时间（毫秒）
            relative_times = (event_times - first_event_time) * 1e3
            
            # 分别绘制每个event
            ax.plot(relative_times, ch0_waveform, 'b-', linewidth=0.5, alpha=0.7)
        
        ax.set_xlabel('Time (ms) - Relative to First Event', fontsize=10)
        ax.set_ylabel('Amplitude (ADC counts)', fontsize=10)
        ax.set_title(f'Waveforms with Relative Time Axis (showing event intervals and gaps)\n'
                    f'Each event is drawn separately - gaps between events are clearly visible',
                    fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 标记event边界（相对时间），显示间隔
        for i, info in enumerate(event_info[:10]):  # 标记前10个event
            rel_start = (info['start_time'] - first_event_time) * 1e3
            rel_end = (info['end_time'] - first_event_time) * 1e3
            ax.axvline(rel_start, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.axvline(rel_end, color='green', linestyle='--', linewidth=0.5, alpha=0.5)
            # 标注间隔（在event之间）
            if i > 0:
                prev_end = (event_info[i-1]['end_time'] - first_event_time) * 1e3
                gap_center = (rel_start + prev_end) / 2
                gap_duration = (info['trigger_time'] - event_info[i-1]['end_time']) * 1e3
                if gap_duration > 0:  # 只在有间隔时标注
                    ax.text(gap_center, ax.get_ylim()[1] * 0.95, f'Gap: {gap_duration:.2f} ms',
                           ha='center', va='top', fontsize=7, 
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'\n图片已保存至: {save_path}')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    print(f'\n绘制完成！')

# 示例使用
if __name__ == '__main__':
    # 方法1: 分析单个文件对
    print('=' * 70)
    print('方法1: 分析单个文件对')
    print('=' * 70)
    
    try:
        # 自动查找匹配的文件对，分析第一对
        matched_pairs = find_matched_file_pairs()
        if matched_pairs:
            ch0_3_file, ch5_file = matched_pairs[0]
            stats = analyze_coincident_events(
                ch0_3_file, ch5_file,
                rt_cut=6000.0,  # RT信号截断阈值
                ch0_idx=0,      # CH0通道索引
                ch5_idx=0,      # CH5通道索引
                verbose=True
            )
            
            print(f'\n关键统计结果:')
            print(f'  Random Trigger 信号数量: {stats["rt_count"]}')
            print(f'  Inhibit 信号数量: {stats["inhibit_count"]}')
            print(f'  Coincident 信号数量: {stats["coincident_count"]}')
            
            # 可视化单个符合条件的event
            print(f'\n' + '=' * 70)
            print('可视化单个仅RT信号（非Inhibit）的event')
            print('=' * 70)
            plot_single_rt_only_event(
                ch0_3_file, ch5_file,
                event_index=1,  # 查看第一个符合条件的event
                rt_cut=6000.0,
                ch0_idx=0,
                ch5_idx=0,
                show_plot=True
            )
            
            # 绘制所有仅RT信号（非Inhibit）的波形
            print(f'\n' + '=' * 70)
            print('绘制所有仅RT信号（非Inhibit）的波形')
            print('=' * 70)
            plot_rt_only_waveforms(
                ch0_3_file, ch5_file,
                rt_cut=6000.0,
                ch0_idx=0,
                ch5_idx=0,
                max_events_to_plot=10,  # 最多绘制10个波形
                show_plot=True
            )
    except Exception as e:
        print(f'分析失败: {e}')
    

