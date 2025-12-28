#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
原始脉冲数据可视化工具
用于读取和可视化 HDF5 格式的原始脉冲数据
"""
import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any

def get_h5_files(base_dir: str = None) -> Dict[str, List[str]]:
    """
    获取 raw_pulse 目录下所有 h5 文件
    
    参数:
        base_dir: raw_pulse 目录路径，如果为None则自动查找项目根目录
    
    返回:
        字典，键为子目录名（如'CH0-3', 'CH5'），值为该目录下的h5文件列表
    """
    if base_dir is None:
        # 自动查找项目根目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        base_dir = os.path.join(project_root, 'data', 'hdf5', 'raw_pulse')
    
    h5_files = {}
    
    # 遍历 raw_pulse 目录下的所有子目录
    if os.path.exists(base_dir):
        for subdir in os.listdir(base_dir):
            subdir_path = os.path.join(base_dir, subdir)
            if os.path.isdir(subdir_path):
                # 查找所有 .h5 文件
                files = glob.glob(os.path.join(subdir_path, '*.h5'))
                if files:
                    h5_files[subdir] = sorted(files)
    else:
        print(f'警告：目录不存在: {base_dir}')
    
    return h5_files

def show_h5_structure(h5_file: str) -> Dict[str, Any]:
    """
    显示 HDF5 文件的数据结构
    
    参数:
        h5_file: HDF5 文件路径
    
    返回:
        包含数据结构信息的字典
    """
    if not os.path.exists(h5_file):
        raise FileNotFoundError(f'文件不存在: {h5_file}')
    
    print('=' * 70)
    print(f'文件: {os.path.basename(h5_file)}')
    print(f'路径: {h5_file}')
    print('=' * 70)
    
    structure_info = {}
    
    try:
        with h5py.File(h5_file, 'r') as f:
            # 显示文件大小
            file_size = os.path.getsize(h5_file)
            file_size_GB = file_size / (1024 ** 3)
            print(f'\n文件大小: {file_size_GB:.4f} GB ({file_size:,} bytes)')
            
            # 遍历所有数据集
            print('\n数据集结构:')
            print('-' * 70)
            
            for key in f.keys():
                dataset = f[key]
                
                print(f'\n数据集名称: {key}')
                print(f'  形状 (shape): {dataset.shape}')
                print(f'  数据类型 (dtype): {dataset.dtype}')
                print(f'  维度数: {len(dataset.shape)}')
                
                # 计算大小
                if isinstance(dataset, h5py.Dataset):
                    size_GB = dataset.nbytes / (1024 ** 3)
                    print(f'  大小: {size_GB:.4f} GB ({dataset.nbytes:,} bytes)')
                    
                    # 显示统计信息
                    if dataset.size > 0:
                        data_sample = dataset[:min(10000, dataset.size)]  # 采样计算统计信息
                        print(f'  数值范围: [{np.min(data_sample)}, {np.max(data_sample)}]')
                    
                    # 根据维度解释数据
                    if key == 'channel_data':
                        if len(dataset.shape) == 3:
                            time_samples, num_channels, num_events = dataset.shape
                            print(f'  解释: (时间采样点数, 通道数, 事件数)')
                            print(f'        = ({time_samples}, {num_channels}, {num_events})')
                            print(f'        每个波形有 {time_samples} 个时间点')
                            print(f'        共 {num_channels} 个通道')
                            print(f'        共 {num_events} 个事件')
                    elif key == 'time_data':
                        if len(dataset.shape) == 1:
                            num_events = dataset.shape[0]
                            print(f'  解释: (事件数,) = ({num_events},)')
                            print(f'        每个事件对应一个时间戳')
                
                structure_info[key] = {
                    'shape': dataset.shape,
                    'dtype': str(dataset.dtype),
                    'size_GB': dataset.nbytes / (1024 ** 3) if isinstance(dataset, h5py.Dataset) else 0
                }
            
            # 显示属性
            if f.attrs:
                print('\n文件属性:')
                print('-' * 70)
                for attr_name in f.attrs:
                    print(f'  {attr_name}: {f.attrs[attr_name]}')
        
        print('\n' + '=' * 70)
        
    except Exception as e:
        print(f'读取文件时出错: {e}')
        raise
    
    return structure_info

def visualize_waveform(h5_file: str, 
                      event_idx: int = 0,
                      channel_idx: int = 0,
                      time_unit: str = 'ns',
                      save_path: Optional[str] = None,
                      show_plot: bool = True,
                      figsize: Tuple[int, int] = (12, 6)) -> Tuple[np.ndarray, np.ndarray]:
    """
    可视化指定事件和通道的波形
    参数:
        h5_file: HDF5 文件路径
        event_idx: 事件索引（从0开始）
        channel_idx: 通道索引（从0开始，对应channel_list中的索引）
        time_unit: 时间单位 ('ns', 'us', 'ms', 's')，默认'ns'（假设采样间隔4ns）
        save_path: 保存图片的路径，如果为None则不保存
        show_plot: 是否显示图片
        figsize: 图片大小 (宽度, 高度)
    
    返回:
        (时间数组, 幅度数组) 的元组
    """
    if not os.path.exists(h5_file):
        raise FileNotFoundError(f'文件不存在: {h5_file}')
    
    try:
        with h5py.File(h5_file, 'r') as f:
            # 检查数据集是否存在
            if 'channel_data' not in f:
                raise KeyError('文件中没有找到 channel_data 数据集')
            if 'time_data' not in f:
                raise KeyError('文件中没有找到 time_data 数据集')
            
            channel_data = f['channel_data']
            time_data = f['time_data']
            
            # 检查索引范围
            time_samples, num_channels, num_events = channel_data.shape
            
            if event_idx < 0 or event_idx >= num_events:
                raise IndexError(f'事件索引 {event_idx} 超出范围 [0, {num_events-1}]')
            if channel_idx < 0 or channel_idx >= num_channels:
                raise IndexError(f'通道索引 {channel_idx} 超出范围 [0, {num_channels-1}]')
            
            # 读取波形数据
            waveform = channel_data[:, channel_idx, event_idx]  # shape: (time_samples,)
            event_time = time_data[event_idx]
            
            # 创建时间轴（假设采样间隔为4ns，这是V1725的默认采样率）
            sampling_interval_ns = 4.0  # 4ns per sample
            time_axis_ns = np.arange(len(waveform)) * sampling_interval_ns
            
            # 转换时间单位
            time_units = {
                'ns': 1.0,
                'us': 1e-3,
                'ms': 1e-6,
                's': 1e-9
            }
            if time_unit not in time_units:
                raise ValueError(f'不支持的时间单位: {time_unit}，支持: {list(time_units.keys())}')
            
            scale_factor = time_units[time_unit]
            time_axis = time_axis_ns * scale_factor
            
            # 绘制波形
            plt.figure(figsize=figsize)
            plt.plot(time_axis, waveform, linewidth=1.0)
            plt.xlabel(f'Time ({time_unit})', fontsize=12)
            plt.ylabel('Amplitude (ADC counts)', fontsize=12)
            plt.title(f'Waveform Visualization\nFile: {os.path.basename(h5_file)}\nEvent #{event_idx}, Channel #{channel_idx}, Event Time: {event_time:.6f} s', 
                     fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # 添加统计信息
            stats_text = (f'Time Points: {len(waveform)}\n'
                         f'Min: {np.min(waveform)}\n'
                         f'Max: {np.max(waveform)}\n'
                         f'Mean: {np.mean(waveform):.2f}\n'
                         f'Std: {np.std(waveform):.2f}')
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=9)
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f'图片已保存至: {save_path}')
            
            if show_plot:
                plt.show()
            else:
                plt.close()
            
            return time_axis, waveform
    
    except Exception as e:
        print(f'可视化波形时出错: {e}')
        raise

def visualize_multiple_channels(h5_file: str,
                                event_idx: int = 0,
                                channel_indices: Optional[List[int]] = None,
                                time_unit: str = 'ns',
                                save_path: Optional[str] = None,
                                show_plot: bool = True,
                                figsize: Tuple[int, int] = (14, 8)) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    同时可视化多个通道的波形（在同一事件中）
    
    参数:
        h5_file: HDF5 文件路径
        event_idx: 事件索引（从0开始）
        channel_indices: 要显示的通道索引列表，如果为None则显示所有通道
        time_unit: 时间单位 ('ns', 'us', 'ms', 's')
        save_path: 保存图片的路径，如果为None则不保存
        show_plot: 是否显示图片
        figsize: 图片大小 (宽度, 高度)
    
    返回:
        字典，键为通道索引，值为(time_array, waveform_array)的元组
    """
    if not os.path.exists(h5_file):
        raise FileNotFoundError(f'文件不存在: {h5_file}')
    
    try:
        with h5py.File(h5_file, 'r') as f:
            channel_data = f['channel_data']
            time_data = f['time_data']
            
            time_samples, num_channels, num_events = channel_data.shape
            
            if event_idx < 0 or event_idx >= num_events:
                raise IndexError(f'事件索引 {event_idx} 超出范围 [0, {num_events-1}]')
            
            if channel_indices is None:
                channel_indices = list(range(num_channels))
            else:
                # 验证通道索引
                for ch_idx in channel_indices:
                    if ch_idx < 0 or ch_idx >= num_channels:
                        raise IndexError(f'通道索引 {ch_idx} 超出范围 [0, {num_channels-1}]')
            
            event_time = time_data[event_idx]
            
            # 创建时间轴
            sampling_interval_ns = 4.0
            time_units = {
                'ns': 1.0, 'us': 1e-3, 'ms': 1e-6, 's': 1e-9
            }
            if time_unit not in time_units:
                raise ValueError(f'不支持的时间单位: {time_unit}')
            
            time_axis_ns = np.arange(time_samples) * sampling_interval_ns
            time_axis = time_axis_ns * time_units[time_unit]
            
            # 绘制所有通道
            fig, ax = plt.subplots(figsize=figsize)
            
            waveforms_dict = {}
            colors = plt.cm.tab10(np.linspace(0, 1, len(channel_indices)))
            
            for i, ch_idx in enumerate(channel_indices):
                waveform = channel_data[:, ch_idx, event_idx]
                waveforms_dict[ch_idx] = (time_axis, waveform)
                ax.plot(time_axis, waveform, label=f'Channel #{ch_idx}', 
                       linewidth=1.0, color=colors[i], alpha=1.0)
            
            ax.set_xlabel(f'Time ({time_unit})', fontsize=12)
            ax.set_ylabel('Amplitude (ADC counts)', fontsize=12)
            ax.set_title(f'Multi-Channel Waveform Comparison\nFile: {os.path.basename(h5_file)}\nEvent #{event_idx}, Event Time: {event_time:.6f} s',
                        fontsize=11)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f'图片已保存至: {save_path}')
            
            if show_plot:
                plt.show()
            else:
                plt.close()
            
            return waveforms_dict
    
    except Exception as e:
        print(f'可视化多通道波形时出错: {e}')
        raise

def list_all_h5_files(base_dir: str = None) -> None:
    """
    列出 raw_pulse 目录下所有 h5 文件
    
    参数:
        base_dir: raw_pulse 目录路径，如果为None则自动查找项目根目录
    """
    h5_files = get_h5_files(base_dir)
    
    print('=' * 70)
    print('HDF5 文件列表')
    print('=' * 70)
    
    if not h5_files:
        print('未找到任何 h5 文件')
        return
    
    total_files = 0
    for subdir, files in h5_files.items():
        print(f'\n目录: {subdir}')
        print('-' * 70)
        for i, file_path in enumerate(files, 1):
            file_size = os.path.getsize(file_path) / (1024 ** 3)
            print(f'  [{i}] {os.path.basename(file_path)}')
            print(f'      大小: {file_size:.4f} GB')
            print(f'      路径: {file_path}')
        total_files += len(files)
        print(f'  共 {len(files)} 个文件')
    
    print('\n' + '=' * 70)
    print(f'总计: {total_files} 个 h5 文件')

# 示例使用
if __name__ == '__main__':

    # 列出所有文件
    print('列出所有 h5 文件:')
    list_all_h5_files()
    # 获取文件列表
    h5_files = get_h5_files()
    # 如果有文件，显示第一个文件的结构
    if h5_files:
        first_subdir = list(h5_files.keys())[0]
        first_file = h5_files[first_subdir][0]
        print(f'\n\n显示文件结构: {first_file}')
        show_h5_structure(first_file)
        # 可视化第一个事件的第一个通道
        print(f'\n\n可视化单个波形:')
        #visualize_waveform(first_file, event_idx=0, channel_idx=0, time_unit='us', show_plot=True)
        
        # 可视化多个通道的波形（同一事件）
        print(f'\n\n可视化多通道波形:')
        # 根据文件所在的目录决定显示哪些通道
        if first_subdir == 'CH0-3':
            # CH0-3 目录包含4个通道（索引0-3）
            channel_indices = [0, 1, 2, 3]
        elif first_subdir == 'CH5':
            # CH5 目录只有1个通道（索引0，对应原始通道5）
            channel_indices = [0]
        else:
            # 默认显示所有通道
            channel_indices = None
        
        visualize_multiple_channels(first_file, 
                                   event_idx=1, 
                                   channel_indices=channel_indices,
                                   time_unit='us', 
                                   show_plot=True)

    '''
    # 查看某个文件的波形
    # 获取 CH5 目录中的 h5 文件
    h5_files = get_h5_files()
    if 'CH5' not in h5_files or not h5_files['CH5']:
        print('错误: 在 data/hdf5/raw_pulse/CH5 目录中未找到 h5 文件')
    else:
        # 使用 CH5 目录中的第一个文件
        ch5_file = h5_files['CH5'][0]

        print('=' * 70)
        print('查看 CH5 目录中的波形数据')
        print('=' * 70)
        
        # 显示文件结构
        print(f'\n显示文件结构:')
        show_h5_structure(ch5_file)
        
        # 可视化波形 - 可以选择不同的事件
        event_idx = 3       # 要查看的事件索引，可以修改
        channel_idx = 0     # CH5 目录只有一个通道（索引0）
        time_unit = 'us'    # 时间单位：'ns', 'us', 'ms', 's'
        
        print(f'\n可视化波形 (事件 #{event_idx}, 通道 #{channel_idx}):')
        visualize_waveform(ch5_file, 
                          event_idx=event_idx, 
                          channel_idx=channel_idx, 
                          time_unit=time_unit, 
                          show_plot=True)
    '''