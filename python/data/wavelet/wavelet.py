#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对所有event的CH0信号拼接（event之间补0），然后进行时频分析
"""
import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt
from typing import Optional, Tuple, Dict
from joblib import Parallel, delayed
import multiprocessing as mp

# 添加路径以便导入 utils 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
python_dir = os.path.dirname(os.path.dirname(current_dir))
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from utils.visualize import get_h5_files

# 模块级函数，用于并行处理（可被pickle）
def _process_event_batch_worker(args):
    """批量处理多个events（模块级函数，可被pickle）
    接收已读取的数据，避免重复打开文件
    """
    event_indices, batch_waveforms, batch_trigger_times, time_samples, sampling_interval_s, detrend = args
    
    num_events_batch = len(event_indices)
    # 预分配数组（避免频繁extend）
    batch_signals = np.zeros(num_events_batch * time_samples, dtype=np.float64)
    batch_times = np.zeros(num_events_batch * time_samples, dtype=np.float64)
    
    # 批量处理（向量化操作）
    time_offsets = np.arange(time_samples) * sampling_interval_s  # shape: (time_samples,)
    
    for i, (waveform, trigger_time) in enumerate(zip(batch_waveforms, batch_trigger_times)):
        # 去趋势
        if detrend:
            waveform = signal.detrend(waveform)
        
        # 填充到预分配的数组
        start_idx = i * time_samples
        end_idx = start_idx + time_samples
        batch_signals[start_idx:end_idx] = waveform
        batch_times[start_idx:end_idx] = trigger_time + time_offsets
    
    return batch_signals.tolist(), batch_times.tolist()

def _process_slice_wavelet_worker(args):
    """
    处理单个切片的worker函数（用于并行处理，可被pickle）
    
    参数:
        args: 元组，包含 (slice_idx, slice_signal_values, slice_time_stamps,
              slice_start_time, slice_end_time, uniform_sampling_rate,
              sampling_interval_s, wavename, totalscal, min_freq, max_freq,
              detrend, slices_to_process)
    
    返回:
        包含分析结果的字典
    """
    (slice_idx, slice_signal_values, slice_time_stamps,
     slice_start_time, slice_end_time, uniform_sampling_rate,
     sampling_interval_s, wavename, totalscal, min_freq, max_freq,
     detrend, slices_to_process) = args
    
    # 调用原有的处理函数（不显示图）
    return _process_slice_wavelet(
        slice_signal_values, slice_time_stamps,
        slice_start_time, slice_end_time,
        uniform_sampling_rate, sampling_interval_s,
        wavename, totalscal, min_freq, max_freq,
        detrend, None, False, slice_idx, slices_to_process
    )

def _process_slice_wavelet(slice_signal_values, slice_time_stamps,
                           slice_start_time, slice_end_time,
                           uniform_sampling_rate, sampling_interval_s,
                           wavename, totalscal, min_freq, max_freq,
                           detrend, save_path, show_plot, slice_idx, num_slices):
    """
    处理单个时间切片：降采样 + 小波变换
    
    参数:
        slice_signal_values: 切片内的信号值
        slice_time_stamps: 切片内的时间戳
        slice_start_time: 切片开始时间
        slice_end_time: 切片结束时间
        uniform_sampling_rate: 均匀采样率
        sampling_interval_s: 原始采样间隔
        wavename: 小波名称
        totalscal: 总尺度数
        min_freq: 最小频率
        max_freq: 最大频率
        detrend: 是否去趋势
        save_path: 保存路径（如果提供，会添加切片索引）
        show_plot: 是否显示图片
        slice_idx: 切片索引
        num_slices: 总切片数
    
    返回:
        包含分析结果的字典
    """
    # 3.1. 构建降采样后的信号（event之间补0）
    slice_duration = slice_end_time - slice_start_time
    uniform_dt = 1.0 / uniform_sampling_rate
    num_uniform_points = int(slice_duration * uniform_sampling_rate) + 1
    actual_sampling_rate = 1.0 / uniform_dt
    
    print(f'  降采样: {len(slice_signal_values)} 点 -> {num_uniform_points} 点 (采样率: {actual_sampling_rate:.1f}Hz)')
    
    # 创建降采样后的时间网格
    uniform_times = slice_start_time + np.arange(num_uniform_points) * uniform_dt
    uniform_signal = np.zeros(num_uniform_points, dtype=np.float64)
    
    # 计算容差：原始采样间隔的一半
    tolerance = sampling_interval_s / 2.0
    
    # 使用searchsorted快速查找（向量化操作）
    uniform_indices = np.searchsorted(uniform_times, slice_time_stamps, side='left')
    
    # 过滤掉越界的索引
    valid_mask = (uniform_indices < num_uniform_points)
    
    if not np.all(valid_mask):
        print(f'  警告: {np.sum(~valid_mask)} 个时间戳超出范围，将被跳过')
        uniform_indices = uniform_indices[valid_mask]
        slice_time_stamps_valid = slice_time_stamps[valid_mask]
        slice_signal_values_valid = slice_signal_values[valid_mask]
    else:
        slice_time_stamps_valid = slice_time_stamps
        slice_signal_values_valid = slice_signal_values
    
    # 向量化映射
    mask1 = (uniform_indices < num_uniform_points) & \
            (np.abs(uniform_times[uniform_indices] - slice_time_stamps_valid) < tolerance)
    uniform_signal[uniform_indices[mask1]] = slice_signal_values_valid[mask1]
    
    # 处理前一个索引位置
    mask2 = (uniform_indices > 0) & (~mask1) & \
            (uniform_indices - 1 < num_uniform_points) & \
            (np.abs(uniform_times[uniform_indices - 1] - slice_time_stamps_valid) < tolerance)
    uniform_signal[uniform_indices[mask2] - 1] = slice_signal_values_valid[mask2]
    
    print(f'  映射完成: {np.sum(mask1 | mask2)}/{len(slice_time_stamps_valid)} 个点成功映射')
    print(f'  信号点占比: {np.sum(uniform_signal != 0) / num_uniform_points * 100:.2f}%')
    
    # 去趋势
    if detrend:
        uniform_signal = signal.detrend(uniform_signal)
    
    # 4. 小波变换参数设置
    if wavename.startswith('cmor'):
        params_str = wavename[4:]
        parts = params_str.split('-')
        if len(parts) == 2:
            bandwidth_param = float(parts[0])
            center_freq_param = float(parts[1])
            Fc = center_freq_param / (2 * np.pi)
        else:
            Fc = 1.0
    else:
        try:
            Fc = pywt.central_frequency(wavename)
        except:
            Fc = 1.0
    
    # 计算尺度范围（基于降采样后的采样率）
    max_scale = (Fc * actual_sampling_rate) / min_freq
    min_scale = (Fc * actual_sampling_rate) / max_freq
    scales = np.logspace(np.log10(min_scale), np.log10(max_scale), totalscal)
    frequencies = (Fc * actual_sampling_rate) / scales
    
    print(f'  小波变换: {totalscal} 个尺度, 频率范围 {frequencies[-1]:.3f} - {frequencies[0]:.3f} Hz')
    
    # 5. 进行小波变换
    print(f'  计算小波变换...')
    coefficients, _ = pywt.cwt(uniform_signal, scales, wavename, sampling_period=uniform_dt)
    
    # 计算功率
    if np.iscomplexobj(coefficients):
        power = np.abs(coefficients) ** 2
    else:
        power = coefficients ** 2
    
    mean_power_per_scale = np.mean(power, axis=1)
    dominant_freq = frequencies[np.argmax(mean_power_per_scale)]
    
    print(f'  小波变换完成: 功率矩阵形状 {power.shape}, 主导频率 {dominant_freq:.4f}Hz')
    
    # 6. 可视化
    if show_plot or save_path:
        # 限制显示的频率范围
        freq_mask = (frequencies >= min_freq) & (frequencies <= max_freq)
        frequencies_display = frequencies[freq_mask]
        power_display = power[freq_mask, :]
        
        # 相对时间（从切片开始时间起）
        time_display = uniform_times - slice_start_time
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # 使用对数显示功率
        log_power = np.log10(power_display + 1e-20)
        
        im = ax.contourf(time_display, frequencies_display, log_power, 
                        levels=50, cmap='jet')
        ax.set_xlabel('Time (s, relative to slice start)', fontsize=12)
        ax.set_ylabel('Frequency (Hz)', fontsize=12)
        ax.set_title(f'Wavelet Time-Frequency Analysis (Slice {slice_idx + 1}/{num_slices})\n'
                    f'Absolute Time: {slice_start_time:.1f}s - {slice_end_time:.1f}s, '
                    f'Frequency: {min_freq:.1f}Hz - {max_freq:.1f}Hz',
                    fontsize=13)
        
        plt.colorbar(im, ax=ax, label='Log10(Power)')
        plt.tight_layout()
        
        if save_path:
            # 在保存路径中添加切片索引
            base, ext = os.path.splitext(save_path)
            slice_save_path = f'{base}_slice{slice_idx + 1:03d}{ext}'
            plt.savefig(slice_save_path, dpi=150, bbox_inches='tight')
            print(f'  图片已保存: {slice_save_path}')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    return {
        'slice_idx': slice_idx,
        'slice_start_time': slice_start_time,
        'slice_end_time': slice_end_time,
        'uniform_times': uniform_times,
        'uniform_signal': uniform_signal,
        'scales': scales,
        'frequencies': frequencies,
        'power': power,
        'dominant_freq': dominant_freq,
        'num_points': len(slice_signal_values),
        'num_uniform_points': num_uniform_points
    }

def analyze_all_events_wavelet(h5_file: str = None,
                                channel_idx: int = 0,
                                detrend: bool = True,
                                uniform_sampling_rate: float = 1000.0,
                                wavename: str = 'cmor1.5-1',
                                totalscal: int = 128,
                                min_freq: float = 0.1,
                                max_freq: float = 200.0,
                                slice_duration: Optional[float] = None,  # 每个切片的时间长度（秒），None表示不切片
                                num_slices_to_average: Optional[int] = None,  # 平均多少个切片的结果，None表示平均所有切片，正整数表示只平均前N个切片
                                save_path: Optional[str] = None,
                                show_plot: bool = True) -> Dict:
    """
    拼接所有event的CH0信号（event之间补0），然后进行时频分析
    参数:
        h5_file: HDF5文件路径，如果为None则自动获取CH0-3目录中的第一个文件
        channel_idx: 通道索引，默认0表示CH0通道
        detrend: 是否去趋势
        uniform_sampling_rate: 均匀采样率（Hz），用于降采样
        wavename: 小波名称
        totalscal: 总尺度数
        min_freq: 最小频率（Hz）
        max_freq: 最大频率（Hz）
        slice_duration: 每个切片的时间长度（秒），None表示不切片，直接处理整个信号
        num_slices_to_average: 平均多少个切片的结果，None表示平均所有切片，正整数表示只平均前N个切片。例如设置为10表示只平均前10个切片
        save_path: 保存图片路径
        show_plot: 是否显示图片
    
    返回:
        包含分析结果的字典
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
    print(f'对所有event的CH0信号进行拼接和时频分析')
    print(f'文件: {os.path.basename(h5_file)}')
    print('=' * 70)
    
    # 1. 参数设置
    sampling_interval_ns = 4.0
    sampling_interval_s = sampling_interval_ns * 1e-9
    original_sampling_rate = 1.0 / sampling_interval_s  # 250 MSPS
    
    print(f'\n信号参数:')
    print(f'  采样间隔: {sampling_interval_ns} ns')
    print(f'  原始采样率: {original_sampling_rate/1e6:.2f} MSPS')
    
    # 2. 读取文件并拼接所有event的CH0信号
    print(f'\n正在读取文件并拼接所有event的CH0信号...')
    
    # 在主进程中一次性读取所有数据（避免每个进程重复打开文件）
    print(f'  在主进程中一次性读取所有数据...')
    with h5py.File(h5_file, 'r') as f:
        channel_data = f['channel_data']
        time_data = f['time_data']  # 每个event的触发时间
        time_samples, num_channels, num_events = channel_data.shape
        
        print(f'  数据维度: (时间点={time_samples}, 通道数={num_channels}, 事件数={num_events})')
        
        if channel_idx < 0 or channel_idx >= num_channels:
            raise IndexError(f'通道索引 {channel_idx} 超出范围 [0, {num_channels-1}]')
        
        # 批量读取所有CH0数据（一次性读取，利用HDF5批量读取优势）
        print(f'  批量读取所有events的CH0数据...')
        all_waveforms = channel_data[:, channel_idx, :].astype(np.float64)  # shape: (time_samples, num_events)
        all_waveforms = all_waveforms.T  # shape: (num_events, time_samples) - 转置以便按event访问
        all_trigger_times = time_data[:]  # shape: (num_events,)
        print(f'  数据读取完成: {all_waveforms.shape}')
    
    # 批处理设置：减少任务数量，提高效率
    n_jobs = mp.cpu_count()  # 使用所有CPU核心
    batch_size = max(50, num_events // (n_jobs * 4))  # 每个核心4个批次
    event_batches = [list(range(i, min(i + batch_size, num_events)))
                    for i in range(0, num_events, batch_size)]

    print(f'  批处理模式: {len(event_batches)} 个批次, 每批 {batch_size} 个events')
    print(f'  使用进程池并行处理 ({n_jobs} 个进程，使用所有CPU核心)...')

    # 准备参数（传递已读取的数据，而不是文件路径）
    batch_args = []
    for batch in event_batches:
        # 批量提取该批次的数据（避免逐个访问）
        batch_waveforms = all_waveforms[batch, :]  # shape: (batch_size, time_samples)
        batch_trigger_times = all_trigger_times[batch]  # shape: (batch_size,)
        batch_args.append((batch, batch_waveforms, batch_trigger_times, 
                          time_samples, sampling_interval_s, detrend))

    # 使用joblib的loky后端（使用模块级函数，可被pickle）
    results = Parallel(n_jobs=n_jobs, backend='loky', verbose=1)(
        delayed(_process_event_batch_worker)(args) for args in batch_args
    )

    # 合并结果（预分配数组，避免频繁extend）
    print(f'  合并结果...')
    total_points = num_events * time_samples
    all_signal_values = np.zeros(total_points, dtype=np.float64)
    all_time_stamps = np.zeros(total_points, dtype=np.float64)
    
    current_idx = 0
    for batch_signals, batch_times in results:
        batch_len = len(batch_signals)
        all_signal_values[current_idx:current_idx + batch_len] = batch_signals
        all_time_stamps[current_idx:current_idx + batch_len] = batch_times
        current_idx += batch_len

    print(f'  批处理完成')
    
    # 排序（按时间）
    print(f'  排序数据...')
    sort_indices = np.argsort(all_time_stamps)
    all_signal_values = all_signal_values[sort_indices]
    all_time_stamps = all_time_stamps[sort_indices]
    total_duration = all_time_stamps[-1] - all_time_stamps[0]
    start_time = all_time_stamps[0]
    end_time = all_time_stamps[-1]
    
    print(f'\n拼接完成:')
    print(f'  总数据点数: {len(all_signal_values)}')
    print(f'  时间跨度: {total_duration:.6f} s ({total_duration/60:.2f} 分钟)')
    print(f'  起始时间: {start_time:.6f} s')
    print(f'  结束时间: {end_time:.6f} s')
    
    # 3. 时间切片划分（如果设置了slice_duration）
    if slice_duration is not None and slice_duration > 0:
        num_slices = int(np.ceil(total_duration / slice_duration))
        print(f'\n时间切片划分:')
        print(f'  切片时长: {slice_duration:.1f} 秒')
        print(f'  切片数量: {num_slices}')
        
        # 确定需要处理的切片数量
        if num_slices_to_average is not None and num_slices_to_average > 0:
            slices_to_process = min(num_slices_to_average, num_slices)
            print(f'  将处理前 {slices_to_process} 个切片进行平均（总共 {num_slices} 个切片）')
        else:
            slices_to_process = num_slices
            print(f'  将处理所有 {slices_to_process} 个切片')
        
        # 准备所有切片的数据（在主进程中完成，避免并行时重复计算）
        print(f'\n准备 {slices_to_process} 个切片的数据...')
        slice_args_list = []
        
        for slice_idx in range(slices_to_process):
            slice_start_time = start_time + slice_idx * slice_duration
            slice_end_time = min(start_time + (slice_idx + 1) * slice_duration, end_time)
            
            # 提取当前切片内的数据点
            slice_mask = (all_time_stamps >= slice_start_time) & (all_time_stamps < slice_end_time)
            slice_signal_values = all_signal_values[slice_mask]
            slice_time_stamps = all_time_stamps[slice_mask]
            
            if len(slice_signal_values) == 0:
                print(f'  警告: 切片 {slice_idx + 1} 没有数据点，跳过')
                continue
            
            # 准备参数（所有数据都作为参数传递，可被pickle）
            slice_args = (
                slice_idx, slice_signal_values, slice_time_stamps,
                slice_start_time, slice_end_time, uniform_sampling_rate,
                sampling_interval_s, wavename, totalscal, min_freq, max_freq,
                detrend, slices_to_process
            )
            slice_args_list.append(slice_args)
        
        print(f'  准备完成: {len(slice_args_list)} 个有效切片')
        
        # 并行处理所有切片
        n_jobs = mp.cpu_count()
        if len(slice_args_list) > 1:
            print(f'\n使用 {n_jobs} 个CPU核心并行处理 {len(slice_args_list)} 个切片...')
            slice_results = Parallel(n_jobs=n_jobs, backend='loky', verbose=1)(
                delayed(_process_slice_wavelet_worker)(args) for args in slice_args_list
            )
        else:
            # 只有一个切片，直接处理
            print(f'\n处理单个切片...')
            slice_results = [_process_slice_wavelet_worker(slice_args_list[0])]
        
        # 按slice_idx排序（并行处理可能乱序）
        slice_results.sort(key=lambda x: x['slice_idx'])
        
        print(f'\n{"="*60}')
        if num_slices_to_average is not None and slices_to_process < num_slices:
            print(f'前 {slices_to_process} 个切片处理完成!（总共 {num_slices} 个切片，剩余 {num_slices - slices_to_process} 个未处理）')
        else:
            print(f'所有切片处理完成!')
        
        # 4. 对多个切片进行平均统计（减小方差）
        if len(slice_results) > 0:
            print(f'\n{"="*60}')
            print(f'对 {len(slice_results)} 个切片进行平均统计...')
            slices_to_avg = slice_results
            
            # 获取第一个切片的信息作为参考
            first_result = slice_results[0]
            frequencies = first_result['frequencies']
            scales = first_result['scales']
            
            if len(slices_to_avg) == 0:
                print(f'  警告: 没有切片可用于平均统计')
            else:
                # 获取每个切片的功率谱
                slice_powers = []
                slice_time_lengths = []
                
                for result in slices_to_avg:
                    power = result['power']  # shape: (totalscal, num_uniform_points)
                    slice_powers.append(power)
                    slice_time_lengths.append(power.shape[1])
            
            # 找到最短的时间长度（用于对齐）
            min_time_length = min(slice_time_lengths)
            print(f'  最短切片长度: {min_time_length} 点')
            print(f'  切片长度范围: {min(slice_time_lengths)} - {max(slice_time_lengths)} 点')
            
            # 对齐所有切片到相同的时间长度（截取前min_time_length个点）
            aligned_powers = []
            for power in slice_powers:
                if power.shape[1] >= min_time_length:
                    aligned_powers.append(power[:, :min_time_length])
                else:
                    # 如果切片太短，跳过
                    print(f'  警告: 某个切片长度 ({power.shape[1]}) 小于最小长度，跳过')
            
            if len(aligned_powers) > 0:
                # 转换为numpy数组: shape (num_slices, totalscal, time_points)
                all_power_array = np.array(aligned_powers)
                
                print(f'  功率谱数组形状: {all_power_array.shape} (切片数, 尺度数, 时间点数)')
                
                # 直接平均所有选中的切片
                mean_power = np.mean(all_power_array, axis=0)  # shape: (totalscal, time_points)
                std_power = np.std(all_power_array, axis=0)    # 标准差
                median_power = np.median(all_power_array, axis=0)  # 中位数
                
                # 计算每个频率的平均功率（跨时间维度平均）
                mean_power_per_freq = np.mean(mean_power, axis=1)  # shape: (totalscal,)
                std_power_per_freq = np.mean(std_power, axis=1)
                
                # 计算主导频率
                dominant_freq_idx = np.argmax(mean_power_per_freq)
                dominant_freq = frequencies[dominant_freq_idx]
                
                print(f'\n平均功率谱统计:')
                if num_slices_to_average is None:
                    print(f'  有效切片数: {len(aligned_powers)}/{len(slice_results)}')
                else:
                    print(f'  有效切片数: {len(aligned_powers)}/{num_slices_to_average} (从总共 {len(slice_results)} 个切片中)')
                print(f'  平均功率范围: {np.min(mean_power):.2e} - {np.max(mean_power):.2e}')
                print(f'  平均功率 (整体): {np.mean(mean_power):.2e}')
                print(f'  主导频率: {dominant_freq:.4f} Hz')
                print(f'  主导频率平均功率: {mean_power_per_freq[dominant_freq_idx]:.2e}')
                
                # 可视化平均结果
                if show_plot or save_path:
                    # 限制显示的频率范围
                    freq_mask = (frequencies >= min_freq) & (frequencies <= max_freq)
                    frequencies_display = frequencies[freq_mask]
                    mean_power_display = mean_power[freq_mask, :]
                    std_power_display = std_power[freq_mask, :]
                    
                    # 创建相对时间轴（基于最短切片）
                    uniform_dt = 1.0 / uniform_sampling_rate
                    time_display = np.arange(min_time_length) * uniform_dt
                    
                    # 创建两个子图：时频图和平均频率谱
                    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
                    
                    # 子图1：平均时频图
                    ax1 = axes[0]
                    log_mean_power = np.log10(mean_power_display + 1e-20)
                    im1 = ax1.contourf(time_display, frequencies_display, log_mean_power, 
                                     levels=50, cmap='jet')
                    ax1.set_xlabel('Time (s, relative to slice start)', fontsize=12)
                    ax1.set_ylabel('Frequency (Hz)', fontsize=12)
                    ax1.set_title(f'Averaged Wavelet Time-Frequency Analysis ({len(aligned_powers)} slices)\n'
                                f'Frequency: {min_freq:.1f}Hz - {max_freq:.1f}Hz, '
                                f'Dominant Freq: {dominant_freq:.4f}Hz',
                                fontsize=13)
                    ax1.set_ylim(min_freq, max_freq)
                    plt.colorbar(im1, ax=ax1, label='Log10(Mean Power)')
                    ax1.grid(True, alpha=0.3)
                    
                    # 子图2：平均频率谱（跨时间平均）
                    ax2 = axes[1]
                    ax2.plot(frequencies_display, mean_power_per_freq[freq_mask], 
                            'b-', linewidth=2, label='Mean Power')
                    ax2.fill_between(frequencies_display, 
                                    mean_power_per_freq[freq_mask] - std_power_per_freq[freq_mask],
                                    mean_power_per_freq[freq_mask] + std_power_per_freq[freq_mask],
                                    alpha=0.3, color='blue', label='±1 Std')
                    ax2.axvline(dominant_freq, color='r', linestyle='--', linewidth=2, 
                               label=f'Dominant: {dominant_freq:.4f}Hz')
                    ax2.set_xlabel('Frequency (Hz)', fontsize=12)
                    ax2.set_ylabel('Average Power', fontsize=12)
                    if num_slices_to_average is None:
                        title_suffix = f' ({len(aligned_powers)} slices averaged)'
                    else:
                        title_suffix = f' (first {len(aligned_powers)}/{len(slice_results)} slices averaged)'
                    ax2.set_title(f'Averaged Frequency Spectrum{title_suffix}',
                                fontsize=13)
                    ax2.set_xlim(min_freq, max_freq)
                    ax2.set_yscale('log')
                    ax2.legend(fontsize=11)
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    
                    if save_path:
                        # 保存平均结果图
                        base, ext = os.path.splitext(save_path)
                        avg_save_path = f'{base}_averaged{ext}'
                        plt.savefig(avg_save_path, dpi=150, bbox_inches='tight')
                        print(f'\n平均结果图片已保存: {avg_save_path}')
                    
                    if show_plot:
                        plt.show()
                    else:
                        plt.close()
                
                # 在返回结果中添加平均统计
                return {
                    'num_events': num_events,
                    'num_slices': num_slices,
                    'num_valid_slices': len(aligned_powers),
                    'slice_duration': slice_duration,
                    'slice_results': slice_results,
                    'all_signal_values': all_signal_values,
                    'all_time_stamps': all_time_stamps,
                    # 平均统计结果
                    'averaged': {
                        'mean_power': mean_power,
                        'std_power': std_power,
                        'median_power': median_power,
                        'mean_power_per_freq': mean_power_per_freq,
                        'std_power_per_freq': std_power_per_freq,
                        'dominant_freq': dominant_freq,
                        'frequencies': frequencies,
                        'scales': scales,
                        'time_points': min_time_length
                    }
                }
            else:
                print(f'  警告: 没有有效的切片用于平均统计')
        
        # 返回所有切片的结果
        return {
            'num_events': num_events,
            'num_slices': num_slices,
            'slice_duration': slice_duration,
            'slice_results': slice_results,
            'all_signal_values': all_signal_values,
            'all_time_stamps': all_time_stamps
        }
    
    # 如果没有切片，按原来的方式处理整个信号
    # 3. 检查数据规模，决定处理策略
    uniform_dt = 1.0 / uniform_sampling_rate
    num_uniform_points = int(total_duration * uniform_sampling_rate) + 1
    actual_sampling_rate = 1.0 / uniform_dt
    
    print(f'\n数据规模评估:')
    print(f'  如果创建均匀网格点数: {num_uniform_points}')
    
    # 构建降采样后的信号（event之间补0）
    start_time = all_time_stamps[0]
    end_time = all_time_stamps[-1]
    
    # 创建降采样后的时间网格
    uniform_times = start_time + np.arange(num_uniform_points) * uniform_dt
    uniform_signal = np.zeros(num_uniform_points, dtype=np.float64)
    
    # 计算容差：原始采样间隔的一半
    tolerance = sampling_interval_s / 2.0
    
    # 使用searchsorted快速查找（向量化操作）
    print(f'  映射数据点到网格...')
    print(f'  使用searchsorted查找映射位置...')
    uniform_indices = np.searchsorted(uniform_times, all_time_stamps, side='left')
    
    # 过滤掉越界的索引（searchsorted可能返回num_uniform_points）
    valid_mask = (uniform_indices < num_uniform_points)
    
    if not np.all(valid_mask):
        print(f'  警告: {np.sum(~valid_mask)} 个时间戳超出范围，将被跳过')
        uniform_indices = uniform_indices[valid_mask]
        all_time_stamps_valid = all_time_stamps[valid_mask]
        all_signal_values_valid = all_signal_values[valid_mask]
    else:
        all_time_stamps_valid = all_time_stamps
        all_signal_values_valid = all_signal_values
    
    # 向量化映射（避免循环和append）
    print(f'  向量化映射（避免并行开销）...')
    # 处理当前索引位置
    mask1 = (uniform_indices < num_uniform_points) & \
            (np.abs(uniform_times[uniform_indices] - all_time_stamps_valid) < tolerance)
    uniform_signal[uniform_indices[mask1]] = all_signal_values_valid[mask1]
    
    # 处理前一个索引位置（如果当前索引不匹配）
    mask2 = (uniform_indices > 0) & (~mask1) & \
            (uniform_indices - 1 < num_uniform_points) & \
            (np.abs(uniform_times[uniform_indices - 1] - all_time_stamps_valid) < tolerance)
    uniform_signal[uniform_indices[mask2] - 1] = all_signal_values_valid[mask2]
    
    print(f'  映射完成: {np.sum(mask1 | mask2)}/{len(all_time_stamps_valid)} 个点成功映射')
    print(f'\n降采样完成:')
    print(f'  均匀网格点数: {num_uniform_points}')
    print(f'  采样率: {actual_sampling_rate:.1f}Hz')
    print(f'  信号点占比: {np.sum(uniform_signal != 0) / num_uniform_points * 100:.2f}%')
    
    # 4. 小波变换参数设置
    if wavename.startswith('cmor'):
        params_str = wavename[4:]
        parts = params_str.split('-')
        if len(parts) == 2:
            bandwidth_param = float(parts[0])
            center_freq_param = float(parts[1])
            Fc = center_freq_param / (2 * np.pi)
        else:
            Fc = 1.0
    else:
        try:
            Fc = pywt.central_frequency(wavename)
        except:
            Fc = 1.0
    
    print(f'\n小波参数:')
    print(f'  小波类型: {wavename}')
    print(f'  小波中心频率: {Fc:.4f}')
    print(f'  总尺度数: {totalscal}')
    
    # 计算尺度范围（基于降采样后的采样率）
    max_scale = (Fc * actual_sampling_rate) / min_freq
    min_scale = (Fc * actual_sampling_rate) / max_freq
    scales = np.logspace(np.log10(min_scale), np.log10(max_scale), totalscal)
    frequencies = (Fc * actual_sampling_rate) / scales
    
    print(f'  尺度范围: {scales[0]:.2f} - {scales[-1]:.2f}')
    print(f'  对应频率范围: {frequencies[-1]:.3f} - {frequencies[0]:.3f} Hz')
    
    # 5. 分段小波变换（避免内存爆炸）
    # 估算内存需求：每个尺度需要存储 num_uniform_points 个复数
    # complex128 = 16字节，每个尺度需要 num_uniform_points * 16 字节
    # 为了节省内存，我们直接计算功率（float64 = 8字节），而不是存储复数系数
    max_points_per_segment = 5_000_000  # 每段最多500万个点（减少内存使用）
    
    if num_uniform_points > max_points_per_segment:
        # 需要分段处理
        segment_size = max_points_per_segment
        num_segments = int(np.ceil(num_uniform_points / segment_size))
        
        print(f'\n信号太长（{num_uniform_points} 个点），使用分段处理策略')
        print(f'  分段数: {num_segments}')
        print(f'  每段点数: {segment_size}')
        print(f'  直接计算功率以节省内存（不存储复数系数）')
        
        # 直接为功率分配数组（float64，比复数节省一半内存）
        power = np.zeros((totalscal, num_uniform_points), dtype=np.float64)
        
        # 计算每个尺度所需的重叠长度（基于最大尺度）
        # 小波变换在边界处需要一些重叠以避免边界效应
        max_overlap = int(max_scale * 2)  # 基于最大尺度的重叠长度
        
        for seg_idx in range(num_segments):
            start_idx = seg_idx * segment_size
            end_idx = min((seg_idx + 1) * segment_size + max_overlap, num_uniform_points)
            segment_signal = uniform_signal[start_idx:end_idx]
            
            if len(segment_signal) == 0:
                continue
            
            print(f'  处理段 {seg_idx + 1}/{num_segments} (索引 {start_idx}:{end_idx}, {len(segment_signal)} 点)...')
            
            # 对这段信号进行小波变换
            segment_coeff, _ = pywt.cwt(segment_signal, scales, wavename, sampling_period=uniform_dt)
            
            # 计算功率（如果是复数）
            if np.iscomplexobj(segment_coeff):
                segment_power = np.abs(segment_coeff) ** 2
            else:
                segment_power = segment_coeff ** 2
            
            # 提取有效部分（去除重叠区域，但保留边界区域以避免边界效应）
            if seg_idx == 0:
                # 第一段：保留全部
                valid_start = 0
                valid_end = min(segment_size, len(segment_signal))
            elif seg_idx == num_segments - 1:
                # 最后一段：从重叠处开始
                valid_start = max_overlap
                valid_end = len(segment_signal)
            else:
                # 中间段：去除两端重叠
                valid_start = max_overlap // 2
                valid_end = min(segment_size + max_overlap // 2, len(segment_signal))
            
            # 映射回全局索引
            global_start = start_idx + valid_start
            global_end = start_idx + valid_end
            segment_local_start = valid_start
            segment_local_end = valid_end
            
            # 存储功率（而不是复数系数）
            power[:, global_start:global_end] = segment_power[:, segment_local_start:segment_local_end]
            
            # 释放内存
            del segment_coeff, segment_power
            
        print(f'  分段小波变换完成')
        # 为了兼容后续代码，创建一个虚拟的coefficients（实际上我们只有power）
        coefficients = None
    else:
        # 信号足够短，可以直接处理
        print(f'\n进行小波变换...')
        coefficients, _ = pywt.cwt(uniform_signal, scales, wavename, sampling_period=uniform_dt)
        
        # 计算功率
        if np.iscomplexobj(coefficients):
            power = np.abs(coefficients) ** 2
        else:
            power = coefficients ** 2
    
    mean_power_per_scale = np.mean(power, axis=1)
    dominant_freq = frequencies[np.argmax(mean_power_per_scale)]
    
    print(f'\n小波变换完成')
    print(f'  系数矩阵形状: {power.shape}')
    print(f'  主导频率: {dominant_freq:.4f}Hz')
    
    # 6. 可视化
    if show_plot or save_path:
        # 限制显示的频率范围
        freq_mask = (frequencies >= min_freq) & (frequencies <= max_freq)
        frequencies_display = frequencies[freq_mask]
        power_display = power[freq_mask, :]
        
        # 限制显示的时间范围（前20秒）
        max_time_seconds = 20.0
        max_time_points = int(max_time_seconds * actual_sampling_rate) + 1
        max_time_points = min(max_time_points, len(uniform_times))
        
        time_display = uniform_times[:max_time_points] - uniform_times[0]
        power_display = power_display[:, :max_time_points]
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # 使用对数显示功率
        log_power = np.log10(power_display + 1e-20)
        
        im = ax.contourf(time_display, frequencies_display, log_power, 
                        levels=50, cmap='jet')
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Frequency (Hz)', fontsize=12)
        ax.set_title(f'Wavelet Time-Frequency Analysis (All Events)\n'
                    f'{num_events} events, {len(all_signal_values)} points, '
                    f'Frequency: {min_freq:.1f}Hz - {max_freq:.1f}Hz, Time: ≤{max_time_seconds}s',
                    fontsize=13)
        ax.set_ylim(min_freq, max_freq)
        plt.colorbar(im, ax=ax, label='Log10(Power)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'\n图片已保存: {save_path}')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    return {
        'num_events': num_events,
        'all_signal_values': all_signal_values,
        'all_time_stamps': all_time_stamps,
        'uniform_times': uniform_times,
        'uniform_signal': uniform_signal,
        'scales': scales,
        'frequencies': frequencies,
        'power': power,
        'dominant_freq': dominant_freq
    }

if __name__ == '__main__':
    try:
        results = analyze_all_events_wavelet(
            h5_file=None,  # 自动选择CH0-3目录中的第一个文件
            channel_idx=0,  # CH0通道
            detrend=True,
            uniform_sampling_rate=1000.0,  # 1kHz采样率
            wavename='cmor1.5-1',
            totalscal=128,
            min_freq=0.1,  # 0.1 Hz
            max_freq=200.0,  # 200 Hz
            slice_duration=10,  # 每个切片3600秒（1小时），None表示不切片
            num_slices_to_average=10000,  # 平均前10个切片
            show_plot=True
        )
        
        if 'num_slices' in results:
            print(f'\n分析完成: {results["num_events"]} 个events, {results["num_slices"]} 个切片')
            print(f'切片时长: {results["slice_duration"]:.1f} 秒')
            
            # 如果有平均统计结果，显示平均信息
            if 'averaged' in results:
                avg_results = results['averaged']
                print(f'\n平均统计结果:')
                print(f'  有效切片数: {results["num_valid_slices"]}/{results["num_slices"]}')
                print(f'  主导频率: {avg_results["dominant_freq"]:.4f} Hz')
                print(f'  平均功率: {np.mean(avg_results["mean_power"]):.2e}')
        else:
            print(f'\n分析完成: {results["num_events"]} 个events')
            print(f'主导频率: {results["dominant_freq"]:.4f}Hz')
    except Exception as e:
        print(f'分析失败: {e}')
        import traceback
        traceback.print_exc()

