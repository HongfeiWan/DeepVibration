#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用Lomb-Scargle方法分析拼接的event信号
每73个event拼接一次CH0信号，根据时间戳拼接，间隔处补0
"""
import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import lombscargle
from typing import Optional, Tuple, Dict

# 添加路径以便导入 utils 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
python_dir = os.path.dirname(os.path.dirname(current_dir))
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from utils.visualize import get_h5_files

def concatenate_events_by_time(h5_file: str,
                               channel_idx: int = 0,
                               num_events: int = 7,
                               event_start_idx: int = 0,
                               event_indices: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据时间戳拼接event信号
    
    参数:
        h5_file: HDF5文件路径
        channel_idx: 通道索引（CH0为0）
        num_events: 拼接的event数量（默认7，如果event_indices提供则忽略此参数）
        event_start_idx: 起始event索引（默认0，如果event_indices提供则忽略此参数）
        event_indices: 要拼接的事件索引数组，如果提供则使用此参数而不是num_events和event_start_idx
    
    返回:
        (拼接后的信号数组, 对应的时间戳数组) 的元组
    """
    with h5py.File(h5_file, 'r') as f:
        channel_data = f['channel_data']
        time_data = f['time_data']
        
        time_samples, num_channels, total_events = channel_data.shape
        
        # 如果提供了event_indices，使用它；否则使用event_start_idx和num_events
        if event_indices is not None:
            if len(event_indices) == 0:
                raise ValueError(f'提供的event索引数组为空')
            event_indices_array = np.array(event_indices)
            actual_num_events = len(event_indices_array)
            print(f'拼接 {actual_num_events} 个筛选后的events')
        else:
            # 检查索引范围
            event_end_idx = min(event_start_idx + num_events, total_events)
            actual_num_events = event_end_idx - event_start_idx
            
            if actual_num_events < 1:
                raise ValueError(f'没有足够的event进行拼接')
            
            print(f'拼接 {actual_num_events} 个events (索引 {event_start_idx} 到 {event_end_idx-1})')
            event_indices_array = np.arange(event_start_idx, event_end_idx)
        
        # 提取指定范围的event数据和时间戳
        event_timestamps = np.array(time_data[event_indices_array])
        
        # 参数设置
        sampling_interval_ns = 4.0  # 4ns per sample
        sampling_interval_s = sampling_interval_ns * 1e-9  # 转换为秒
        event_duration_s = time_samples * sampling_interval_s  # 120μs = 0.00012秒
        
        print(f'每个event参数:')
        print(f'  采样点数: {time_samples}')
        print(f'  采样间隔: {sampling_interval_ns} ns = {sampling_interval_s} s')
        print(f'  Event时长: {event_duration_s*1e6:.2f} μs')
        
        # 计算总时间跨度
        first_event_start_time = event_timestamps[0]
        last_event_start_time = event_timestamps[-1]
        last_event_end_time = last_event_start_time + event_duration_s
        total_time_span = last_event_end_time - first_event_start_time
        
        print(f'\n时间信息:')
        print(f'  第一个event开始时间: {first_event_start_time:.6f} s')
        print(f'  最后一个event结束时间: {last_event_end_time:.6f} s')
        print(f'  总时间跨度: {total_time_span:.6f} s')
        
        # 直接拼接所有event的波形和时间戳，不创建完整的均匀时间网格
        # 这样可以避免内存问题，同时Lomb-Scargle适合处理这种不规则采样
        print(f'\n正在拼接events（直接拼接，不补0）...')
        
        # 预分配数组（只包含实际信号点，不包含补0）
        total_signal_points = actual_num_events * time_samples
        concatenated_signal = np.zeros(total_signal_points, dtype=np.float64)
        concatenated_times = np.zeros(total_signal_points, dtype=np.float64)
        
        signal_idx = 0
        for i, event_idx in enumerate(event_indices_array):
            # 获取event的波形数据
            event_waveform = channel_data[:, channel_idx, event_idx]
            event_start_time = event_timestamps[i]
            
            # 计算这个event内每个采样点的时间戳
            event_times = event_start_time + np.arange(time_samples) * sampling_interval_s
            
            # 拼接信号和时间戳
            concatenated_signal[signal_idx:signal_idx + time_samples] = event_waveform
            concatenated_times[signal_idx:signal_idx + time_samples] = event_times
            
            signal_idx += time_samples
            
            if (i + 1) % 10 == 0 or i == len(event_indices_array) - 1:
                print(f'  已处理 {i+1}/{actual_num_events} 个events')
        
        print(f'\n拼接完成:')
        print(f'  信号长度: {len(concatenated_signal)} 个点（全部是非零点）')
        print(f'  时间跨度: {concatenated_times[-1] - concatenated_times[0]:.6f} s')
        return concatenated_signal, concatenated_times

def lomb_scargle_analysis(signal_data: np.ndarray,
                          timestamps: np.ndarray,
                          min_freq: float,
                          max_freq: float,
                          detrend: bool = True,
                          oversampling: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用Lomb-Scargle方法分析信号（专注于低频分析）
    
    参数:
        signal_data: 信号数据数组
        timestamps: 对应的时间戳数组（秒）
        min_freq: 最小频率（Hz），必需参数
        max_freq: 最大频率（Hz），必需参数（例如200Hz用于低频分析）
        detrend: 是否去趋势
        oversampling: 过采样因子
    
    返回:
        (频率数组, 功率谱) 的元组
    """
    # 归一化时间（避免数值问题）
    t0 = timestamps[0]
    t_normalized = timestamps - t0
    
    print(f'\nLomb-Scargle分析:')
    print(f'  总信号点数: {len(signal_data)}')
    print(f'  目标频率范围: {min_freq:.3f} - {max_freq:.3f} Hz')
    
    # 根据目标频率范围的5倍计算目标采样率
    target_sampling_rate = max_freq * 5.0  # 5倍过采样
    print(f'  目标采样率: {target_sampling_rate:.1f} Hz (max_freq的5倍)')
    
    # 计算Nyquist频率要求（至少需要max_freq的2倍采样率，为了安全使用5倍）
    min_required_sampling_rate = max_freq * 2.0  # Nyquist最低要求
    print(f'  最低所需采样率: {min_required_sampling_rate:.1f} Hz (Nyquist: max_freq × 2)')
    
    # 设置最大数据点数限制，避免内存溢出
    # Lomb-Scargle的计算复杂度是O(N*M)，N是数据点数，M是频率点数
    # 但首先要确保满足采样率要求
    max_data_points = 100000  # 最多50万个数据点（提高限制以确保满足采样率要求）
    
    # 计算当前信号的平均采样率
    if len(timestamps) > 1:
        dt_array = np.diff(timestamps)
        avg_dt = np.mean(dt_array)
        current_sampling_rate = 1.0 / avg_dt
        print(f'  当前平均采样率: {current_sampling_rate:.1e} Hz')
        
        # 计算基于目标采样率的降采样因子（优先考虑满足目标采样率）
        downsample_factor_by_rate = max(1, int(current_sampling_rate / target_sampling_rate))
        
        # 计算基于Nyquist要求的最大降采样因子
        max_downsample_factor = int(current_sampling_rate / min_required_sampling_rate)
        if max_downsample_factor < 1:
            max_downsample_factor = 1
        
        # 计算基于数据点数的降采样因子（如果数据点太多，但要满足采样率要求）
        if len(signal_data) > max_data_points:
            # 计算基于数据点数需要的降采样因子
            ideal_downsample_by_points = len(signal_data) / max_data_points
            
            # 但降采样因子不能超过max_downsample_factor，否则无法满足Nyquist要求
            downsample_factor_by_points = min(
                max(1, int(ideal_downsample_by_points)),
                max_downsample_factor
            )
            print(f'  数据点数过多 ({len(signal_data)})，理想降采样因子: {ideal_downsample_by_points:.1f}')
            print(f'  但受Nyquist限制，最大允许降采样因子: {max_downsample_factor}')
        else:
            downsample_factor_by_points = 1
        
        # 使用满足采样率要求的降采样因子
        # 优先满足目标采样率，但如果数据点太多，在满足Nyquist的前提下进行降采样
        downsample_factor = min(
            max(downsample_factor_by_rate, downsample_factor_by_points),
            max_downsample_factor
        )
        
        print(f'  降采样因子: {downsample_factor} (每{downsample_factor}个点取1个)')
        
        # 降采样信号和时间戳
        if downsample_factor > 1:
            signal_downsampled = signal_data[::downsample_factor]
            timestamps_downsampled = t_normalized[::downsample_factor]
            actual_sampling_rate_after_downsample = current_sampling_rate / downsample_factor
            print(f'  降采样后点数: {len(signal_downsampled)} (从 {len(signal_data)} 点)')
            print(f'  降采样后采样率: {actual_sampling_rate_after_downsample:.1f} Hz')
        else:
            signal_downsampled = signal_data
            timestamps_downsampled = t_normalized
            print(f'  无需降采样（当前采样率已足够）')
    else:
        signal_downsampled = signal_data
        timestamps_downsampled = t_normalized
        downsample_factor = 1
    
    # 提取非零点（如果有0点的话）
    non_zero_mask = signal_downsampled != 0
    if np.any(~non_zero_mask):
        signal_nonzero = signal_downsampled[non_zero_mask].copy()
        timestamps_nonzero = timestamps_downsampled[non_zero_mask].copy()
        print(f'  非零点数: {len(signal_nonzero)} (从 {len(signal_downsampled)} 点)')
    else:
        signal_nonzero = signal_downsampled
        timestamps_nonzero = timestamps_downsampled
        print(f'  所有点都是非零点: {len(signal_nonzero)} 个点')
    
    # 去趋势
    if detrend:
        signal_nonzero = signal.detrend(signal_nonzero)
    
    # 计算时间跨度
    if len(timestamps_nonzero) == 0:
        raise ValueError('没有信号点可供分析')
    duration = timestamps_nonzero[-1] - timestamps_nonzero[0]
    print(f'  时间跨度: {duration:.6f} s')
    
    # 验证频率范围
    if min_freq >= max_freq:
        raise ValueError(f'min_freq ({min_freq}) 必须小于 max_freq ({max_freq})')
    
    # 计算频率网格
    # 频率分辨率至少是 1/duration
    freq_resolution = 1.0 / duration
    print(f'  频率分辨率: {freq_resolution:.6f} Hz (1/duration)')
    
    nfreq = int(oversampling * duration * (max_freq - min_freq))
    if nfreq < 100:
        nfreq = 100  # 至少100个点
    elif nfreq > 50000:
        # 限制频率点数，避免内存问题
        # 同时考虑实际数据点数，确保N*M不会太大
        max_freq_points_by_memory = min(50000, max(1000, int(len(signal_nonzero) * 0.1)))
        nfreq = min(nfreq, max_freq_points_by_memory)
        print(f'  警告: 频率点数过多，限制到 {nfreq} 个点（基于内存考虑）')
    
    freqs = np.linspace(2 * np.pi * min_freq, 2 * np.pi * max_freq, nfreq)
    print(f'  频率点数: {nfreq}')
    
    # 显示实际采样率（降采样后）
    if len(signal_nonzero) > 1:
        actual_dt = np.mean(np.diff(timestamps_nonzero))
        actual_sampling_rate = 1.0 / actual_dt
        nyquist_freq = actual_sampling_rate / 2.0
        print(f'  实际采样率（降采样后）: {actual_sampling_rate:.1f} Hz')
        print(f'  Nyquist频率: {nyquist_freq:.1f} Hz')
        if nyquist_freq < max_freq:
            print(f'  警告: Nyquist频率 ({nyquist_freq:.1f} Hz) 低于目标最大频率 ({max_freq:.1f} Hz)！')
            print(f'        可能导致频率混叠，建议降低max_freq或增加数据点数限制')
    
    # Lomb-Scargle周期图
    power = lombscargle(timestamps_nonzero, signal_nonzero, freqs, normalize=True)
    
    # 转换回Hz
    freqs_hz = freqs / (2 * np.pi)
    
    return freqs_hz, power

def filter_rt_non_inhibit_events(ch0_3_file: str,
                                 ch5_file: str,
                                 rt_cut: float = 6000.0,
                                 ch0_idx: int = 0,
                                 ch5_idx: int = 0,
                                 verbose: bool = True) -> np.ndarray:
    """
    筛选符合RT条件且不符合Inhibit条件的事件索引
    
    参数:
        ch0_3_file: CH0-3文件路径
        ch5_file: CH5文件路径
        rt_cut: RT信号的截断阈值（CH5最大值 > rt_cut）
        ch0_idx: CH0通道索引
        ch5_idx: CH5通道索引
        verbose: 是否打印详细信息
    
    返回:
        符合条件的事件索引数组
    """
    if verbose:
        print('=' * 70)
        print('筛选RT且非Inhibit的events')
        print('=' * 70)
    
    # 读取CH0-3文件（分析CH0的最小值）
    with h5py.File(ch0_3_file, 'r') as f_ch0:
        ch0_channel_data = f_ch0['channel_data']
        ch0_time_samples, ch0_num_channels, ch0_num_events = ch0_channel_data.shape
        
        if verbose:
            print(f'\nCH0-3文件维度: (时间点={ch0_time_samples}, 通道数={ch0_num_channels}, 事件数={ch0_num_events})')
        
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
    
    # 非Inhibit信号：CH0最小值 > 0
    non_inhibit_mask = ch0_min_values > 0
    
    # 同时满足RT且非Inhibit条件
    selected_mask = rt_mask & non_inhibit_mask
    selected_indices = np.where(selected_mask)[0]
    
    if verbose:
        rt_count = np.sum(rt_mask)
        inhibit_count = np.sum(ch0_min_values <= 0)
        selected_count = len(selected_indices)
        
        print(f'\n筛选结果:')
        print(f'  总事件数: {ch0_num_events}')
        print(f'  RT信号数 (CH5 max > {rt_cut:.2f}): {rt_count}')
        print(f'  Inhibit信号数 (CH0 min <= 0): {inhibit_count}')
        print(f'  符合条件的event数 (RT且非Inhibit): {selected_count}')
        print(f'  符合条件比例: {selected_count/ch0_num_events*100:.2f}%')
        print('=' * 70)
    
    return selected_indices

def analyze_filtered_rt_non_inhibit_events(ch0_3_file: str,
                                           ch5_file: str,
                                           min_freq: float,
                                           max_freq: float,
                                           rt_cut: float = 6000.0,
                                           max_events: Optional[int] = None,
                                           ch0_idx: int = 0,
                                           ch5_idx: int = 0,
                                           detrend: bool = True,
                                           oversampling: float = 2.0,
                                           save_path: Optional[str] = None,
                                           show_plot: bool = True,
                                           figsize: Tuple[int, int] = (16, 10)) -> Dict:
    """
    筛选RT且非Inhibit的events，拼接其CH0信号，并进行Lomb-Scargle分析
    
    参数:
        ch0_3_file: CH0-3文件路径
        ch5_file: CH5文件路径
        min_freq: 最小频率（Hz），必需参数（例如0.1）
        max_freq: 最大频率（Hz），必需参数（例如200.0用于低频分析）
        rt_cut: RT信号的截断阈值
        max_events: 最多拼接的event数量，如果为None则使用所有符合条件的events
        ch0_idx: CH0通道索引
        ch5_idx: CH5通道索引
        detrend: 是否去趋势
        oversampling: 过采样因子
        save_path: 保存图片路径
        show_plot: 是否显示图片
        figsize: 图片大小
    
    返回:
        包含分析结果的字典
    """
    print('=' * 70)
    print(f'分析RT且非Inhibit的events的CH0信号')
    print('=' * 70)
    
    # 1. 筛选符合条件的events
    selected_indices = filter_rt_non_inhibit_events(
        ch0_3_file, ch5_file, rt_cut, ch0_idx, ch5_idx, verbose=True
    )
    
    if len(selected_indices) == 0:
        raise ValueError('没有符合条件的events（RT且非Inhibit）')
    
    # 2. 限制使用的event数量
    if max_events is not None and len(selected_indices) > max_events:
        print(f'\n限制使用前 {max_events} 个符合条件的events')
        selected_indices = selected_indices[:max_events]
    
    # 3. 拼接筛选后的events
    print(f'\n拼接 {len(selected_indices)} 个符合条件的events...')
    concatenated_signal, concatenated_times = concatenate_events_by_time(
        ch0_3_file, ch0_idx, num_events=0, event_start_idx=0,
        event_indices=selected_indices
    )
    
    # 4. Lomb-Scargle分析
    freqs, power = lomb_scargle_analysis(
        concatenated_signal, concatenated_times,
        min_freq=min_freq, max_freq=max_freq,
        detrend=detrend, oversampling=oversampling
    )
    
    # 5. 绘制结果
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    
    # 1. 拼接后的完整时域信号
    ax = axes[0, 0]
    ax.plot(concatenated_times * 1e3, concatenated_signal, 'b-', linewidth=0.5, alpha=0.7)
    ax.set_xlabel('Time (ms)', fontsize=10)
    ax.set_ylabel('Amplitude (ADC counts)', fontsize=10)
    ax.set_title(f'Concatenated Signal (RT & Non-Inhibit)\n({len(selected_indices)} events, {len(concatenated_signal)} points)', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 2. 拼接信号的前一部分（放大显示）
    ax = axes[0, 1]
    zoom_samples = min(10000, len(concatenated_signal))
    ax.plot(concatenated_times[:zoom_samples] * 1e3, concatenated_signal[:zoom_samples], 
            'b-', linewidth=0.8, alpha=0.8)
    ax.set_xlabel('Time (ms)', fontsize=10)
    ax.set_ylabel('Amplitude (ADC counts)', fontsize=10)
    ax.set_title(f'Zoomed View (First {zoom_samples} points)', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 3. 非零信号的时域图
    ax = axes[1, 0]
    non_zero_mask = concatenated_signal != 0
    ax.plot(concatenated_times[non_zero_mask] * 1e3, concatenated_signal[non_zero_mask], 
            'b-', linewidth=0.5, alpha=0.7)
    ax.set_xlabel('Time (ms)', fontsize=10)
    ax.set_ylabel('Amplitude (ADC counts)', fontsize=10)
    ax.set_title(f'Non-Zero Signal Points Only\n({np.sum(non_zero_mask)} points)', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 4. 频谱（线性尺度，低频部分）
    ax = axes[1, 1]
    ax.plot(freqs, power, 'b-', linewidth=1)
    ax.set_xlabel('Frequency (Hz)', fontsize=10)
    ax.set_ylabel('Power Spectrum', fontsize=10)
    ax.set_title('Lomb-Scargle Power Spectrum (Linear)', fontsize=11)
    max_display_freq = min(1000, freqs[-1])
    ax.set_xlim(0, max_display_freq)
    ax.grid(True, alpha=0.3)
    
    # 5. 频谱（对数尺度，低频部分）
    ax = axes[2, 0]
    ax.semilogy(freqs, power, 'b-', linewidth=1)
    ax.set_xlabel('Frequency (Hz)', fontsize=10)
    ax.set_ylabel('Power Spectrum (log scale)', fontsize=10)
    ax.set_title('Lomb-Scargle Power Spectrum (Log Scale)', fontsize=11)
    ax.set_xlim(0, max_display_freq)
    ax.grid(True, alpha=0.3)
    
    # 6. 频谱（对数-对数尺度，全频段）
    ax = axes[2, 1]
    ax.loglog(freqs, power, 'b-', linewidth=1)
    ax.set_xlabel('Frequency (Hz)', fontsize=10)
    ax.set_ylabel('Power Spectrum (log-log scale)', fontsize=10)
    ax.set_title('Lomb-Scargle Power Spectrum (Log-Log)', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'\n图片已保存至: {save_path}')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    # 返回结果
    results = {
        'concatenated_signal': concatenated_signal,
        'concatenated_times': concatenated_times,
        'frequencies': freqs,
        'power_spectrum': power,
        'selected_event_indices': selected_indices,
        'num_selected_events': len(selected_indices),
        'rt_cut': rt_cut,
        'zero_ratio': float(np.sum(concatenated_signal == 0) / len(concatenated_signal) * 100)
    }
    
    return results

def analyze_concatenated_events(h5_file: str,
                                min_freq: float,
                                max_freq: float,
                                channel_idx: int = 0,
                                num_events: int = 73,
                                event_start_idx: int = 0,
                                detrend: bool = True,
                                oversampling: float = 2.0,
                                save_path: Optional[str] = None,
                                show_plot: bool = True,
                                figsize: Tuple[int, int] = (16, 10)) -> Dict:
    """
    分析拼接的event信号
    
    参数:
        h5_file: HDF5文件路径
        min_freq: 最小频率（Hz），必需参数（例如0.1）
        max_freq: 最大频率（Hz），必需参数（例如200.0用于低频分析）
        channel_idx: 通道索引
        num_events: 拼接的event数量
        event_start_idx: 起始event索引
        detrend: 是否去趋势
        oversampling: 过采样因子
        save_path: 保存图片路径
        show_plot: 是否显示图片
        figsize: 图片大小
    
    返回:
        包含分析结果的字典
    """
    print('=' * 70)
    print(f'分析拼接的event信号: {os.path.basename(h5_file)}')
    print('=' * 70)
    
    # 拼接events
    concatenated_signal, concatenated_times = concatenate_events_by_time(
        h5_file, channel_idx, num_events, event_start_idx
    )
    
    # Lomb-Scargle分析
    freqs, power = lomb_scargle_analysis(
        concatenated_signal, concatenated_times,
        min_freq=min_freq, max_freq=max_freq,
        detrend=detrend, oversampling=oversampling
    )
    
    # 绘制结果
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    
    # 1. 拼接后的完整时域信号
    ax = axes[0, 0]
    ax.plot(concatenated_times * 1e3, concatenated_signal, 'b-', linewidth=0.5, alpha=0.7)
    ax.set_xlabel('Time (ms)', fontsize=10)
    ax.set_ylabel('Amplitude (ADC counts)', fontsize=10)
    ax.set_title(f'Concatenated Signal\n({num_events} events, {len(concatenated_signal)} points)', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 2. 拼接信号的前一部分（放大显示）
    ax = axes[0, 1]
    zoom_samples = min(10000, len(concatenated_signal))
    ax.plot(concatenated_times[:zoom_samples] * 1e3, concatenated_signal[:zoom_samples], 
            'b-', linewidth=0.8, alpha=0.8)
    ax.set_xlabel('Time (ms)', fontsize=10)
    ax.set_ylabel('Amplitude (ADC counts)', fontsize=10)
    ax.set_title(f'Zoomed View (First {zoom_samples} points)', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 3. 非零信号的时域图
    ax = axes[1, 0]
    non_zero_mask = concatenated_signal != 0
    ax.plot(concatenated_times[non_zero_mask] * 1e3, concatenated_signal[non_zero_mask], 
            'b-', linewidth=0.5, alpha=0.7)
    ax.set_xlabel('Time (ms)', fontsize=10)
    ax.set_ylabel('Amplitude (ADC counts)', fontsize=10)
    ax.set_title(f'Non-Zero Signal Points Only\n({np.sum(non_zero_mask)} points)', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 4. 频谱（线性尺度，低频部分）
    ax = axes[1, 1]
    ax.plot(freqs, power, 'b-', linewidth=1)
    ax.set_xlabel('Frequency (Hz)', fontsize=10)
    ax.set_ylabel('Power Spectrum', fontsize=10)
    ax.set_title('Lomb-Scargle Power Spectrum (Linear)', fontsize=11)
    # 限制显示范围以便看清低频
    max_display_freq = min(1000, freqs[-1])
    ax.set_xlim(0, max_display_freq)
    ax.grid(True, alpha=0.3)
    
    # 5. 频谱（对数尺度，低频部分）
    ax = axes[2, 0]
    ax.semilogy(freqs, power, 'b-', linewidth=1)
    ax.set_xlabel('Frequency (Hz)', fontsize=10)
    ax.set_ylabel('Power Spectrum (log scale)', fontsize=10)
    ax.set_title('Lomb-Scargle Power Spectrum (Log Scale)', fontsize=11)
    ax.set_xlim(0, max_display_freq)
    ax.grid(True, alpha=0.3)
    
    # 6. 频谱（对数-对数尺度，全频段）
    ax = axes[2, 1]
    ax.loglog(freqs, power, 'b-', linewidth=1)
    ax.set_xlabel('Frequency (Hz)', fontsize=10)
    ax.set_ylabel('Power Spectrum (log-log scale)', fontsize=10)
    ax.set_title('Lomb-Scargle Power Spectrum (Log-Log)', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'\n图片已保存至: {save_path}')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    # 返回结果
    results = {
        'concatenated_signal': concatenated_signal,
        'concatenated_times': concatenated_times,
        'frequencies': freqs,
        'power_spectrum': power,
        'num_events': num_events,
        'event_start_idx': event_start_idx,
        'zero_ratio': float(np.sum(concatenated_signal == 0) / len(concatenated_signal) * 100)
    }
    
    return results


# 示例使用
if __name__ == '__main__':
    # 获取匹配的文件对
    h5_files = get_h5_files()
    
    if 'CH0-3' in h5_files and h5_files['CH0-3'] and 'CH5' in h5_files and h5_files['CH5']:
        # 查找匹配的文件对
        ch0_3_files = h5_files['CH0-3']
        ch5_files = h5_files['CH5']

        # 匹配文件名
        ch0_3_dict = {os.path.basename(f): f for f in ch0_3_files}
        ch5_dict = {os.path.basename(f): f for f in ch5_files}
        
        matched_pairs = []
        for filename in ch0_3_dict.keys():
            if filename in ch5_dict:
                matched_pairs.append((ch0_3_dict[filename], ch5_dict[filename]))
        
        if matched_pairs:
            ch0_3_file, ch5_file = matched_pairs[0]
            
            print('=' * 70)
            print('分析RT且非Inhibit的events的CH0信号')
            print('=' * 70)
            
            try:
                results = analyze_filtered_rt_non_inhibit_events(
                    ch0_3_file, ch5_file,
                    min_freq=0.1,       # 最小频率（Hz）
                    max_freq=200.0,    # 最大频率（Hz），用于低频分析
                    rt_cut=6000.0,      # RT信号截断阈值
                    max_events=None,    # 使用所有符合条件的events，或设置一个数字限制
                    ch0_idx=0,          # CH0通道索引
                    ch5_idx=0,          # CH5通道索引
                    detrend=True,
                    oversampling=2.0,
                    show_plot=True
                )
                
                print(f'\n分析完成！')
                print(f'  符合条件的event数: {results["num_selected_events"]}')
                print(f'  拼接信号长度: {len(results["concatenated_signal"])} 点')
                print(f'  补0比例: {results["zero_ratio"]:.2f}%')
                print(f'  频谱频率点数: {len(results["frequencies"])}')
                
            except Exception as e:
                print(f'分析失败: {e}')
                import traceback
                traceback.print_exc()
        else:
            print('未找到匹配的文件对')
    else:
        print('未找到CH0-3或CH5文件')
