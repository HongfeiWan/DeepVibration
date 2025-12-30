#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用小波变换和Lomb-Scargle方法分析拼接的event信号
筛选RT且非Inhibit的events，拼接其CH0信号，并进行时频分析
"""
import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import lombscargle
from typing import Optional, Tuple, Dict

# 尝试导入PyWavelets用于小波变换
try:
    import pywt
    PYWAVELETS_AVAILABLE = True
except ImportError:
    PYWAVELETS_AVAILABLE = False
    print("警告: PyWavelets未安装，小波变换功能不可用。请安装: pip install PyWavelets")

# 添加路径以便导入 utils 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
python_dir = os.path.dirname(os.path.dirname(current_dir))
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from utils.visualize import get_h5_files

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

def continuous_wavelet_transform(signal_data: np.ndarray,
                                 sampling_rate: float,
                                 wavename: str = 'cmor3-3',
                                 totalscal: int = 64,
                                 min_freq: Optional[float] = None,
                                 max_freq: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对信号进行连续小波变换（CWT）
    
    参数:
        signal_data: 信号数据数组
        sampling_rate: 采样率（Hz）
        wavename: 小波名称，格式为'cmorB-C'（B为带宽参数，C为中心频率参数）
                  例如'cmor3-3'对应MATLAB的'cmor3-3'
        totalscal: 总尺度数
        min_freq: 最小频率（Hz），如果为None则自动计算
        max_freq: 最大频率（Hz），如果为None则自动计算
    
    返回:
        (尺度数组, 频率数组, 小波系数矩阵) 的元组
        小波系数矩阵形状为 (尺度数, 时间点数)
    """
    if not PYWAVELETS_AVAILABLE:
        raise ImportError("PyWavelets未安装，无法进行小波变换。请安装: pip install PyWavelets")
    
    print(f'\n连续小波变换（CWT）分析:')
    print(f'  信号长度: {len(signal_data)} 个点')
    print(f'  采样率: {sampling_rate:.1f} Hz')
    print(f'  小波类型: {wavename}')
    print(f'  总尺度数: {totalscal}')
    
    # 解析小波参数（cmorB-C格式）
    if wavename.startswith('cmor'):
        # 提取带宽和中心频率参数
        parts = wavename[4:].split('-')
        if len(parts) == 2:
            bandwidth_param = float(parts[0])
            center_freq_param = float(parts[1])
            print(f'  带宽参数: {bandwidth_param}, 中心频率参数: {center_freq_param}')
        else:
            bandwidth_param = 1.0
            center_freq_param = 1.0
            print(f'  警告: 无法解析小波参数，使用默认值')
    else:
        raise ValueError(f'不支持的小波类型: {wavename}，目前只支持cmor类型')
    
    # 计算尺度范围
    # 对于复Morlet小波，尺度与频率的关系: f = Fc * fs / scale
    # 其中Fc是中心频率，fs是采样率，scale是尺度
    # PyWavelets中的中心频率可以通过pywt.central_frequency获取
    try:
        # 构造完整的小波名称
        full_wavename = f'cmor{bandwidth_param}-{center_freq_param}'
        fc = pywt.central_frequency(pywt.ContinuousWavelet(full_wavename))
        print(f'  小波中心频率: {fc:.4f}')
    except:
        # 如果无法获取，使用近似值
        fc = center_freq_param / (2 * np.pi)
        print(f'  使用近似中心频率: {fc:.4f}')
    
    # 计算频率范围
    if max_freq is None:
        # 默认最大频率为采样率的1/2（Nyquist频率）
        max_freq = sampling_rate / 2.0
    if min_freq is None:
        # 默认最小频率根据信号长度和采样率计算
        min_freq = sampling_rate / len(signal_data)
    
    print(f'  频率范围: {min_freq:.3f} - {max_freq:.3f} Hz')
    
    # 计算尺度范围
    # 尺度与频率的关系: scale = fc * fs / f
    max_scale = fc * sampling_rate / min_freq
    min_scale = fc * sampling_rate / max_freq
    
    print(f'  尺度范围: {min_scale:.2f} - {max_scale:.2f}')
    
    # 生成尺度数组（对数分布，类似MATLAB）
    scales = np.logspace(np.log10(min_scale), np.log10(max_scale), totalscal)
    
    # 执行连续小波变换
    print(f'  正在执行CWT...')
    coefficients, frequencies_cwt = pywt.cwt(signal_data, scales, full_wavename, 1.0/sampling_rate)
    
    # 计算对应的频率（从尺度反推）
    freq_array = fc * sampling_rate / scales
    
    # 注意：pywt.cwt返回的frequencies可能与我们的计算略有不同，使用我们计算的频率
    # 但保留系数矩阵的顺序（从高频到低频，或反之）
    
    print(f'  CWT完成，系数矩阵形状: {coefficients.shape}')
    
    return scales, freq_array, coefficients

def analyze_filtered_rt_non_inhibit_events_wavelet(ch0_3_file: str,
                                                   ch5_file: str,
                                                   rt_cut: float = 6000.0,
                                                   max_events: Optional[int] = None,
                                                   ch0_idx: int = 0,
                                                   ch5_idx: int = 0,
                                                   wavename: str = 'cmor3-3',
                                                   totalscal: int = 64,
                                                   min_freq: Optional[float] = None,
                                                   max_freq: Optional[float] = None,
                                                   detrend: bool = True,
                                                   save_path: Optional[str] = None,
                                                   show_plot: bool = True,
                                                   figsize: Tuple[int, int] = (16, 12)) -> Dict:
    """
    筛选RT且非Inhibit的events，拼接其CH0信号，并进行小波变换分析
    
    参数:
        ch0_3_file: CH0-3文件路径
        ch5_file: CH5文件路径
        rt_cut: RT信号的截断阈值
        max_events: 最多拼接的event数量，如果为None则使用所有符合条件的events
        ch0_idx: CH0通道索引
        ch5_idx: CH5通道索引
        wavename: 小波名称（例如'cmor3-3'）
        totalscal: 总尺度数
        min_freq: 最小频率（Hz），如果为None则自动计算
        max_freq: 最大频率（Hz），如果为None则自动计算
        detrend: 是否去趋势
        save_path: 保存图片路径
        show_plot: 是否显示图片
        figsize: 图片大小
    
    返回:
        包含分析结果的字典
    """
    if not PYWAVELETS_AVAILABLE:
        raise ImportError("PyWavelets未安装，无法进行小波变换。请安装: pip install PyWavelets")
    
    print('=' * 70)
    print(f'分析RT且非Inhibit的events的CH0信号（小波变换）')
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
    
    # 4. 去趋势
    if detrend:
        concatenated_signal = signal.detrend(concatenated_signal)
    
    # 5. 计算采样率
    if len(concatenated_times) > 1:
        dt_array = np.diff(concatenated_times)
        avg_dt = np.mean(dt_array)
        sampling_rate = 1.0 / avg_dt
        print(f'\n计算采样率:')
        print(f'  平均时间间隔: {avg_dt:.6e} s')
        print(f'  采样率: {sampling_rate:.1f} Hz')
    else:
        raise ValueError('信号长度不足，无法计算采样率')
    
    # 6. 降采样（如果数据点太多）
    max_points = 100000  # 最多5万个点进行CWT分析
    if len(concatenated_signal) > max_points:
        print(f'\n数据点过多 ({len(concatenated_signal)})，降采样到 {max_points} 个点')
        indices = np.linspace(0, len(concatenated_signal) - 1, max_points, dtype=int)
        concatenated_signal = concatenated_signal[indices]
        concatenated_times = concatenated_times[indices]
        # 重新计算采样率
        dt_array = np.diff(concatenated_times)
        avg_dt = np.mean(dt_array)
        sampling_rate = 1.0 / avg_dt
        print(f'  降采样后采样率: {sampling_rate:.1f} Hz')
    
    # 7. 执行小波变换
    scales, frequencies, coefficients = continuous_wavelet_transform(
        concatenated_signal, sampling_rate, wavename, totalscal, min_freq, max_freq
    )
    
    # 8. 绘制结果
    fig, axes = plt.subplots(4, 1, figsize=figsize)
    
    # 1. 原始信号
    ax = axes[0]
    ax.plot(concatenated_times * 1e3, concatenated_signal, 'b-', linewidth=0.5, alpha=0.7)
    ax.set_xlabel('Time (ms)', fontsize=10)
    ax.set_ylabel('Amplitude (ADC counts)', fontsize=10)
    ax.set_title(f'Concatenated Signal (RT & Non-Inhibit)\n({len(selected_indices)} events, {len(concatenated_signal)} points)', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 2. 小波系数幅度（时频图）
    ax = axes[1]
    # 使用绝对值显示小波系数
    coeff_abs = np.abs(coefficients)
    # 转置以匹配时间-频率坐标
    im = ax.pcolormesh(concatenated_times * 1e3, frequencies, coeff_abs, 
                       shading='gouraud', cmap='jet')
    ax.set_xlabel('Time (ms)', fontsize=10)
    ax.set_ylabel('Frequency (Hz)', fontsize=10)
    ax.set_title(f'Wavelet Transform (Magnitude) - {wavename}', fontsize=11)
    ax.set_yscale('log')
    plt.colorbar(im, ax=ax, label='Magnitude')
    ax.grid(True, alpha=0.3)
    
    # 3. 小波系数实部
    ax = axes[2]
    im = ax.pcolormesh(concatenated_times * 1e3, frequencies, np.real(coefficients),
                       shading='gouraud', cmap='RdBu_r')
    ax.set_xlabel('Time (ms)', fontsize=10)
    ax.set_ylabel('Frequency (Hz)', fontsize=10)
    ax.set_title(f'Wavelet Transform (Real Part)', fontsize=11)
    ax.set_yscale('log')
    plt.colorbar(im, ax=ax, label='Real')
    ax.grid(True, alpha=0.3)
    
    # 4. 平均功率谱（在所有时间上的平均）
    ax = axes[3]
    mean_power = np.mean(coeff_abs**2, axis=1)
    ax.semilogy(frequencies, mean_power, 'b-', linewidth=1.5)
    ax.set_xlabel('Frequency (Hz)', fontsize=10)
    ax.set_ylabel('Mean Power', fontsize=10)
    ax.set_title('Average Power Spectrum (from Wavelet Transform)', fontsize=11)
    ax.set_xscale('log')
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
        'scales': scales,
        'frequencies': frequencies,
        'wavelet_coefficients': coefficients,
        'selected_event_indices': selected_indices,
        'num_selected_events': len(selected_indices),
        'rt_cut': rt_cut,
        'sampling_rate': sampling_rate,
        'wavename': wavename,
        'totalscal': totalscal
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
            print('分析RT且非Inhibit的events的CH0信号（小波变换）')
            print('=' * 70)
            
            try:
                results = analyze_filtered_rt_non_inhibit_events_wavelet(
                    ch0_3_file, ch5_file,
                    rt_cut=6000.0,      # RT信号截断阈值
                    max_events=None,    # 使用所有符合条件的events，或设置一个数字限制
                    ch0_idx=0,          # CH0通道索引
                    ch5_idx=0,          # CH5通道索引
                    wavename='cmor3-3', # 复Morlet小波，带宽3，中心频率3
                    totalscal=64,       # 总尺度数
                    min_freq=None,      # 自动计算
                    max_freq=None,      # 自动计算（默认到Nyquist频率）
                    detrend=True,
                    show_plot=True
                )
                
                print(f'\n分析完成！')
                print(f'  符合条件的event数: {results["num_selected_events"]}')
                print(f'  拼接信号长度: {len(results["concatenated_signal"])} 点')
                print(f'  采样率: {results["sampling_rate"]:.1f} Hz')
                print(f'  小波系数矩阵形状: {results["wavelet_coefficients"].shape}')
                print(f'  频率范围: {results["frequencies"][-1]:.3f} - {results["frequencies"][0]:.3f} Hz')
                
            except Exception as e:
                print(f'分析失败: {e}')
                import traceback
                traceback.print_exc()
        else:
            print('未找到匹配的文件对')
    else:
        print('未找到CH0-3或CH5文件')

