#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
筛选RT信号对应的CH0，对每个event独立进行小波变换，然后平均降低方差
频率范围：100kHz-10MHz
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
python_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from utils.visualize import get_h5_files

# 导入coincident模块的筛选函数
coincident_module_path = os.path.dirname(os.path.dirname(current_dir))
randomtrigger_inhibit_file = os.path.join(coincident_module_path, "randomtrigger&inhibit.py")

analyze_coincident_events = None
if os.path.exists(randomtrigger_inhibit_file):
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("randomtrigger_inhibit", randomtrigger_inhibit_file)
        randomtrigger_inhibit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(randomtrigger_inhibit)
        analyze_coincident_events = randomtrigger_inhibit.analyze_coincident_events
    except Exception as e:
        print(f'警告: 无法导入randomtrigger&inhibit模块: {e}')


def analyze_rt_only_events_wavelet(ch0_3_file: str,
                                   ch5_file: str,
                                   rt_cut: float = 6000.0,
                                   ch0_idx: int = 0,
                                   ch5_idx: int = 0,
                                   max_events: Optional[int] = None,
                                   detrend: bool = True,
                                   wavename: str = 'cmor3-3',
                                   totalscal: int = 128,
                                   min_freq: float = 100e3,  # 100 kHz
                                   max_freq: float = 10e6,   # 10 MHz
                                   save_path: Optional[str] = None,
                                   show_plot: bool = True) -> Dict:
    """
    筛选RT信号对应的CH0，对每个event独立进行小波变换，然后平均降低方差
    
    参数:
        ch0_3_file: CH0-3文件路径
        ch5_file: CH5文件路径
        rt_cut: RT信号截断阈值
        ch0_idx: CH0通道索引
        ch5_idx: CH5通道索引
        max_events: 最多分析的event数量
        detrend: 是否去趋势
        wavename: 小波名称
        totalscal: 总尺度数
        min_freq: 最小频率（Hz），默认100kHz
        max_freq: 最大频率（Hz），默认10MHz
        save_path: 保存图片路径
        show_plot: 是否显示图片
    
    返回:
        包含分析结果的字典
    """
    print('筛选RT信号对应的CH0，对每个event独立进行小波变换分析...')
    print(f'目标频率范围: {min_freq/1e3:.1f}kHz - {max_freq/1e6:.2f}MHz')
    
    # 1. 筛选RT且非Inhibit的events
    if analyze_coincident_events is not None:
        stats = analyze_coincident_events(ch0_3_file, ch5_file, rt_cut, ch0_idx, ch5_idx, verbose=False)
        rt_only_mask = stats['rt_mask'] & ~stats['inhibit_mask']
        selected_indices = np.where(rt_only_mask)[0]
    else:
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
        
        rt_mask = ch5_max_values > rt_cut
        non_inhibit_mask = ch0_min_values > 0
        selected_indices = np.where(rt_mask & non_inhibit_mask)[0]
    
    if len(selected_indices) == 0:
        raise ValueError('没有符合条件的events')
    
    if max_events is not None and len(selected_indices) > max_events:
        selected_indices = selected_indices[:max_events]
    
    num_events = len(selected_indices)
    print(f'筛选到 {num_events} 个events')
    
    # 2. 参数设置
    sampling_interval_ns = 4.0
    sampling_interval_s = sampling_interval_ns * 1e-9
    original_sampling_rate = 1.0 / sampling_interval_s  # 250 MSPS
    
    print(f'\n信号参数:')
    print(f'  采样间隔: {sampling_interval_ns} ns')
    print(f'  原始采样率: {original_sampling_rate/1e6:.2f} MSPS')
    
    # 3. 小波参数设置
    if wavename.startswith('cmor'):
        # 去掉'cmor'前缀，然后按'-'分割
        params_str = wavename[4:]  # 去掉'cmor'，得到类似'1.5-1'的字符串
        parts = params_str.split('-')
        if len(parts) == 2:
            bandwidth_param = float(parts[0])  # 带宽参数
            center_freq_param = float(parts[1])  # 中心频率参数
            Fc = center_freq_param / (2 * np.pi)
        else:
            bandwidth_param = 3.0
            center_freq_param = 3.0
            Fc = 1.0
    else:
        try:
            Fc = pywt.central_frequency(wavename)
            bandwidth_param = 3.0
            center_freq_param = 3.0
        except:
            Fc = 1.0
            bandwidth_param = 3.0
            center_freq_param = 3.0
    
    print(f'  小波类型: {wavename}')
    if wavename.startswith('cmor'):
        print(f'  带宽参数: {bandwidth_param}, 中心频率参数: {center_freq_param}')
        print(f'  说明: 带宽参数越小，时间分辨率越好（适合高频瞬态信号）')
        print(f'        带宽参数越大，频率分辨率越好（适合频率分析）')
        if bandwidth_param >= 2.5:
            print(f'  建议: 对于高频分析（100kHz-10MHz），考虑使用cmor1.5-1或cmor2-1以获得更好的时间分辨率')
    print(f'  小波中心频率: {Fc:.4f}')
    print(f'  总尺度数: {totalscal}')
    
    # 计算尺度范围（基于原始采样率）
    max_scale = (Fc * original_sampling_rate) / min_freq
    min_scale = (Fc * original_sampling_rate) / max_freq
    scales = np.logspace(np.log10(min_scale), np.log10(max_scale), totalscal)
    frequencies = (Fc * original_sampling_rate) / scales
    
    print(f'  尺度范围: {scales[0]:.2f} - {scales[-1]:.2f}')
    print(f'  对应频率范围: {frequencies[-1]/1e3:.1f}kHz - {frequencies[0]/1e6:.2f}MHz')
    
    # 4. 对每个event独立进行小波变换
    print(f'\n正在对每个event独立进行小波变换...')
    
    def compute_single_event_cwt(event_idx, time_samples):
        """计算单个event的小波变换"""
        with h5py.File(ch0_3_file, 'r') as f_ch0:
            ch0_channel_data = f_ch0['channel_data']
            ch0_waveform = ch0_channel_data[:, ch0_idx, event_idx].astype(np.float64)
        
        if detrend:
            ch0_waveform = signal.detrend(ch0_waveform)
        
        # 计算CWT
        coeff, _ = pywt.cwt(ch0_waveform, scales, wavename, sampling_period=sampling_interval_s)
        
        # 计算功率
        if np.iscomplexobj(coeff):
            power = np.abs(coeff) ** 2
        else:
            power = coeff ** 2
        
        return power
    
    # 读取文件获取时间样本数
    with h5py.File(ch0_3_file, 'r') as f_ch0:
        ch0_channel_data = f_ch0['channel_data']
        time_samples = ch0_channel_data.shape[0]
    
    # 并行计算每个event的小波变换
    n_jobs = min(mp.cpu_count(), num_events)
    
    if n_jobs > 1 and num_events > 1:
        print(f'使用 {n_jobs} 个CPU核心并行计算...')
        all_power_list = Parallel(n_jobs=n_jobs, backend='threading', verbose=1)(
            delayed(compute_single_event_cwt)(event_idx, time_samples) 
            for event_idx in selected_indices
        )
    else:
        all_power_list = []
        for i, event_idx in enumerate(selected_indices):
            power = compute_single_event_cwt(event_idx, time_samples)
            all_power_list.append(power)
            if (i + 1) % 10 == 0:
                print(f'  已处理 {i+1}/{num_events} 个events')
    
    # 转换为numpy数组
    all_power_array = np.array(all_power_list)  # shape: (num_events, scales, time_samples)
    
    print(f'\n所有event的小波变换完成')
    print(f'  功率谱矩阵形状: {all_power_array.shape}')
    
    # 5. 统计平均（降低方差）
    mean_power = np.mean(all_power_array, axis=0)  # 平均所有events
    std_power = np.std(all_power_array, axis=0)    # 标准差
    median_power = np.median(all_power_array, axis=0)  # 中位数
    
    print(f'  平均功率谱形状: {mean_power.shape}')
    
    # 计算主导频率
    mean_power_per_scale = np.mean(mean_power, axis=1)
    dominant_scale_idx = np.argmax(mean_power_per_scale)
    dominant_freq = frequencies[dominant_scale_idx]
    
    print(f'\n主导频率: {dominant_freq/1e6:.4f}MHz')
    
    # 统计分析功率谱
    print(f'\n功率谱统计:')
    print(f'  平均功率范围: {np.min(mean_power):.2e} - {np.max(mean_power):.2e}')
    print(f'  平均功率 (整体): {np.mean(mean_power):.2e}')
    print(f'  功率非零区域比例: {np.sum(mean_power > np.max(mean_power) * 0.01) / mean_power.size * 100:.2f}%')
    
    # 检查是否有边界效应
    time_axis_us = np.arange(time_samples) * sampling_interval_ns / 1000.0
    edge_window = int(time_samples * 0.1)  # 前后10%作为边界
    central_power = mean_power[:, edge_window:-edge_window] if edge_window > 0 else mean_power
    edge_power_left = mean_power[:, :edge_window] if edge_window > 0 else mean_power[:, :1]
    edge_power_right = mean_power[:, -edge_window:] if edge_window > 0 else mean_power[:, -1:]
    
    print(f'  中心区域平均功率: {np.mean(central_power):.2e}')
    print(f'  左边界平均功率: {np.mean(edge_power_left):.2e}')
    print(f'  右边界平均功率: {np.mean(edge_power_right):.2e}')
    
    # 6. 可视化（时频图和FFT对比）
    if show_plot or save_path:
        # 限制显示的频率范围
        freq_mask = (frequencies >= min_freq) & (frequencies <= max_freq)
        frequencies_display = frequencies[freq_mask]
        mean_power_display = mean_power[freq_mask, :]
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # 1. 时频图（使用对数显示功率，避免被大值掩盖）
        ax = axes[0]
        # 避免log(0)，添加小的偏移
        log_power = np.log10(mean_power_display + 1e-20)
        
        im = ax.contourf(time_axis_us, frequencies_display / 1e6, log_power, 
                        levels=50, cmap='jet')
        ax.set_xlabel('Time (μs)', fontsize=12)
        ax.set_ylabel('Frequency (MHz)', fontsize=12)
        ax.set_title(f'Wavelet Time-Frequency Analysis (RT Only, Non-Inhibit) - Log Scale\n'
                    f'{num_events} events averaged, Frequency: {min_freq/1e3:.0f}kHz - {max_freq/1e6:.1f}MHz',
                    fontsize=13)
        ax.set_ylim(min_freq / 1e6, max_freq / 1e6)
        plt.colorbar(im, ax=ax, label='Log10(Power)')
        ax.grid(True, alpha=0.3)
        
        # 2. 平均功率谱（FFT对比，用于验证频率内容）
        ax = axes[1]
        
        # 计算一个示例event的FFT功率谱（用于对比）
        with h5py.File(ch0_3_file, 'r') as f_ch0:
            ch0_channel_data = f_ch0['channel_data']
            sample_waveform = ch0_channel_data[:, ch0_idx, selected_indices[0]].astype(np.float64)
        
        if detrend:
            sample_waveform = signal.detrend(sample_waveform)
        
        # FFT
        fft_freqs = np.fft.fftfreq(len(sample_waveform), sampling_interval_s)
        fft_spectrum = np.fft.fft(sample_waveform)
        fft_power = np.abs(fft_spectrum) ** 2
        
        # 只显示正频率和感兴趣的频率范围
        pos_freq_mask = (fft_freqs > 0) & (fft_freqs >= min_freq) & (fft_freqs <= max_freq)
        fft_freqs_display = fft_freqs[pos_freq_mask]
        fft_power_display = fft_power[pos_freq_mask]
        
        ax.loglog(fft_freqs_display, fft_power_display, 'b-', linewidth=1.5, alpha=0.7, label='FFT Power Spectrum')
        
        # 小波平均功率谱（按频率平均）
        wav_power_per_freq = np.mean(mean_power_display, axis=1)
        ax.loglog(frequencies_display, wav_power_per_freq, 'r-', linewidth=1.5, alpha=0.7, label='Wavelet Average Power')
        
        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel('Power', fontsize=12)
        ax.set_title(f'Power Spectrum Comparison (FFT vs Wavelet Average)\n'
                    f'Frequency Range: {min_freq/1e3:.0f}kHz - {max_freq/1e6:.1f}MHz',
                    fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'图片已保存: {save_path}')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    return {
        'selected_event_indices': selected_indices,
        'num_events': num_events,
        'scales': scales,
        'frequencies': frequencies,
        'all_power_array': all_power_array,
        'mean_power': mean_power,
        'std_power': std_power,
        'median_power': median_power,
        'dominant_freq': dominant_freq,
        'time_samples': time_samples,
        'sampling_interval_s': sampling_interval_s
    }


if __name__ == '__main__':
    h5_files = get_h5_files()
    
    if 'CH0-3' in h5_files and h5_files['CH0-3'] and 'CH5' in h5_files and h5_files['CH5']:
        ch0_3_files = h5_files['CH0-3']
        ch5_files = h5_files['CH5']
        
        ch0_3_dict = {os.path.basename(f): f for f in ch0_3_files}
        ch5_dict = {os.path.basename(f): f for f in ch5_files}
        
        matched_pairs = []
        for filename in ch0_3_dict.keys():
            if filename in ch5_dict:
                matched_pairs.append((ch0_3_dict[filename], ch5_dict[filename]))
        
        if matched_pairs:
            ch0_3_file, ch5_file = matched_pairs[0]
            
            try:
                results = analyze_rt_only_events_wavelet(
                    ch0_3_file, ch5_file,
                    rt_cut=6000.0,
                    max_events=None,  # 分析所有符合条件的events
                    ch0_idx=0,
                    ch5_idx=0,
                    detrend=True,
                    wavename='cmor1.5-1',  # 高频分析推荐：较小的带宽参数以获得更好的时间分辨率
                    totalscal=128,  # 增加尺度数以提高频率分辨率
                    min_freq=100e3,  # 100 kHz
                    max_freq=10e6,   # 10 MHz
                    show_plot=True
                )
                print(f'\n分析完成: {results["num_events"]} 个events')
                print(f'主导频率: {results["dominant_freq"]/1e6:.4f}MHz')
            except Exception as e:
                print(f'分析失败: {e}')
                import traceback
                traceback.print_exc()
        else:
            print('未找到匹配的文件对')
    else:
        print('未找到CH0-3或CH5文件')

