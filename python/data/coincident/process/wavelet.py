#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
筛选RT信号对应的CH0，拼接后做小波变换
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
                                   uniform_sampling_rate: float = 1000.0,
                                   wavename: str = 'cmor3-3',
                                   totalscal: int = 64,
                                   save_path: Optional[str] = None,
                                   show_plot: bool = True) -> Dict:
    """
    筛选RT信号对应的CH0，拼接后做小波变换
    
    参数:
        ch0_3_file: CH0-3文件路径
        ch5_file: CH5文件路径
        rt_cut: RT信号截断阈值
        ch0_idx: CH0通道索引
        ch5_idx: CH5通道索引
        max_events: 最多分析的event数量
        detrend: 是否去趋势
        uniform_sampling_rate: 均匀采样率（Hz）
        wavename: 小波名称
        totalscal: 总尺度数
        save_path: 保存图片路径
        show_plot: 是否显示图片
    
    返回:
        包含分析结果的字典
    """
    print('筛选RT信号对应的CH0，拼接后做小波变换...')
    
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
    
    # 2. 拼接CH0信号
    sampling_interval_ns = 4.0
    sampling_interval_s = sampling_interval_ns * 1e-9
    time_samples = 30000
    
    with h5py.File(ch0_3_file, 'r') as f_ch0:
        ch0_channel_data = f_ch0['channel_data']
        ch0_time_data = f_ch0['time_data']
        ch0_time_samples_actual, ch0_num_channels, ch0_num_events = ch0_channel_data.shape
        
        if ch0_time_samples_actual != time_samples:
            time_samples = ch0_time_samples_actual
        
        all_signal_values = []
        all_time_stamps = []
        
        for i, event_idx in enumerate(selected_indices):
            event_trigger_time = ch0_time_data[event_idx]
            ch0_waveform = ch0_channel_data[:, ch0_idx, event_idx].astype(np.float64)
            
            if detrend:
                ch0_waveform = signal.detrend(ch0_waveform)
            
            event_times = event_trigger_time + np.arange(time_samples) * sampling_interval_s
            all_signal_values.extend(ch0_waveform)
            all_time_stamps.extend(event_times)
        
        all_signal_values = np.array(all_signal_values)
        all_time_stamps = np.array(all_time_stamps)
        
        sort_indices = np.argsort(all_time_stamps)
        all_signal_values = all_signal_values[sort_indices]
        all_time_stamps = all_time_stamps[sort_indices]
        
        total_duration = all_time_stamps[-1] - all_time_stamps[0]
        print(f'拼接完成: {len(all_signal_values)} 个点, 时间跨度 {total_duration:.2f}s')
    
    # 3. 直接构建降采样后的信号（避免创建巨大的连续数组）
    start_time = all_time_stamps[0]
    end_time = all_time_stamps[-1]
    total_duration = end_time - start_time
    
    # 计算降采样后的点数
    uniform_dt = 1.0 / uniform_sampling_rate
    num_uniform_points = int(total_duration * uniform_sampling_rate) + 1
    actual_sampling_rate = 1.0 / uniform_dt
    
    # 创建降采样后的时间网格
    uniform_times = start_time + np.arange(num_uniform_points) * uniform_dt
    uniform_signal = np.zeros(num_uniform_points, dtype=np.float64)
    
    # 计算容差：原始采样间隔的一半
    tolerance = sampling_interval_s / 2.0
    
    # 对于每个原始数据点，找到它在降采样网格中的位置
    # 使用searchsorted快速查找
    uniform_indices = np.searchsorted(uniform_times, all_time_stamps, side='left')
    
    # 将原始值填入降采样网格
    for i, idx in enumerate(uniform_indices):
        if idx < num_uniform_points:
            # 检查当前点
            if abs(uniform_times[idx] - all_time_stamps[i]) < tolerance:
                uniform_signal[idx] = all_signal_values[i]
            # 检查前一个点（可能更接近）
            elif idx > 0 and abs(uniform_times[idx - 1] - all_time_stamps[i]) < tolerance:
                uniform_signal[idx - 1] = all_signal_values[i]
    
    print(f'降采样完成: {num_uniform_points} 个点, 采样率 {actual_sampling_rate:.1f}Hz')
    
    # 4. 小波变换
    if wavename.startswith('cmor'):
        parts = wavename.split('-')
        if len(parts) == 2:
            Fc = float(parts[1]) / (2 * np.pi)
        else:
            Fc = 1.0
    else:
        try:
            Fc = pywt.central_frequency(wavename)
        except:
            Fc = 1.0
    
    target_min_freq = 0.1
    target_max_freq = 1000.0
    
    max_scale = (Fc * actual_sampling_rate) / target_min_freq
    min_scale = (Fc * actual_sampling_rate) / target_max_freq
    scales = np.logspace(np.log10(min_scale), np.log10(max_scale), totalscal)
    
    # 并行计算小波变换（按尺度分块）
    n_jobs = min(mp.cpu_count(), totalscal)  # 不超过尺度数
    
    def compute_cwt_chunk(scales_chunk):
        """计算一组尺度的小波系数"""
        coeff_chunk, _ = pywt.cwt(uniform_signal, scales_chunk, wavename, sampling_period=uniform_dt)
        return coeff_chunk
    
    # 将尺度分成多个块，并行处理
    if n_jobs > 1 and totalscal > 8:  # 只有当尺度数足够多时才并行
        chunk_size = max(1, totalscal // n_jobs)
        scale_chunks = [scales[i:i+chunk_size] for i in range(0, totalscal, chunk_size)]
        
        print(f'进行小波变换（使用 {n_jobs} 个CPU核心并行计算，{len(scale_chunks)} 个块）...')
        
        # 并行计算每个块
        coeff_chunks = Parallel(n_jobs=n_jobs, backend='threading', verbose=0)(
            delayed(compute_cwt_chunk)(chunk) for chunk in scale_chunks
        )

        # 合并结果
        coefficients = np.vstack(coeff_chunks)
    else:
        # 单核直接计算
        print(f'进行小波变换...')
        coefficients, _ = pywt.cwt(uniform_signal, scales, wavename, sampling_period=uniform_dt)
    
    # 计算对应的频率
    frequencies = (Fc * actual_sampling_rate) / scales

    if np.iscomplexobj(coefficients):
        power = np.abs(coefficients) ** 2
    else:
        power = coefficients ** 2

    mean_power_per_scale = np.mean(power, axis=1)
    dominant_freq = frequencies[np.argmax(mean_power_per_scale)]
    
    print(f'小波变换完成，主导频率: {dominant_freq:.4f}Hz')
    
    # 5. 可视化（只显示时频谱，限制频率≤1000Hz，时间≤20s）
    if show_plot or save_path:
        # 限制频率范围（≤1000Hz）
        freq_mask = frequencies <= 1000.0
        power_filtered = power[freq_mask, :]
        frequencies_filtered = frequencies[freq_mask]
        
        # 限制时间范围（≤20s）
        max_time_seconds = 20.0
        max_time_points = int(max_time_seconds * actual_sampling_rate) + 1
        max_time_points = min(max_time_points, len(uniform_times))
        
        time_display = uniform_times[:max_time_points] - uniform_times[0]
        power_display = power_filtered[:, :max_time_points]
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        im = ax.contourf(time_display, frequencies_filtered, power_display, levels=50, cmap='jet')
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Frequency (Hz)', fontsize=12)
        ax.set_title(f'Wavelet Time-Frequency Analysis ({num_events} events)\n'
                    f'Frequency: 0-1000Hz, Time: ≤{max_time_seconds}s', fontsize=13)
        ax.set_ylim(0, 1000)
        plt.colorbar(im, ax=ax, label='Power')
        ax.grid(True, alpha=0.3)
        
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
        'uniform_times': uniform_times,
        'uniform_signal': uniform_signal,
        'scales': scales,
        'frequencies': frequencies,
        'power': power,
        'dominant_freq': dominant_freq
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
                    max_events=None,
                    ch0_idx=0,
                    ch5_idx=0,
                    detrend=True,
                    uniform_sampling_rate=1000.0,
                    wavename='cmor3-3',
                    totalscal=64,
                    show_plot=True
                )
                print(f'\n分析完成: {results["num_events"]} 个events, 主导频率 {results["dominant_freq"]:.4f}Hz')
            except Exception as e:
                print(f'分析失败: {e}')
                import traceback
                traceback.print_exc()
        else:
            print('未找到匹配的文件对')
    else:
        print('未找到CH0-3或CH5文件')
