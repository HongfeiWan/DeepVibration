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

def analyze_all_events_wavelet(h5_file: str = None,
                                channel_idx: int = 0,
                                detrend: bool = True,
                                uniform_sampling_rate: float = 1000.0,
                                wavename: str = 'cmor1.5-1',
                                totalscal: int = 128,
                                min_freq: float = 0.1,
                                max_freq: float = 200.0,
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
    with h5py.File(h5_file, 'r') as f:
        channel_data = f['channel_data']
        time_data = f['time_data']  # 每个event的触发时间
        time_samples, num_channels, num_events = channel_data.shape
        
        print(f'  数据维度: (时间点={time_samples}, 通道数={num_channels}, 事件数={num_events})')
        
        if channel_idx < 0 or channel_idx >= num_channels:
            raise IndexError(f'通道索引 {channel_idx} 超出范围 [0, {num_channels-1}]')
        
        # 并行处理所有event（收集信号点和时间戳）
        def process_single_event(event_idx):
            """处理单个event"""
            event_trigger_time = time_data[event_idx]
            ch0_waveform = channel_data[:, channel_idx, event_idx].astype(np.float64)
            
            if detrend:
                ch0_waveform = signal.detrend(ch0_waveform)
            
            event_times = event_trigger_time + np.arange(time_samples) * sampling_interval_s
            return ch0_waveform, event_times
        
        print(f'  使用 {mp.cpu_count()} 个CPU核心并行处理events...')
        n_jobs = mp.cpu_count()
        
        results = Parallel(n_jobs=n_jobs, backend='threading', verbose=1)(
            delayed(process_single_event)(i) for i in range(num_events)
        )
        
        # 合并结果
        all_signal_values = []
        all_time_stamps = []
        for waveform, times in results:
            all_signal_values.extend(waveform)
            all_time_stamps.extend(times)
        
        # 转换为numpy数组并排序（按时间）
        all_signal_values = np.array(all_signal_values)
        all_time_stamps = np.array(all_time_stamps)
        
        sort_indices = np.argsort(all_time_stamps)
        all_signal_values = all_signal_values[sort_indices]
        all_time_stamps = all_time_stamps[sort_indices]
        
        total_duration = all_time_stamps[-1] - all_time_stamps[0]
        
        print(f'\n拼接完成:')
        print(f'  总数据点数: {len(all_signal_values)}')
        print(f'  时间跨度: {total_duration:.6f} s ({total_duration/60:.2f} 分钟)')
        print(f'  起始时间: {all_time_stamps[0]:.6f} s')
        print(f'  结束时间: {all_time_stamps[-1]:.6f} s')
    
    # 3. 构建降采样后的信号（event之间补0）
    start_time = all_time_stamps[0]
    end_time = all_time_stamps[-1]
    
    # 计算降采样后的点数
    uniform_dt = 1.0 / uniform_sampling_rate
    num_uniform_points = int(total_duration * uniform_sampling_rate) + 1
    actual_sampling_rate = 1.0 / uniform_dt
    
    # 创建降采样后的时间网格
    uniform_times = start_time + np.arange(num_uniform_points) * uniform_dt
    uniform_signal = np.zeros(num_uniform_points, dtype=np.float64)
    
    # 计算容差：原始采样间隔的一半
    tolerance = sampling_interval_s / 2.0
    
    # 使用searchsorted快速查找（向量化操作）
    uniform_indices = np.searchsorted(uniform_times, all_time_stamps, side='left')
    
    # 并行映射原始数据点到降采样网格
    def map_points_to_grid_batch(batch_indices):
        """批量映射数据点到网格"""
        batch_signal = np.zeros(num_uniform_points, dtype=np.float64)
        for i in batch_indices:
            idx = uniform_indices[i]
            if idx < num_uniform_points:
                if abs(uniform_times[idx] - all_time_stamps[i]) < tolerance:
                    batch_signal[idx] = all_signal_values[i]
                elif idx > 0 and abs(uniform_times[idx - 1] - all_time_stamps[i]) < tolerance:
                    batch_signal[idx - 1] = all_signal_values[i]
        return batch_signal
    
    # 并行处理映射（分批处理）
    num_points = len(all_time_stamps)
    batch_size = max(10000, num_points // (mp.cpu_count() * 4))  # 每个核心处理多个批次
    batches = [list(range(i, min(i + batch_size, num_points))) 
               for i in range(0, num_points, batch_size)]
    
    if len(batches) > 1:
        print(f'  使用 {mp.cpu_count()} 个CPU核心并行映射数据点到网格...')
        print(f'  数据点数: {num_points}, 批次大小: {batch_size}, 批次数: {len(batches)}')
        
        batch_results = Parallel(n_jobs=mp.cpu_count(), backend='threading', verbose=0)(
            delayed(map_points_to_grid_batch)(batch) for batch in batches
        )
        
        # 合并批次结果
        for batch_signal in batch_results:
            uniform_signal += batch_signal
    else:
        # 如果只有一批，直接处理
        uniform_signal = map_points_to_grid_batch(batches[0])
    
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
    
    # 计算尺度范围
    max_scale = (Fc * actual_sampling_rate) / min_freq
    min_scale = (Fc * actual_sampling_rate) / max_freq
    scales = np.logspace(np.log10(min_scale), np.log10(max_scale), totalscal)
    frequencies = (Fc * actual_sampling_rate) / scales
    
    print(f'  尺度范围: {scales[0]:.2f} - {scales[-1]:.2f}')
    print(f'  对应频率范围: {frequencies[-1]:.3f} - {frequencies[0]:.3f} Hz')
    
    # 5. 并行计算小波变换（按尺度分块）
    n_jobs = min(mp.cpu_count(), totalscal)
    
    def compute_cwt_chunk(scales_chunk):
        """计算一组尺度的小波系数"""
        coeff_chunk, _ = pywt.cwt(uniform_signal, scales_chunk, wavename, sampling_period=uniform_dt)
        return coeff_chunk
    
    if n_jobs > 1 and totalscal > 8:
        chunk_size = max(1, totalscal // n_jobs)
        scale_chunks = [scales[i:i+chunk_size] for i in range(0, totalscal, chunk_size)]
        
        print(f'\n进行小波变换（使用 {n_jobs} 个CPU核心并行计算，{len(scale_chunks)} 个块）...')
        
        coeff_chunks = Parallel(n_jobs=n_jobs, backend='threading', verbose=0)(
            delayed(compute_cwt_chunk)(chunk) for chunk in scale_chunks
        )
        
        coefficients = np.vstack(coeff_chunks)
    else:
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
            show_plot=True
        )
        print(f'\n分析完成: {results["num_events"]} 个events')
        print(f'主导频率: {results["dominant_freq"]:.4f}Hz')
    except Exception as e:
        print(f'分析失败: {e}')
        import traceback
        traceback.print_exc()

