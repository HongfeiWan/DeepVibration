#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对既非RT也非Inhibit的信号进行小波变换分析
筛选条件：既非RT（CH5 max <= rt_cut）也非Inhibit（CH0 min != 0）
频率范围：100kHz - 25MHz
对每个event独立进行小波变换，然后统计平均降低方差
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

# 导入coincident模块的筛选函数
coincident_module_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "coincident")
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

# 模块级函数，用于并行处理（可被pickle）
def _compute_single_event_cwt(args):
    """
    计算单个event的小波变换（模块级函数，可被pickle）
    
    参数:
        args: 元组，包含 (ch0_3_file, event_idx, ch0_idx, scales, wavename, 
              sampling_interval_s, detrend)
    
    返回:
        功率谱矩阵 (scales, time_samples)
    """
    (ch0_3_file, event_idx, ch0_idx, scales, wavename, 
     sampling_interval_s, detrend) = args
    
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

def analyze_neither_rt_nor_inhibit_wavelet(ch0_3_file: str = None,
                                           ch5_file: str = None,
                                           rt_cut: float = 6000.0,
                                           ch0_idx: int = 0,
                                           ch5_idx: int = 0,
                                           max_events: Optional[int] = None,
                                           detrend: bool = True,
                                           wavename: str = 'cmor3-3',
                                           totalscal: int = 128,
                                           min_freq: float = 100e3,  # 100 kHz
                                           max_freq: float = 25e6,   # 25 MHz
                                           save_path: Optional[str] = None,
                                           show_plot: bool = True) -> Dict:
    """
    筛选既非RT也非Inhibit的信号，对每个event独立进行小波变换，然后统计平均
    参数:
        ch0_3_file: CH0-3文件路径，如果为None则自动获取
        ch5_file: CH5文件路径，如果为None则自动获取
        rt_cut: RT信号截断阈值（CH5最大值 > rt_cut 为RT信号）
        ch0_idx: CH0通道索引
        ch5_idx: CH5通道索引
        max_events: 最多分析的event数量，None表示分析所有
        detrend: 是否去趋势
        wavename: 小波名称
        totalscal: 总尺度数
        min_freq: 最小频率（Hz），默认100kHz
        max_freq: 最大频率（Hz），默认25MHz
        save_path: 保存图片路径
        show_plot: 是否显示图片
    
    返回:
        包含分析结果的字典
    """
    print('=' * 70)
    print('对既非RT也非Inhibit的信号进行小波变换分析')
    print('=' * 70)
    print(f'目标频率范围: {min_freq/1e3:.1f}kHz - {max_freq/1e6:.2f}MHz')
    
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
    
    # 1. 筛选既非RT也非Inhibit的events
    print(f'\n正在筛选既非RT也非Inhibit的events...')
    
    if analyze_coincident_events is not None:
        stats = analyze_coincident_events(ch0_3_file, ch5_file, rt_cut, ch0_idx, ch5_idx, verbose=False)
        # 既非RT也非Inhibit：非RT 且 非Inhibit
        neither_mask = ~stats['rt_mask'] & ~stats['inhibit_mask']
        selected_indices = np.where(neither_mask)[0]
        trigger_times = None  # 稍后从文件读取
    else:
        # 如果没有导入模块，手动筛选
        print('  警告: 无法导入coincident模块，手动筛选...')
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
    
    num_events = len(selected_indices)
    print(f'  筛选完成: {num_events} 个既非RT也非Inhibit的events')
    
    if num_events == 0:
        print('  没有符合条件的events，无法进行分析')
        return {}
    
    # 限制分析的event数量
    if max_events is not None and max_events > 0:
        num_events = min(num_events, max_events)
        selected_indices = selected_indices[:num_events]
        print(f'  限制分析数量: {num_events} 个events')
    
    # 读取文件获取触发时间和参数
    with h5py.File(ch0_3_file, 'r') as f_ch0:
        ch0_channel_data = f_ch0['channel_data']
        ch0_time_data = f_ch0['time_data']
        time_samples, num_channels, total_events = ch0_channel_data.shape
        
        # 获取选中events的触发时间
        trigger_times = ch0_time_data[selected_indices]
        
        print(f'\n数据参数:')
        print(f'  总事件数: {total_events}')
        print(f'  选中事件数: {num_events}')
        print(f'  每个event采样点数: {time_samples}')
        print(f'  采样间隔: 4.0 ns')
    
    # 2. 小波变换参数设置
    sampling_interval_ns = 4.0
    sampling_interval_s = sampling_interval_ns * 1e-9
    original_sampling_rate = 1.0 / sampling_interval_s  # 250 MSPS
    
    # 小波参数
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
    
    # 计算尺度范围
    max_scale = (Fc * original_sampling_rate) / min_freq
    min_scale = (Fc * original_sampling_rate) / max_freq
    scales = np.logspace(np.log10(min_scale), np.log10(max_scale), totalscal)
    frequencies = (Fc * original_sampling_rate) / scales
    
    print(f'\n小波参数:')
    print(f'  小波类型: {wavename}')
    print(f'  小波中心频率: {Fc:.4f}')
    print(f'  总尺度数: {totalscal}')
    print(f'  尺度范围: {scales[0]:.2f} - {scales[-1]:.2f}')
    print(f'  对应频率范围: {frequencies[-1]/1e3:.1f}kHz - {frequencies[0]/1e6:.2f}MHz')
    
    # 3. 对每个event独立进行小波变换（小批次增量统计，避免内存溢出）
    print(f'\n正在对每个event独立进行小波变换（增量统计模式）...')
    
    # 估算单个event的功率谱矩阵大小
    single_event_memory_mb = (totalscal * time_samples * 8) / (1024 * 1024)
    print(f'  单个event功率谱矩阵大小: {single_event_memory_mb:.2f} MB')
    
    # 小批次处理：每批处理少量events，立即累加到统计量，然后释放
    # 批次大小根据内存情况调整：每批最多保存 batch_size 个功率谱矩阵
    # 例如：128 scales × 30000 samples × 8 = 30.7 MB per event
    # 每批20个events = 614 MB，可以接受
    batch_size = 20  # 每批处理20个events（平衡内存和并行效率）
    n_batches = (num_events + batch_size - 1) // batch_size
    
    print(f'  总events数: {num_events}')
    print(f'  如果全部保存需要: {single_event_memory_mb * num_events / 1024:.2f} GB')
    print(f'  使用增量统计模式，每批最多 {batch_size} 个events，峰值内存: ~{single_event_memory_mb * batch_size:.2f} MB')
    print(f'  分批处理: {n_batches} 个批次')
    
    # 初始化累加器（增量统计，不保存所有中间结果）
    sum_power = None  # 累加和
    sum_power_sq = None  # 平方和（用于计算标准差）
    count = 0  # 已处理的event数量
    
    # 用于计算中位数的样本（只保存少量样本用于近似中位数）
    # 或者使用在线算法，但为了简化，这里使用平均值近似中位数
    power_samples_for_median = []  # 可选：保存少量样本用于近似中位数
    max_samples_for_median = 100  # 最多保存100个样本用于近似中位数
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_events)
        batch_indices = selected_indices[start_idx:end_idx]
        batch_size_actual = len(batch_indices)
        
        print(f'  处理批次 {batch_idx + 1}/{n_batches} (events {start_idx+1}-{end_idx})...')
        
        # 准备该批次的参数
        batch_args_list = [
            (ch0_3_file, event_idx, ch0_idx, scales, wavename, 
             sampling_interval_s, detrend)
            for event_idx in batch_indices
        ]
        
        # 并行计算该批次
        n_jobs = min(mp.cpu_count(), batch_size_actual)
        
        if n_jobs > 1 and batch_size_actual > 1:
            batch_power_list = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
                delayed(_compute_single_event_cwt)(args) for args in batch_args_list
            )
        else:
            batch_power_list = [
                _compute_single_event_cwt(args) for args in batch_args_list
            ]
        
        # 转换为numpy数组并立即计算统计量（不长时间保存完整数组）
        batch_power_array = np.array(batch_power_list)  # shape: (batch_size, scales, time_samples)
        
        # 增量累加统计量（立即计算，避免保存完整批次数组）
        batch_sum = np.sum(batch_power_array, axis=0)  # shape: (scales, time_samples)
        batch_sum_sq = np.sum(batch_power_array ** 2, axis=0)
        
        # 立即释放批次数组
        del batch_power_array, batch_power_list
        
        if sum_power is None:
            # 初始化（使用第一个批次的形状）
            sum_power = batch_sum.copy()
            sum_power_sq = batch_sum_sq.copy()
        else:
            # 累加
            sum_power += batch_sum
            sum_power_sq += batch_sum_sq
        
        count += batch_size_actual
        
        # 释放其他临时变量
        del batch_sum, batch_sum_sq
    
    print(f'\n所有event的小波变换完成')
    
    # 4. 计算统计量（从累加和计算）
    mean_power = sum_power / count  # 平均值
    
    # 计算标准差: std = sqrt(mean(x^2) - mean(x)^2)
    mean_power_sq = sum_power_sq / count
    variance = mean_power_sq - mean_power ** 2
    # 避免负值（浮点数误差）
    variance = np.maximum(variance, 0)
    std_power = np.sqrt(variance)
    
    # 中位数：使用平均值近似（增量统计模式无法计算精确中位数）
    # 如果需要精确中位数，需要保存所有数据，但会占用大量内存
    median_power = mean_power
    print(f'  注意: 使用平均值近似中位数（增量统计模式，无法计算精确中位数）')
    print(f'  平均功率谱形状: {mean_power.shape}')
    
    # 释放累加器（不再需要）
    del sum_power, sum_power_sq
    
    # 计算每个频率的平均功率（跨时间维度平均）
    mean_power_per_freq = np.mean(mean_power, axis=1)  # shape: (scales,)
    std_power_per_freq = np.mean(std_power, axis=1)
    
    # 计算主导频率
    dominant_freq_idx = np.argmax(mean_power_per_freq)
    dominant_freq = frequencies[dominant_freq_idx]
    
    print(f'\n主导频率: {dominant_freq/1e6:.4f}MHz')
    
    # 统计分析功率谱
    print(f'\n功率谱统计:')
    print(f'  平均功率范围: {np.min(mean_power):.2e} - {np.max(mean_power):.2e}')
    print(f'  平均功率 (整体): {np.mean(mean_power):.2e}')
    print(f'  功率非零区域比例: {np.sum(mean_power > np.max(mean_power) * 0.01) / mean_power.size * 100:.2f}%')
    
    # 5. 可视化
    if show_plot or save_path:
        # 限制显示的频率范围
        freq_mask = (frequencies >= min_freq) & (frequencies <= max_freq)
        frequencies_display = frequencies[freq_mask]
        mean_power_display = mean_power[freq_mask, :]
        
        # 时间轴（微秒）
        time_axis_us = np.arange(time_samples) * sampling_interval_ns / 1000.0
        
        # 创建两个子图：时频图和平均频率谱
        fig, axes = plt.subplots(2, 1, figsize=(14, 12))
        
        # 子图1：平均时频图
        ax1 = axes[0]
        log_mean_power = np.log10(mean_power_display + 1e-20)
        im1 = ax1.contourf(time_axis_us, frequencies_display, log_mean_power, 
                         levels=50, cmap='jet')
        ax1.set_xlabel('Time (μs)', fontsize=12)
        ax1.set_ylabel('Frequency (Hz)', fontsize=12)
        ax1.set_title(f'Wavelet Time-Frequency Analysis (Neither RT nor Inhibit)\n'
                    f'{num_events} events averaged, Frequency: {min_freq/1e3:.1f}kHz - {max_freq/1e6:.2f}MHz, '
                    f'Dominant Freq: {dominant_freq/1e6:.4f}MHz',
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
                   label=f'Dominant: {dominant_freq/1e6:.4f}MHz')
        ax2.set_xlabel('Frequency (Hz)', fontsize=12)
        ax2.set_ylabel('Average Power', fontsize=12)
        ax2.set_title(f'Averaged Frequency Spectrum ({num_events} events averaged)',
                    fontsize=13)
        ax2.set_xlim(min_freq, max_freq)
        ax2.set_yscale('log')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'\n图片已保存: {save_path}')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    # 返回结果
    return {
        'num_events': num_events,
        'selected_indices': selected_indices.tolist(),
        'trigger_times': trigger_times.tolist(),
        'scales': scales,
        'frequencies': frequencies,
        'mean_power': mean_power,
        'std_power': std_power,
        'median_power': median_power,
        'mean_power_per_freq': mean_power_per_freq,
        'std_power_per_freq': std_power_per_freq,
        'dominant_freq': dominant_freq,
        'time_samples': time_samples,
        'sampling_interval_s': sampling_interval_s
    }

if __name__ == '__main__':
    try:
        results = analyze_neither_rt_nor_inhibit_wavelet(
            ch0_3_file=None,  # 自动选择匹配的文件对
            ch5_file=None,
            rt_cut=6000.0,
            ch0_idx=0,
            ch5_idx=0,
            max_events=None,  # 分析所有符合条件的events
            detrend=True,
            wavename='cmor3-3',
            totalscal=128,
            min_freq=100e3,  # 100 kHz
            max_freq=25e6,  # 25 MHz
            show_plot=True
        )
        
        if results:
            print(f'\n分析完成: {results["num_events"]} 个events')
            print(f'主导频率: {results["dominant_freq"]/1e6:.4f}MHz')
    except Exception as e:
        print(f'分析失败: {e}')
        import traceback
        traceback.print_exc()

