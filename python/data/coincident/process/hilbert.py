#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对筛选出的RT且非Inhibit的events的CH0信号进行Hilbert变换分析
提取包络和瞬时相位，分析包络统计特性和相位一致性
"""
import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import hilbert
from typing import Optional, Tuple, Dict, List

# 添加路径以便导入 utils 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
python_dir = os.path.dirname(os.path.dirname(current_dir))
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from utils.visualize import get_h5_files

# 导入coincident模块的筛选函数
coincident_module_path = os.path.join(python_dir, 'python', 'data', 'coincident')
randomtrigger_inhibit_file = os.path.join(coincident_module_path, "randomtrigger&inhibit.py")

# 尝试导入analyze_coincident_events函数
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
        analyze_coincident_events = None


def analyze_rt_only_events_hilbert(ch0_3_file: str,
                                     ch5_file: str,
                                     rt_cut: float = 6000.0,
                                     ch0_idx: int = 0,
                                     ch5_idx: int = 0,
                                     max_events: Optional[int] = None,
                                     detrend: bool = True,
                                     save_path: Optional[str] = None,
                                     show_plot: bool = True,
                                     figsize: Tuple[int, int] = (18, 14)) -> Dict:
    """
    对筛选出的RT且非Inhibit的events的CH0信号进行Hilbert变换分析
    
    参数:
        ch0_3_file: CH0-3文件路径
        ch5_file: CH5文件路径
        rt_cut: RT信号的截断阈值
        ch0_idx: CH0通道索引
        ch5_idx: CH5通道索引
        max_events: 最多分析的event数量，如果为None则分析所有符合条件的events
        detrend: 是否去趋势
        save_path: 保存图片路径
        show_plot: 是否显示图片
        figsize: 图片大小
    
    返回:
        包含分析结果的字典
    """
    print('=' * 70)
    print(f'Hilbert变换分析：RT且非Inhibit的events的CH0信号')
    print('=' * 70)
    
    # 1. 筛选符合条件的events
    if analyze_coincident_events is not None:
        # 使用analyze_coincident_events函数获取筛选结果
        stats = analyze_coincident_events(
            ch0_3_file, ch5_file, rt_cut, ch0_idx, ch5_idx, verbose=True
        )
        # 仅RT信号（非Inhibit）：RT信号且非Inhibit信号
        rt_only_mask = stats['rt_mask'] & ~stats['inhibit_mask']
        selected_indices = np.where(rt_only_mask)[0]
    else:
        # 如果无法导入，直接实现筛选逻辑
        print('使用内置筛选逻辑...')
        batch_size = 1000
        with h5py.File(ch0_3_file, 'r') as f_ch0:
            ch0_channel_data = f_ch0['channel_data']
            ch0_time_samples, ch0_num_channels, ch0_num_events = ch0_channel_data.shape
            ch0_min_values = np.zeros(ch0_num_events, dtype=np.float64)
            for i in range(0, ch0_num_events, batch_size):
                end_idx = min(i + batch_size, ch0_num_events)
                batch_data = ch0_channel_data[:, ch0_idx, i:end_idx]
                batch_min = np.min(batch_data, axis=0)
                ch0_min_values[i:end_idx] = batch_min
        
        with h5py.File(ch5_file, 'r') as f_ch5:
            ch5_channel_data = f_ch5['channel_data']
            ch5_time_samples, ch5_num_channels, ch5_num_events = ch5_channel_data.shape
            ch5_max_values = np.zeros(ch5_num_events, dtype=np.float64)
            for i in range(0, ch5_num_events, batch_size):
                end_idx = min(i + batch_size, ch5_num_events)
                batch_data = ch5_channel_data[:, ch5_idx, i:end_idx]
                batch_max = np.max(batch_data, axis=0)
                ch5_max_values[i:end_idx] = batch_max
        
        rt_mask = ch5_max_values > rt_cut
        non_inhibit_mask = ch0_min_values > 0
        selected_mask = rt_mask & non_inhibit_mask
        selected_indices = np.where(selected_mask)[0]
        
        print(f'筛选结果:')
        print(f'  总事件数: {ch0_num_events}')
        print(f'  RT信号数: {np.sum(rt_mask)}')
        print(f'  非Inhibit信号数: {np.sum(non_inhibit_mask)}')
        print(f'  RT且非Inhibit信号数: {len(selected_indices)}')
    
    if len(selected_indices) == 0:
        raise ValueError('没有符合条件的events（RT且非Inhibit）')
    
    # 2. 限制使用的event数量
    if max_events is not None and len(selected_indices) > max_events:
        print(f'\n限制使用前 {max_events} 个符合条件的events')
        selected_indices = selected_indices[:max_events]
    
    num_events = len(selected_indices)
    print(f'\n将分析 {num_events} 个events的CH0信号')
    
    # 3. 参数设置
    sampling_interval_ns = 4.0  # 4ns per sample
    sampling_interval_s = sampling_interval_ns * 1e-9  # 转换为秒
    time_samples = 30000  # 每个event的采样点数
    time_axis_us = np.arange(time_samples) * sampling_interval_ns / 1000.0  # 微秒
    
    print(f'\n信号参数:')
    print(f'  采样点数: {time_samples}')
    print(f'  采样间隔: {sampling_interval_ns} ns')
    print(f'  Event时长: {time_samples * sampling_interval_s * 1e6:.2f} μs')
    
    # 4. 读取文件并进行Hilbert变换分析
    print(f'\n正在进行Hilbert变换分析...')
    
    all_envelopes = []  # 存储所有event的包络
    all_phases = []  # 存储所有event的瞬时相位
    all_instantaneous_freqs = []  # 存储所有event的瞬时频率
    
    with h5py.File(ch0_3_file, 'r') as f_ch0:
        ch0_channel_data = f_ch0['channel_data']
        ch0_time_samples_actual, ch0_num_channels, ch0_num_events = ch0_channel_data.shape
        
        if ch0_time_samples_actual != time_samples:
            print(f'警告: 实际采样点数 ({ch0_time_samples_actual}) 与预期 ({time_samples}) 不一致')
            time_samples = ch0_time_samples_actual
            time_axis_us = np.arange(time_samples) * sampling_interval_ns / 1000.0
        
        # 对每个event进行Hilbert变换
        for i, event_idx in enumerate(selected_indices):
            # 获取CH0的波形数据
            ch0_waveform = ch0_channel_data[:, ch0_idx, event_idx].astype(np.float64)
            
            # 去趋势
            if detrend:
                ch0_waveform = signal.detrend(ch0_waveform)
            
            # Hilbert变换
            analytic_signal = hilbert(ch0_waveform)
            
            # 提取包络（瞬时幅度）
            envelope = np.abs(analytic_signal)
            
            # 提取瞬时相位
            phase = np.unwrap(np.angle(analytic_signal))
            
            # 计算瞬时频率（相位的导数）
            instantaneous_freq = np.diff(phase) / (2.0 * np.pi * sampling_interval_s)
            # 补一个点以保持长度一致（使用最后一个值）
            instantaneous_freq = np.append(instantaneous_freq, instantaneous_freq[-1])
            
            all_envelopes.append(envelope)
            all_phases.append(phase)
            all_instantaneous_freqs.append(instantaneous_freq)
            
            if (i + 1) % 10 == 0 or i == num_events - 1:
                print(f'  已处理 {i+1}/{num_events} 个events')
    
    # 转换为numpy数组
    all_envelopes = np.array(all_envelopes)  # 形状: (num_events, time_samples)
    all_phases = np.array(all_phases)  # 形状: (num_events, time_samples)
    all_instantaneous_freqs = np.array(all_instantaneous_freqs)  # 形状: (num_events, time_samples)
    
    print(f'\nHilbert变换分析完成')
    print(f'  分析的events数: {num_events}')
    
    # 5. 包络统计分析
    # 对每个event的包络计算统计量
    envelope_means = np.mean(all_envelopes, axis=1)  # 每个event的包络均值
    envelope_peaks = np.max(all_envelopes, axis=1)  # 每个event的包络峰值
    envelope_vars = np.var(all_envelopes, axis=1)  # 每个event的包络方差
    envelope_stds = np.std(all_envelopes, axis=1)  # 每个event的包络标准差
    
    # 整体统计
    overall_envelope_mean = np.mean(all_envelopes)
    overall_envelope_peak = np.max(all_envelopes)
    overall_envelope_var = np.var(all_envelopes)
    overall_envelope_std = np.std(all_envelopes)
    
    # 时间平均的包络（所有event在相同时间点的平均值）
    mean_envelope_over_time = np.mean(all_envelopes, axis=0)
    std_envelope_over_time = np.std(all_envelopes, axis=0)
    
    print(f'\n包络统计结果:')
    print(f'  整体包络均值: {overall_envelope_mean:.2f}')
    print(f'  整体包络峰值: {overall_envelope_peak:.2f}')
    print(f'  整体包络方差: {overall_envelope_var:.2e}')
    print(f'  整体包络标准差: {overall_envelope_std:.2f}')
    print(f'\n各Event包络统计:')
    print(f'  包络均值 - 平均: {np.mean(envelope_means):.2f}, 标准差: {np.std(envelope_means):.2f}')
    print(f'  包络峰值 - 平均: {np.mean(envelope_peaks):.2f}, 标准差: {np.std(envelope_peaks):.2f}')
    print(f'  包络方差 - 平均: {np.mean(envelope_vars):.2e}, 标准差: {np.std(envelope_vars):.2e}')
    
    # 6. 瞬时相位一致性分析
    # 相位一致性可以通过多种方式衡量：
    # 1. 相位锁定值（Phase Locking Value, PLV）
    # 2. 相位差的标准差
    # 3. 相位分布的集中度（circular statistics）
    # 4. 包络内的相位一致性（在不同包络幅度下的相位分布）
    
    # 计算相位锁定值（PLV）
    # PLV = |mean(exp(i*phase))|，衡量相位同步程度
    complex_phases = np.exp(1j * all_phases)  # 转换为复数
    plv_time = np.abs(np.mean(complex_phases, axis=0))  # 时间点的PLV（0-1，1表示完全同步）
    mean_plv = np.mean(plv_time)
    
    # 计算相位差的标准差（在相同时间点）
    phase_std_over_time = np.std(all_phases, axis=0)
    mean_phase_std = np.mean(phase_std_over_time)
    
    # 计算瞬时频率的一致性（标准差）
    inst_freq_std_over_time = np.std(all_instantaneous_freqs, axis=0)
    mean_inst_freq_std = np.mean(inst_freq_std_over_time)
    
    # 7. 包络内的瞬时相位一致性分析
    # 分析在不同包络幅度下的相位分布和一致性
    print(f'\n分析包络内的瞬时相位一致性...')
    
    # 方法1: 按包络幅度分bin，分析每个bin内的相位一致性
    # 将包络值归一化到0-1，然后分成若干个bin
    num_envelope_bins = 20
    envelope_percentiles = np.linspace(0, 100, num_envelope_bins + 1)
    
    # 对每个event，计算其包络的百分位数阈值
    envelope_bin_plv = []  # 每个bin的PLV
    envelope_bin_phase_std = []  # 每个bin的相位标准差
    envelope_bin_counts = []  # 每个bin的样本数
    envelope_bin_centers = []  # 每个bin的中心包络值
    
    for bin_idx in range(num_envelope_bins):
        # 找到这个bin的包络范围（使用所有events的包络值）
        lower_percentile = envelope_percentiles[bin_idx]
        upper_percentile = envelope_percentiles[bin_idx + 1]
        lower_threshold = np.percentile(all_envelopes, lower_percentile)
        upper_threshold = np.percentile(all_envelopes, upper_percentile)
        bin_center = (lower_threshold + upper_threshold) / 2
        
        # 对每个event，找出属于这个bin的时间点
        bin_phases = []  # 存储这个bin内的所有相位值
        bin_count = 0
        
        for event_idx in range(num_events):
            # 找出这个event中包络值在这个范围内的时间点
            mask = (all_envelopes[event_idx] >= lower_threshold) & (all_envelopes[event_idx] < upper_threshold)
            if bin_idx == num_envelope_bins - 1:  # 最后一个bin包含上界
                mask = all_envelopes[event_idx] >= lower_threshold
            
            if np.any(mask):
                bin_phases.append(all_phases[event_idx, mask])
                bin_count += np.sum(mask)
        
        if len(bin_phases) > 0:
            # 合并所有events中这个bin的相位
            all_bin_phases = np.concatenate(bin_phases)
            
            # 计算这个bin的相位一致性
            # 使用圆形统计量：平均相位向量长度（类似于PLV）
            complex_bin_phases = np.exp(1j * all_bin_phases)
            bin_plv = np.abs(np.mean(complex_bin_phases))
            
            # 计算相位标准差（需要unwrap处理）
            bin_phase_std = np.std(all_bin_phases)
            
            envelope_bin_plv.append(bin_plv)
            envelope_bin_phase_std.append(bin_phase_std)
            envelope_bin_counts.append(bin_count)
            envelope_bin_centers.append(bin_center)
        else:
            envelope_bin_plv.append(0.0)
            envelope_bin_phase_std.append(0.0)
            envelope_bin_counts.append(0)
            envelope_bin_centers.append(bin_center)
    
    envelope_bin_plv = np.array(envelope_bin_plv)
    envelope_bin_phase_std = np.array(envelope_bin_phase_std)
    envelope_bin_counts = np.array(envelope_bin_counts)
    envelope_bin_centers = np.array(envelope_bin_centers)
    
    # 方法2: 分析包络峰值附近的相位一致性
    # 对每个event，找到包络峰值的位置，然后分析峰值附近（±10%）时间窗口内的相位一致性
    peak_window_ratio = 0.1  # 峰值前后10%的时间窗口
    peak_window_samples = int(time_samples * peak_window_ratio)
    
    peak_nearby_phases = []  # 存储所有events峰值附近的相位
    
    for event_idx in range(num_events):
        # 找到包络峰值的位置
        peak_idx = np.argmax(all_envelopes[event_idx])
        
        # 确定窗口范围
        window_start = max(0, peak_idx - peak_window_samples)
        window_end = min(time_samples, peak_idx + peak_window_samples)
        
        # 提取峰值附近的相位
        peak_nearby_phases.append(all_phases[event_idx, window_start:window_end])
    
    # 合并所有events峰值附近的相位，计算一致性
    if len(peak_nearby_phases) > 0:
        all_peak_nearby_phases = np.concatenate(peak_nearby_phases)
        complex_peak_phases = np.exp(1j * all_peak_nearby_phases)
        peak_plv = np.abs(np.mean(complex_peak_phases))
        peak_phase_std = np.std(all_peak_nearby_phases)
    else:
        peak_plv = 0.0
        peak_phase_std = 0.0
    
    # 方法3: 包络上升沿和下降沿的相位分析
    # 对每个event，找到包络的前半部分（上升沿）和后半部分（下降沿）的相位一致性
    rising_edge_phases = []
    falling_edge_phases = []
    
    midpoint = time_samples // 2
    
    for event_idx in range(num_events):
        rising_edge_phases.append(all_phases[event_idx, :midpoint])
        falling_edge_phases.append(all_phases[event_idx, midpoint:])
    
    if len(rising_edge_phases) > 0:
        all_rising_phases = np.concatenate(rising_edge_phases)
        all_falling_phases = np.concatenate(falling_edge_phases)
        
        complex_rising = np.exp(1j * all_rising_phases)
        complex_falling = np.exp(1j * all_falling_phases)
        
        rising_plv = np.abs(np.mean(complex_rising))
        falling_plv = np.abs(np.mean(complex_falling))
        rising_phase_std = np.std(all_rising_phases)
        falling_phase_std = np.std(all_falling_phases)
    else:
        rising_plv = 0.0
        falling_plv = 0.0
        rising_phase_std = 0.0
        falling_phase_std = 0.0
    
    print(f'\n瞬时相位一致性分析:')
    print(f'  平均相位锁定值 (PLV): {mean_plv:.4f} (1.0=完全同步, 0.0=完全随机)')
    print(f'  PLV最小值: {np.min(plv_time):.4f}')
    print(f'  PLV最大值: {np.max(plv_time):.4f}')
    print(f'  平均相位标准差: {mean_phase_std:.4f} rad')
    print(f'  平均瞬时频率标准差: {mean_inst_freq_std/1e6:.3f} MHz')
    
    # 计算相位一致性的时间稳定性（PLV的时间变化）
    plv_var = np.var(plv_time)
    print(f'  PLV时间变异性: {plv_var:.6f} (越小表示相位一致性越稳定)')
    
    print(f'\n包络内的相位一致性分析:')
    valid_bins = envelope_bin_counts > 0
    if np.any(valid_bins):
        print(f'  包络峰值附近的PLV: {peak_plv:.4f}')
        print(f'  包络峰值附近的相位标准差: {peak_phase_std:.4f} rad')
        print(f'  上升沿PLV: {rising_plv:.4f}, 相位标准差: {rising_phase_std:.4f} rad')
        print(f'  下降沿PLV: {falling_plv:.4f}, 相位标准差: {falling_phase_std:.4f} rad')
        print(f'  高包络值bin的平均PLV: {np.mean(envelope_bin_plv[valid_bins][-5:]):.4f} (top 5 bins)')
        print(f'  低包络值bin的平均PLV: {np.mean(envelope_bin_plv[valid_bins][:5]):.4f} (bottom 5 bins)')
    
    # 8. 绘制结果
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(5, 3, hspace=0.3, wspace=0.3)
    
    # 1. 几个代表性event的原始信号和包络
    ax = fig.add_subplot(gs[0, 0])
    num_show = min(5, num_events)
    for i in range(num_show):
        event_idx = selected_indices[i]
        with h5py.File(ch0_3_file, 'r') as f_ch0:
            ch0_waveform = f_ch0['channel_data'][:, ch0_idx, event_idx].astype(np.float64)
            if detrend:
                ch0_waveform = signal.detrend(ch0_waveform)
            ax.plot(time_axis_us[:5000], ch0_waveform[:5000], 
                   alpha=0.6, linewidth=0.8, label=f'Event {event_idx}' if i < 3 else '')
            ax.plot(time_axis_us[:5000], all_envelopes[i, :5000], 
                   'r--', linewidth=1.5, alpha=0.8, label='Envelope' if i == 0 else '')
    ax.set_xlabel('Time (μs)', fontsize=9)
    ax.set_ylabel('Amplitude', fontsize=9)
    ax.set_title(f'Sample Events: Signal and Envelope\n(First 5000 samples)', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 2. 包络的时间平均和标准差
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(time_axis_us[:10000], mean_envelope_over_time[:10000], 
           'b-', linewidth=1.5, label='Mean Envelope')
    ax.fill_between(time_axis_us[:10000],
                    (mean_envelope_over_time - std_envelope_over_time)[:10000],
                    (mean_envelope_over_time + std_envelope_over_time)[:10000],
                    alpha=0.3, label='±1 Std')
    ax.set_xlabel('Time (μs)', fontsize=9)
    ax.set_ylabel('Envelope Amplitude', fontsize=9)
    ax.set_title(f'Average Envelope Over All Events\n(First 10000 samples)', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 3. 包络均值分布
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(envelope_means, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(np.mean(envelope_means), color='red', linestyle='--', linewidth=2, 
              label=f'Mean: {np.mean(envelope_means):.2f}')
    ax.set_xlabel('Envelope Mean', fontsize=9)
    ax.set_ylabel('Frequency', fontsize=9)
    ax.set_title(f'Distribution of Envelope Means\nAcross {num_events} Events', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 4. 包络峰值分布
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(envelope_peaks, bins=30, edgecolor='black', alpha=0.7, color='green')
    ax.axvline(np.mean(envelope_peaks), color='red', linestyle='--', linewidth=2, 
              label=f'Mean: {np.mean(envelope_peaks):.2f}')
    ax.set_xlabel('Envelope Peak', fontsize=9)
    ax.set_ylabel('Frequency', fontsize=9)
    ax.set_title(f'Distribution of Envelope Peaks\nAcross {num_events} Events', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 5. 包络方差分布
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(envelope_vars, bins=30, edgecolor='black', alpha=0.7, color='orange')
    ax.axvline(np.mean(envelope_vars), color='red', linestyle='--', linewidth=2, 
              label=f'Mean: {np.mean(envelope_vars):.2e}')
    ax.set_xlabel('Envelope Variance', fontsize=9)
    ax.set_ylabel('Frequency', fontsize=9)
    ax.set_title(f'Distribution of Envelope Variances\nAcross {num_events} Events', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 6. 相位锁定值（PLV）随时间变化
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(time_axis_us[:20000], plv_time[:20000], 'b-', linewidth=1.5)
    ax.axhline(mean_plv, color='red', linestyle='--', linewidth=2, 
              label=f'Mean PLV: {mean_plv:.4f}')
    ax.set_xlabel('Time (μs)', fontsize=9)
    ax.set_ylabel('Phase Locking Value (PLV)', fontsize=9)
    ax.set_title(f'Phase Locking Value Over Time\n(1.0=Full Sync, 0.0=Random)', fontsize=10)
    ax.set_ylim([0, 1.1])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 7. 相位标准差随时间变化
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(time_axis_us[:20000], phase_std_over_time[:20000], 'g-', linewidth=1.5)
    ax.axhline(mean_phase_std, color='red', linestyle='--', linewidth=2, 
              label=f'Mean: {mean_phase_std:.4f} rad')
    ax.set_xlabel('Time (μs)', fontsize=9)
    ax.set_ylabel('Phase Std (rad)', fontsize=9)
    ax.set_title(f'Phase Standard Deviation Over Time\n(Lower = More Consistent)', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 8. 瞬时频率标准差随时间变化
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(time_axis_us[:20000], inst_freq_std_over_time[:20000] / 1e6, 'm-', linewidth=1.5)
    ax.axhline(mean_inst_freq_std / 1e6, color='red', linestyle='--', linewidth=2, 
              label=f'Mean: {mean_inst_freq_std/1e6:.3f} MHz')
    ax.set_xlabel('Time (μs)', fontsize=9)
    ax.set_ylabel('Instantaneous Freq Std (MHz)', fontsize=9)
    ax.set_title(f'Instantaneous Frequency Std Over Time', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 9. 几个代表性event的相位
    ax = fig.add_subplot(gs[2, 2])
    for i in range(num_show):
        phase_normalized = all_phases[i, :10000] % (2 * np.pi)  # 归一化到0-2π
        ax.plot(time_axis_us[:10000], phase_normalized, 
               alpha=0.6, linewidth=0.8, label=f'Event {selected_indices[i]}' if i < 3 else '')
    ax.set_xlabel('Time (μs)', fontsize=9)
    ax.set_ylabel('Phase (rad, 0-2π)', fontsize=9)
    ax.set_title(f'Sample Phase Trajectories\n(First 10000 samples)', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 10. 相位一致性统计（PLV分布）
    ax = fig.add_subplot(gs[3, 0])
    ax.hist(plv_time, bins=50, edgecolor='black', alpha=0.7, color='purple')
    ax.axvline(mean_plv, color='red', linestyle='--', linewidth=2, 
              label=f'Mean: {mean_plv:.4f}')
    ax.set_xlabel('PLV', fontsize=9)
    ax.set_ylabel('Frequency', fontsize=9)
    ax.set_title(f'Distribution of PLV Values Over Time', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 11. 包络统计量之间的关系
    ax = fig.add_subplot(gs[3, 1])
    scatter = ax.scatter(envelope_means, envelope_peaks, c=envelope_vars, 
                        s=50, alpha=0.6, cmap='viridis')
    ax.set_xlabel('Envelope Mean', fontsize=9)
    ax.set_ylabel('Envelope Peak', fontsize=9)
    ax.set_title(f'Envelope Mean vs Peak\n(Color = Variance)', fontsize=10)
    plt.colorbar(scatter, ax=ax, label='Variance')
    ax.grid(True, alpha=0.3)
    
    # 12. 包络内的相位一致性（按包络幅度bin）
    ax = fig.add_subplot(gs[3, 0])
    valid_bins_mask = envelope_bin_counts > 0
    if np.any(valid_bins_mask):
        ax.plot(envelope_bin_centers[valid_bins_mask], envelope_bin_plv[valid_bins_mask], 
                'o-', linewidth=2, markersize=8, color='purple', label='PLV')
        ax2 = ax.twinx()
        ax2.bar(envelope_bin_centers[valid_bins_mask], envelope_bin_counts[valid_bins_mask],
                alpha=0.3, color='gray', width=(np.max(envelope_bin_centers) - np.min(envelope_bin_centers)) / num_envelope_bins * 0.8,
                label='Sample Count')
        ax.set_xlabel('Envelope Amplitude (Bin Center)', fontsize=9)
        ax.set_ylabel('Phase Locking Value (PLV)', fontsize=9, color='purple')
        ax2.set_ylabel('Sample Count', fontsize=9, color='gray')
        ax.set_title(f'Phase Consistency vs Envelope Amplitude\n(PLV in Different Envelope Bins)', fontsize=10)
        ax.tick_params(axis='y', labelcolor='purple')
        ax2.tick_params(axis='y', labelcolor='gray')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)
        ax2.legend(loc='upper right', fontsize=8)
    
    # 13. 包络bin内的相位标准差
    ax = fig.add_subplot(gs[3, 1])
    if np.any(valid_bins_mask):
        ax.plot(envelope_bin_centers[valid_bins_mask], envelope_bin_phase_std[valid_bins_mask], 
                's-', linewidth=2, markersize=8, color='orange', label='Phase Std')
        ax.set_xlabel('Envelope Amplitude (Bin Center)', fontsize=9)
        ax.set_ylabel('Phase Standard Deviation (rad)', fontsize=9)
        ax.set_title(f'Phase Std vs Envelope Amplitude\n(Lower = More Consistent)', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 14. 上升沿vs下降沿的相位分布
    ax = fig.add_subplot(gs[3, 2])
    if len(rising_edge_phases) > 0:
        # 将相位归一化到0-2π范围
        rising_phases_norm = all_rising_phases % (2 * np.pi)
        falling_phases_norm = all_falling_phases % (2 * np.pi)
        ax.hist(rising_phases_norm, bins=50, alpha=0.6, label=f'Rising Edge (PLV={rising_plv:.3f})', 
                color='green', edgecolor='black')
        ax.hist(falling_phases_norm, bins=50, alpha=0.6, label=f'Falling Edge (PLV={falling_plv:.3f})', 
                color='red', edgecolor='black')
        ax.set_xlabel('Phase (rad, 0-2π)', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.set_title(f'Phase Distribution: Rising vs Falling Edge', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 15. 包络与相位的散点图（一个代表性event）
    ax = fig.add_subplot(gs[4, 0])
    if num_events > 0:
        sample_event_idx = 0
        sample_envelope = all_envelopes[sample_event_idx, :10000]
        sample_phase_norm = (all_phases[sample_event_idx, :10000] % (2 * np.pi))
        scatter = ax.scatter(sample_phase_norm, sample_envelope, c=time_axis_us[:10000],
                           s=10, alpha=0.6, cmap='viridis')
        ax.set_xlabel('Phase (rad, 0-2π)', fontsize=9)
        ax.set_ylabel('Envelope Amplitude', fontsize=9)
        ax.set_title(f'Sample Event: Envelope vs Phase\n(Color = Time)', fontsize=10)
        plt.colorbar(scatter, ax=ax, label='Time (μs)')
        ax.grid(True, alpha=0.3)
    
    # 16. 包络峰值附近的相位分布
    ax = fig.add_subplot(gs[4, 1])
    if len(peak_nearby_phases) > 0:
        peak_phases_norm = all_peak_nearby_phases % (2 * np.pi)
        ax.hist(peak_phases_norm, bins=50, edgecolor='black', alpha=0.7, color='magenta')
        ax.axvline(np.mean(peak_phases_norm), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {np.mean(peak_phases_norm):.3f} rad')
        ax.set_xlabel('Phase (rad, 0-2π)', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.set_title(f'Phase Distribution Near Envelope Peak\n(PLV={peak_plv:.4f}, Std={peak_phase_std:.4f} rad)', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 17. 总结统计信息
    ax = fig.add_subplot(gs[4, 2])
    ax.axis('off')
    
    summary_text = f'Summary Statistics:\n\n'
    summary_text += f'Envelope Statistics:\n'
    summary_text += f'  Overall Mean: {overall_envelope_mean:.2f}\n'
    summary_text += f'  Overall Peak: {overall_envelope_peak:.2f}\n'
    summary_text += f'  Overall Variance: {overall_envelope_var:.2e}\n'
    summary_text += f'  Overall Std: {overall_envelope_std:.2f}\n\n'
    summary_text += f'Across Events:\n'
    summary_text += f'  Mean of Means: {np.mean(envelope_means):.2f} ± {np.std(envelope_means):.2f}\n'
    summary_text += f'  Mean of Peaks: {np.mean(envelope_peaks):.2f} ± {np.std(envelope_peaks):.2f}\n\n'
    summary_text += f'Phase Consistency:\n'
    summary_text += f'  Mean PLV: {mean_plv:.4f}\n'
    summary_text += f'  PLV Range: [{np.min(plv_time):.4f}, {np.max(plv_time):.4f}]\n'
    summary_text += f'  Mean Phase Std: {mean_phase_std:.4f} rad\n\n'
    summary_text += f'Envelope-Based Phase:\n'
    summary_text += f'  Peak Nearby PLV: {peak_plv:.4f}\n'
    summary_text += f'  Rising Edge PLV: {rising_plv:.4f}\n'
    summary_text += f'  Falling Edge PLV: {falling_plv:.4f}\n'
    
    ax.text(0.05, 0.95, summary_text, fontsize=9, family='monospace',
           verticalalignment='top', bbox=dict(boxstyle='round', 
           facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Hilbert Transform Analysis: RT Only (Non-Inhibit) Events\n'
                f'Total Events: {num_events}', fontsize=12, y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'\n图片已保存至: {save_path}')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    # 返回结果
    results = {
        'selected_event_indices': selected_indices,
        'num_events': num_events,
        'all_envelopes': all_envelopes,
        'all_phases': all_phases,
        'all_instantaneous_freqs': all_instantaneous_freqs,
        'envelope_means': envelope_means,
        'envelope_peaks': envelope_peaks,
        'envelope_vars': envelope_vars,
        'overall_envelope_mean': overall_envelope_mean,
        'overall_envelope_peak': overall_envelope_peak,
        'overall_envelope_var': overall_envelope_var,
        'plv_time': plv_time,
        'mean_plv': mean_plv,
        'phase_std_over_time': phase_std_over_time,
        'mean_phase_std': mean_phase_std,
        'envelope_bin_plv': envelope_bin_plv,
        'envelope_bin_phase_std': envelope_bin_phase_std,
        'envelope_bin_centers': envelope_bin_centers,
        'envelope_bin_counts': envelope_bin_counts,
        'peak_plv': peak_plv,
        'peak_phase_std': peak_phase_std,
        'rising_plv': rising_plv,
        'falling_plv': falling_plv,
        'rising_phase_std': rising_phase_std,
        'falling_phase_std': falling_phase_std,
        'sampling_interval_ns': sampling_interval_ns,
        'time_axis_us': time_axis_us
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
            print('Hilbert变换分析：RT且非Inhibit的events的CH0信号')
            print('=' * 70)
            
            try:
                results = analyze_rt_only_events_hilbert(
                    ch0_3_file, ch5_file,
                    rt_cut=6000.0,      # RT信号截断阈值
                    max_events=None,    # 分析所有符合条件的events，或设置一个数字限制
                    ch0_idx=0,          # CH0通道索引
                    ch5_idx=0,          # CH5通道索引
                    detrend=True,       # 去趋势
                    show_plot=True
                )
                
                print(f'\n分析完成！')
                print(f'  分析的event数: {results["num_events"]}')
                print(f'  包络均值: {results["overall_envelope_mean"]:.2f}')
                print(f'  包络峰值: {results["overall_envelope_peak"]:.2f}')
                print(f'  包络方差: {results["overall_envelope_var"]:.2e}')
                print(f'  平均相位锁定值 (PLV): {results["mean_plv"]:.4f}')
                
            except Exception as e:
                print(f'分析失败: {e}')
                import traceback
                traceback.print_exc()
        else:
            print('未找到匹配的文件对')
    else:
        print('未找到CH0-3或CH5文件')

