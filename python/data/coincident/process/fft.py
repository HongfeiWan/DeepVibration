#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对筛选出的RT且非Inhibit的events的CH0信号进行FFT分析
分析这些信号的频率成分
"""
import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from typing import Optional, Tuple, Dict, List

# 添加路径以便导入 utils 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
python_dir = os.path.dirname(os.path.dirname(current_dir))
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from utils.visualize import get_h5_files

# 导入coincident模块的筛选函数
# 由于文件名包含&字符，使用importlib直接导入
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

def analyze_rt_only_events_fft(ch0_3_file: str,
                                ch5_file: str,
                                rt_cut: float = 6000.0,
                                ch0_idx: int = 0,
                                ch5_idx: int = 0,
                                max_events: Optional[int] = None,
                                detrend: bool = True,
                                window: Optional[str] = 'hann',
                                save_path: Optional[str] = None,
                                show_plot: bool = True,
                                figsize: Tuple[int, int] = (16, 12)) -> Dict:
    """
    对筛选出的RT且非Inhibit的events的CH0信号进行FFT分析
    
    参数:
        ch0_3_file: CH0-3文件路径
        ch5_file: CH5文件路径
        rt_cut: RT信号的截断阈值
        ch0_idx: CH0通道索引
        ch5_idx: CH5通道索引
        max_events: 最多分析的event数量，如果为None则分析所有符合条件的events
        detrend: 是否去趋势
        window: 窗函数类型（'hann', 'hamming', None）
        save_path: 保存图片路径
        show_plot: 是否显示图片
        figsize: 图片大小
    
    返回:
        包含分析结果的字典
    """
    print('=' * 70)
    print(f'FFT分析：RT且非Inhibit的events的CH0信号')
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
        with h5py.File(ch0_3_file, 'r') as f_ch0:
            ch0_channel_data = f_ch0['channel_data']
            ch0_time_samples, ch0_num_channels, ch0_num_events = ch0_channel_data.shape
            ch0_min_values = np.zeros(ch0_num_events, dtype=np.float64)
            batch_size = 1000
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
    sampling_rate = 1.0 / sampling_interval_s  # 采样率：250 MSPS
    time_samples = 30000  # 每个event的采样点数
    
    print(f'\n信号参数:')
    print(f'  采样点数: {time_samples}')
    print(f'  采样间隔: {sampling_interval_ns} ns = {sampling_interval_s} s')
    print(f'  采样率: {sampling_rate/1e6:.2f} MSPS')
    print(f'  Event时长: {time_samples * sampling_interval_s * 1e6:.2f} μs')
    print(f'  Nyquist频率: {sampling_rate/2/1e6:.2f} MHz')
    
    # 4. 读取文件并进行FFT分析
    print(f'\n正在进行FFT分析...')
    
    all_fft_results = []  # 存储所有event的FFT结果
    all_frequencies = None
    
    with h5py.File(ch0_3_file, 'r') as f_ch0:
        ch0_channel_data = f_ch0['channel_data']
        ch0_time_samples_actual, ch0_num_channels, ch0_num_events = ch0_channel_data.shape
        
        if ch0_time_samples_actual != time_samples:
            print(f'警告: 实际采样点数 ({ch0_time_samples_actual}) 与预期 ({time_samples}) 不一致')
            time_samples = ch0_time_samples_actual
        
        # 计算频率数组（所有event共享）
        freqs = np.fft.fftfreq(time_samples, sampling_interval_s)
        # 只取正频率部分
        positive_freq_idx = freqs > 0
        freqs_positive = freqs[positive_freq_idx]
        all_frequencies = freqs_positive
        
        # 对每个event进行FFT
        for i, event_idx in enumerate(selected_indices):
            # 获取CH0的波形数据
            ch0_waveform = ch0_channel_data[:, ch0_idx, event_idx].astype(np.float64)
            
            # 去趋势
            if detrend:
                ch0_waveform = signal.detrend(ch0_waveform)
            
            # 应用窗函数
            if window == 'hann':
                window_func = signal.windows.hann(time_samples)
            elif window == 'hamming':
                window_func = signal.windows.hamming(time_samples)
            else:
                window_func = np.ones(time_samples)
            
            ch0_waveform_windowed = ch0_waveform * window_func
            
            # 执行FFT
            fft_result = np.fft.fft(ch0_waveform_windowed)
            
            # 计算功率谱（只取正频率部分）
            power_spectrum = np.abs(fft_result[positive_freq_idx]) ** 2
            
            all_fft_results.append(power_spectrum)
            
            if (i + 1) % 10 == 0 or i == num_events - 1:
                print(f'  已处理 {i+1}/{num_events} 个events')
    
    # 转换为numpy数组
    all_fft_results = np.array(all_fft_results)  # 形状: (num_events, freq_points)
    
    print(f'\nFFT分析完成')
    print(f'  分析的events数: {num_events}')
    print(f'  频率点数: {len(all_frequencies)}')
    print(f'  频率范围: {all_frequencies[0]:.3e} - {all_frequencies[-1]/1e6:.3f} MHz')
    
    # 5. 统计分析
    mean_power = np.mean(all_fft_results, axis=0)
    median_power = np.median(all_fft_results, axis=0)
    std_power = np.std(all_fft_results, axis=0)
    max_power = np.max(all_fft_results, axis=0)
    min_power = np.min(all_fft_results, axis=0)
    
    # 找到主要频率成分（功率谱的峰值）
    # 使用平均功率谱找峰值
    peaks, properties = signal.find_peaks(mean_power, height=np.max(mean_power) * 0.1, distance=10)
    peak_frequencies = all_frequencies[peaks]
    peak_powers = mean_power[peaks]
    
    # 按功率排序，取前10个主要频率
    top_peaks_idx = np.argsort(peak_powers)[-10:][::-1]
    top_frequencies = peak_frequencies[top_peaks_idx]
    top_powers = peak_powers[top_peaks_idx]
    
    print(f'\n主要频率成分（前10个峰值）:')
    for i, (freq, power) in enumerate(zip(top_frequencies, top_powers)):
        if freq < 1e6:
            print(f'  {i+1}. {freq:.3f} Hz, Power: {power:.2e}')
        elif freq < 1e9:
            print(f'  {i+1}. {freq/1e6:.3f} MHz, Power: {power:.2e}')
        else:
            print(f'  {i+1}. {freq/1e9:.3f} GHz, Power: {power:.2e}')
    
    # 6. 绘制结果
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    
    # 1. 平均功率谱（线性尺度，低频部分）
    ax = axes[0, 0]
    max_display_freq_idx = np.searchsorted(all_frequencies, 10e6)  # 显示到10MHz
    max_display_freq_idx = min(max_display_freq_idx, len(all_frequencies))
    ax.plot(all_frequencies[:max_display_freq_idx] / 1e6, mean_power[:max_display_freq_idx], 
            'b-', linewidth=1.5, label='Mean')
    ax.fill_between(all_frequencies[:max_display_freq_idx] / 1e6,
                    (mean_power - std_power)[:max_display_freq_idx],
                    (mean_power + std_power)[:max_display_freq_idx],
                    alpha=0.3, label='±1 Std')
    ax.set_xlabel('Frequency (MHz)', fontsize=10)
    ax.set_ylabel('Power Spectrum', fontsize=10)
    ax.set_title(f'Average Power Spectrum (Linear, 0-10 MHz)\n{num_events} events', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 2. 平均功率谱（对数尺度，低频部分）
    ax = axes[0, 1]
    ax.semilogy(all_frequencies[:max_display_freq_idx] / 1e6, mean_power[:max_display_freq_idx], 
                'b-', linewidth=1.5, label='Mean')
    ax.semilogy(all_frequencies[:max_display_freq_idx] / 1e6, median_power[:max_display_freq_idx], 
                'r--', linewidth=1, label='Median')
    ax.set_xlabel('Frequency (MHz)', fontsize=10)
    ax.set_ylabel('Power Spectrum (log scale)', fontsize=10)
    ax.set_title(f'Average Power Spectrum (Log Scale, 0-10 MHz)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 3. 平均功率谱（对数-对数尺度，全频段）
    ax = axes[1, 0]
    ax.loglog(all_frequencies / 1e6, mean_power, 'b-', linewidth=1.5, label='Mean')
    ax.loglog(all_frequencies / 1e6, median_power, 'r--', linewidth=1, label='Median')
    ax.set_xlabel('Frequency (MHz)', fontsize=10)
    ax.set_ylabel('Power Spectrum (log-log scale)', fontsize=10)
    ax.set_title(f'Average Power Spectrum (Log-Log, Full Range)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 4. 功率谱的统计分布（箱线图风格，选择几个频率点）
    ax = axes[1, 1]
    # 选择几个代表性频率点
    freq_indices = np.linspace(0, len(all_frequencies) - 1, 20, dtype=int)
    selected_freqs = all_frequencies[freq_indices] / 1e6  # 转换为MHz
    
    # 绘制每个频率点的功率分布（使用箱线图）
    power_data = [all_fft_results[:, idx] for idx in freq_indices]
    bp = ax.boxplot(power_data, positions=selected_freqs, widths=selected_freqs[1]-selected_freqs[0] if len(selected_freqs) > 1 else 0.1,
                   patch_artist=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Frequency (MHz)', fontsize=10)
    ax.set_ylabel('Power Spectrum Distribution', fontsize=10)
    ax.set_title(f'Power Spectrum Distribution Across Events\n(Boxplot at selected frequencies)', fontsize=11)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 5. 主要频率成分（峰值频率）
    ax = axes[2, 0]
    ax.barh(range(len(top_frequencies)), top_powers, color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(top_frequencies)))
    ax.set_yticklabels([f'{f/1e6:.3f} MHz' if f < 1e9 else f'{f/1e9:.3f} GHz' 
                        for f in top_frequencies], fontsize=8)
    ax.set_xlabel('Average Power', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title(f'Top 10 Dominant Frequencies', fontsize=11)
    ax.grid(True, alpha=0.3, axis='x')
    
    # 6. 功率谱的变异性（标准差/均值，显示频率稳定性）
    ax = axes[2, 1]
    cv = std_power / (mean_power + 1e-10)  # 变异系数（coefficient of variation）
    ax.plot(all_frequencies[:max_display_freq_idx] / 1e6, cv[:max_display_freq_idx], 
            'g-', linewidth=1.5)
    ax.set_xlabel('Frequency (MHz)', fontsize=10)
    ax.set_ylabel('Coefficient of Variation (Std/Mean)', fontsize=10)
    ax.set_title(f'Frequency Stability Across Events\n(0-10 MHz, Lower = More Stable)', fontsize=11)
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
        'selected_event_indices': selected_indices,
        'num_events': num_events,
        'frequencies': all_frequencies,
        'all_fft_results': all_fft_results,
        'mean_power': mean_power,
        'median_power': median_power,
        'std_power': std_power,
        'top_frequencies': top_frequencies,
        'top_powers': top_powers,
        'sampling_rate': sampling_rate,
        'sampling_interval_ns': sampling_interval_ns
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
            print('FFT分析：RT且非Inhibit的events的CH0信号')
            print('=' * 70)
            
            try:
                results = analyze_rt_only_events_fft(
                    ch0_3_file, ch5_file,
                    rt_cut=6000.0,      # RT信号截断阈值
                    max_events=None,    # 分析所有符合条件的events，或设置一个数字限制
                    ch0_idx=0,          # CH0通道索引
                    ch5_idx=0,          # CH5通道索引
                    detrend=True,       # 去趋势
                    window='hann',      # 窗函数：'hann', 'hamming', None
                    show_plot=True
                )
                
                print(f'\n分析完成！')
                print(f'  分析的event数: {results["num_events"]}')
                print(f'  频率点数: {len(results["frequencies"])}')
                print(f'  主要频率成分数: {len(results["top_frequencies"])}')
                
            except Exception as e:
                print(f'分析失败: {e}')
                import traceback
                traceback.print_exc()
        else:
            print('未找到匹配的文件对')
    else:
        print('未找到CH0-3或CH5文件')

