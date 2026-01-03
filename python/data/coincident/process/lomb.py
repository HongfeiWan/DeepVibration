#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对筛选出的RT且非Inhibit的events的CH0信号进行Lomb-Scargle周期图分析
使用每个event的绝对时间戳进行低频率分析
"""
import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import lombscargle
from typing import Optional, Tuple, Dict, List

# 添加路径以便导入 utils 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
python_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from utils.visualize import get_h5_files

# 导入coincident模块的筛选函数
coincident_module_path = os.path.dirname(os.path.dirname(current_dir))
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


def analyze_rt_only_events_lomb_scargle(ch0_3_file: str,
                                        ch5_file: str,
                                        rt_cut: float = 6000.0,
                                        ch0_idx: int = 0,
                                        ch5_idx: int = 0,
                                        max_events: Optional[int] = None,
                                        min_freq: float = 0.1,
                                        max_freq: float = 200.0,
                                        detrend: bool = True,
                                        oversampling: float = 2.0,
                                        concatenate_all: bool = True,
                                        save_path: Optional[str] = None,
                                        show_plot: bool = True,
                                        figsize: Tuple[int, int] = (16, 12)) -> Dict:
    """
    对筛选出的RT且非Inhibit的events的CH0信号进行Lomb-Scargle周期图分析
    
    参数:
        ch0_3_file: CH0-3文件路径
        ch5_file: CH5文件路径
        rt_cut: RT信号的截断阈值
        ch0_idx: CH0通道索引
        ch5_idx: CH5通道索引
        max_events: 最多分析的event数量，如果为None则分析所有符合条件的events
        min_freq: 最小频率（Hz）
        max_freq: 最大频率（Hz）
        detrend: 是否去趋势
        oversampling: 过采样因子（相对于Nyquist频率）
        concatenate_all: 是否将所有events拼接成一个时间序列（True）或分别分析（False）
        save_path: 保存图片路径
        show_plot: 是否显示图片
        figsize: 图片大小
    
    返回:
        包含分析结果的字典
    """
    print('=' * 70)
    print(f'Lomb-Scargle分析：RT且非Inhibit的events的CH0信号')
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
    
    print(f'\n信号参数:')
    print(f'  采样点数: {time_samples}')
    print(f'  采样间隔: {sampling_interval_ns} ns = {sampling_interval_s} s')
    print(f'  采样率: {1.0/sampling_interval_s/1e6:.2f} MSPS')
    print(f'  Event时长: {time_samples * sampling_interval_s * 1e6:.2f} μs')
    
    # 4. 读取文件并构建时间序列
    print(f'\n正在构建时间序列...')
    
    with h5py.File(ch0_3_file, 'r') as f_ch0:
        ch0_channel_data = f_ch0['channel_data']
        ch0_time_data = f_ch0['time_data']  # 每个event的触发时间
        ch0_time_samples_actual, ch0_num_channels, ch0_num_events = ch0_channel_data.shape
        
        if ch0_time_samples_actual != time_samples:
            print(f'警告: 实际采样点数 ({ch0_time_samples_actual}) 与预期 ({time_samples}) 不一致')
            time_samples = ch0_time_samples_actual
        
        # 收集所有信号点和对应的时间戳
        all_signal_values = []
        all_time_stamps = []
        
        for i, event_idx in enumerate(selected_indices):
            # 获取event的触发时间（秒）
            event_trigger_time = ch0_time_data[event_idx]
            
            # 获取CH0的波形数据
            ch0_waveform = ch0_channel_data[:, ch0_idx, event_idx].astype(np.float64)
            
            # 去趋势
            if detrend:
                ch0_waveform = signal.detrend(ch0_waveform)
            
            # 计算每个采样点的绝对时间戳
            # 绝对时间 = 触发时间 + 采样点索引 * 采样间隔
            event_times = event_trigger_time + np.arange(time_samples) * sampling_interval_s
            
            # 添加到总数组
            all_signal_values.extend(ch0_waveform)
            all_time_stamps.extend(event_times)
            
            if (i + 1) % 10 == 0 or i == num_events - 1:
                print(f'  已处理 {i+1}/{num_events} 个events')
        
        # 转换为numpy数组并排序（按时间）
        all_signal_values = np.array(all_signal_values)
        all_time_stamps = np.array(all_time_stamps)
        
        # 按时间排序（确保时间顺序正确）
        sort_indices = np.argsort(all_time_stamps)
        all_signal_values = all_signal_values[sort_indices]
        all_time_stamps = all_time_stamps[sort_indices]
        
        total_duration = all_time_stamps[-1] - all_time_stamps[0]
        
        print(f'\n时间序列构建完成:')
        print(f'  总数据点数: {len(all_signal_values)}')
        print(f'  时间跨度: {total_duration:.6f} s ({total_duration/60:.2f} 分钟)')
        print(f'  起始时间: {all_time_stamps[0]:.6f} s')
        print(f'  结束时间: {all_time_stamps[-1]:.6f} s')
        print(f'  平均采样率: {len(all_signal_values)/total_duration:.2f} Hz')
        
        # 检查时间间隔
        time_diffs = np.diff(all_time_stamps)
        mean_interval = np.mean(time_diffs)
        std_interval = np.std(time_diffs)
        print(f'  平均时间间隔: {mean_interval*1e9:.2f} ns (std: {std_interval*1e9:.2f} ns)')
        print(f'  最小时间间隔: {np.min(time_diffs)*1e9:.2f} ns')
        print(f'  最大时间间隔: {np.max(time_diffs)*1e9:.2f} ns')
        
        # 5. 准备频率网格
        print(f'\n准备频率网格...')
        print(f'  目标频率范围: {min_freq:.3f} - {max_freq:.3f} Hz')
        
        # 频率分辨率（基于总时长）
        freq_resolution = 1.0 / total_duration
        print(f'  频率分辨率: {freq_resolution:.6f} Hz (1/总时长)')
        
        # 计算合理的频率点数
        # 基于频率范围和分辨率计算
        nfreq_range = int((max_freq - min_freq) / freq_resolution)
        # 应用过采样
        nfreq = int(nfreq_range * oversampling)
        
        # 限制频率点数以避免内存问题
        # Lomb-Scargle 内存需求约为: 数据点数 × 频率点数 × 8 bytes (float64)
        # 假设最大可用内存为 8GB，则最多允许 1e9 bytes = 1GB 用于计算
        max_memory_bytes = 1e9  # 1GB
        max_freqs_by_memory = int(max_memory_bytes / (len(all_signal_values) * 8))
        nfreq = min(nfreq, max_freqs_by_memory)
        nfreq = min(nfreq, 50000)  # 硬性限制最大 50000 个频率点
        
        freqs_hz = np.linspace(min_freq, max_freq, nfreq)
        freqs = 2.0 * np.pi * freqs_hz  # 转换为角频率
        
        # 计算Nyquist频率（基于平均采样间隔）
        nyquist_freq = 1.0 / (2.0 * mean_interval)
        
        print(f'  频率点数: {nfreq}')
        print(f'  基于频率分辨率的需求: {nfreq_range}')
        print(f'  内存限制允许的最大频率点数: {max_freqs_by_memory}')
        print(f'  Nyquist频率（基于平均间隔）: {nyquist_freq/1e6:.3f} MHz')
        print(f'  过采样因子: {oversampling}')
        
        # 检查内存需求
        estimated_memory_gb = (len(all_signal_values) * nfreq * 8) / 1e9
        print(f'  估计内存需求: {estimated_memory_gb:.2f} GB')
        
        if estimated_memory_gb > 8.0:
            print(f'  警告: 内存需求可能过大 ({estimated_memory_gb:.2f} GB)')
            print(f'  将使用分批处理来避免内存问题')
        
        # 6. 执行Lomb-Scargle分析（分批处理以避免内存问题）
        print(f'\n正在进行Lomb-Scargle分析...')
        print(f'  数据点数: {len(all_signal_values)}')
        print(f'  频率点数: {len(freqs)}')
        
        # 归一化时间（减去第一个时间戳，避免数值问题）
        times_normalized = all_time_stamps - all_time_stamps[0]
        
        # 如果数据点太多，使用分批处理
        max_data_points_per_batch = 3000000  # 每批最多50万个数据点
        
        if len(all_signal_values) > max_data_points_per_batch:
            print(f'  数据点过多，使用分批处理...')
            num_batches = int(np.ceil(len(all_signal_values) / max_data_points_per_batch))
            print(f'  将分成 {num_batches} 个批次进行分析')
            
            all_power_spectra = []
            
            for batch_idx in range(num_batches):
                batch_start = batch_idx * max_data_points_per_batch
                batch_end = min((batch_idx + 1) * max_data_points_per_batch, len(all_signal_values))
                
                batch_times = times_normalized[batch_start:batch_end]
                batch_signal = all_signal_values[batch_start:batch_end]
                
                print(f'    处理批次 {batch_idx + 1}/{num_batches} (数据点 {batch_start} - {batch_end})')
                
                # 对当前批次执行Lomb-Scargle
                batch_power = lombscargle(batch_times, batch_signal, freqs, normalize=True)
                all_power_spectra.append(batch_power)
                
                if (batch_idx + 1) % 5 == 0 or batch_idx == num_batches - 1:
                    print(f'      已完成 {batch_idx + 1}/{num_batches} 个批次')
            
            # 合并各批次的结果（平均）
            print(f'  合并 {num_batches} 个批次的结果（平均）...')
            all_power_spectra = np.array(all_power_spectra)
            power = np.mean(all_power_spectra, axis=0)
            print(f'  合并完成')
        else:
            # 数据点不多，直接分析
            print(f'  数据点数量适中，直接进行分析...')
            power = lombscargle(times_normalized, all_signal_values, freqs, normalize=True)
        
        print(f'\nLomb-Scargle分析完成')
        
        # 7. 统计分析
        # 找到峰值频率
        peak_indices, properties = signal.find_peaks(power, height=np.max(power) * 0.1, distance=10)
        peak_frequencies = freqs_hz[peak_indices]
        peak_powers = power[peak_indices]
        
        # 按功率排序，取前10个主要频率
        top_peaks_idx = np.argsort(peak_powers)[-10:][::-1]
        top_frequencies = peak_frequencies[top_peaks_idx]
        top_powers = peak_powers[top_peaks_idx]
        
        print(f'\n主要频率成分（前10个峰值）:')
        for i, (freq, pwr) in enumerate(zip(top_frequencies, top_powers)):
            print(f'  {i+1}. {freq:.4f} Hz, Power: {pwr:.2e}')
        
        # 8. 绘制结果
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        
        # 1. 功率谱（线性尺度）
        ax = axes[0, 0]
        ax.plot(freqs_hz, power, 'b-', linewidth=1.5)
        ax.set_xlabel('Frequency (Hz)', fontsize=10)
        ax.set_ylabel('Power', fontsize=10)
        ax.set_title(f'Lomb-Scargle Periodogram (Linear Scale)\n'
                    f'{num_events} events, {len(all_signal_values)} points, '
                    f'Duration: {total_duration:.2f} s', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 标注主要峰值
        for freq, pwr in zip(top_frequencies[:5], top_powers[:5]):
            ax.plot(freq, pwr, 'ro', markersize=8)
            ax.text(freq, pwr * 1.1, f'{freq:.2f} Hz', 
                   ha='center', va='bottom', fontsize=8)
        
        # 2. 功率谱（对数尺度）
        ax = axes[0, 1]
        ax.semilogy(freqs_hz, power, 'b-', linewidth=1.5)
        ax.set_xlabel('Frequency (Hz)', fontsize=10)
        ax.set_ylabel('Power (log scale)', fontsize=10)
        ax.set_title(f'Lomb-Scargle Periodogram (Log Scale)', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 标注主要峰值
        for freq, pwr in zip(top_frequencies[:5], top_powers[:5]):
            ax.plot(freq, pwr, 'ro', markersize=8)
            ax.text(freq, pwr * 1.3, f'{freq:.2f} Hz', 
                   ha='center', va='bottom', fontsize=8)
        
        # 3. 低频段详细视图（0-10 Hz）
        ax = axes[1, 0]
        low_freq_mask = freqs_hz <= 10.0
        if np.any(low_freq_mask):
            ax.plot(freqs_hz[low_freq_mask], power[low_freq_mask], 'g-', linewidth=2.0)
            ax.set_xlabel('Frequency (Hz)', fontsize=10)
            ax.set_ylabel('Power', fontsize=10)
            ax.set_title(f'Low Frequency Detail (0-10 Hz)', fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # 标注低频峰值
            low_freq_peaks = top_frequencies[top_frequencies <= 10.0]
            low_freq_powers = top_powers[top_frequencies <= 10.0]
            for freq, pwr in zip(low_freq_peaks[:5], low_freq_powers[:5]):
                ax.plot(freq, pwr, 'ro', markersize=10)
                ax.text(freq, pwr * 1.1, f'{freq:.3f} Hz', 
                       ha='center', va='bottom', fontsize=9)
        
        # 4. 时间序列示例（前几个event）
        ax = axes[1, 1]
        num_show_samples = min(50000, len(all_signal_values))
        time_hours = (all_time_stamps[:num_show_samples] - all_time_stamps[0]) / 3600.0  # 转换为小时
        ax.plot(time_hours, all_signal_values[:num_show_samples], 'b-', linewidth=0.5, alpha=0.7)
        ax.set_xlabel('Time (hours from start)', fontsize=10)
        ax.set_ylabel('Signal Amplitude', fontsize=10)
        ax.set_title(f'Time Series (First {num_show_samples} points)\n'
                    f'{num_events} events concatenated', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 5. 主要频率成分（条形图）
        ax = axes[2, 0]
        ax.barh(range(len(top_frequencies)), top_powers, color='steelblue', alpha=0.7)
        ax.set_yticks(range(len(top_frequencies)))
        ax.set_yticklabels([f'{f:.3f} Hz' for f in top_frequencies], fontsize=9)
        ax.set_xlabel('Power', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'Top 10 Dominant Frequencies', fontsize=11)
        ax.grid(True, alpha=0.3, axis='x')
        
        # 6. 统计信息
        ax = axes[2, 1]
        ax.axis('off')
        
        info_text = f'Analysis Summary:\n\n'
        info_text += f'Data Statistics:\n'
        info_text += f'  Number of Events: {num_events}\n'
        info_text += f'  Total Data Points: {len(all_signal_values)}\n'
        info_text += f'  Time Duration: {total_duration:.2f} s ({total_duration/60:.2f} min)\n'
        info_text += f'  Mean Sampling Rate: {len(all_signal_values)/total_duration:.2f} Hz\n'
        info_text += f'  Mean Interval: {mean_interval*1e9:.2f} ns\n\n'
        info_text += f'Frequency Analysis:\n'
        info_text += f'  Frequency Range: {min_freq:.3f} - {max_freq:.3f} Hz\n'
        info_text += f'  Frequency Resolution: {freq_resolution:.6f} Hz\n'
        info_text += f'  Number of Frequencies: {len(freqs_hz)}\n'
        info_text += f'  Nyquist Frequency: {nyquist_freq/1e6:.3f} MHz\n\n'
        info_text += f'Top 5 Frequencies:\n'
        for i, (freq, pwr) in enumerate(zip(top_frequencies[:5], top_powers[:5])):
            info_text += f'  {i+1}. {freq:.4f} Hz (Power: {pwr:.2e})\n'
        
        ax.text(0.05, 0.95, info_text, fontsize=10, family='monospace',
               verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'Lomb-Scargle Analysis: RT Only (Non-Inhibit) Events\n'
                    f'CH0 Signal, Low Frequency Analysis', fontsize=12, y=0.995)
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
            'all_signal_values': all_signal_values,
            'all_time_stamps': all_time_stamps,
            'total_duration': total_duration,
            'frequencies': freqs_hz,
            'power': power,
            'top_frequencies': top_frequencies,
            'top_powers': top_powers,
            'freq_resolution': freq_resolution,
            'mean_interval': mean_interval
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
            print('Lomb-Scargle分析：RT且非Inhibit的events的CH0信号')
            print('=' * 70)
            
            try:
                results = analyze_rt_only_events_lomb_scargle(
                    ch0_3_file, ch5_file,
                    rt_cut=6000.0,      # RT信号截断阈值
                    max_events=None,    # 分析所有符合条件的events，或设置一个数字限制
                    ch0_idx=0,          # CH0通道索引
                    ch5_idx=0,          # CH5通道索引
                    min_freq=0.1,       # 最小频率（Hz）
                    max_freq=200.0,     # 最大频率（Hz）
                    detrend=True,       # 去趋势
                    oversampling=2.0,   # 过采样因子
                    concatenate_all=True,  # 将所有events拼接
                    show_plot=True
                )
                
                print(f'\n分析完成！')
                print(f'  分析的event数: {results["num_events"]}')
                print(f'  总数据点数: {len(results["all_signal_values"])}')
                print(f'  时间跨度: {results["total_duration"]:.2f} s')
                print(f'  主要频率成分数: {len(results["top_frequencies"])}')
                print(f'\n前5个主要频率:')
                for i, (freq, pwr) in enumerate(zip(results["top_frequencies"][:5], results["top_powers"][:5])):
                    print(f'  {i+1}. {freq:.4f} Hz (Power: {pwr:.2e})')
                
            except Exception as e:
                print(f'分析失败: {e}')
                import traceback
                traceback.print_exc()
        else:
            print('未找到匹配的文件对')
    else:
        print('未找到CH0-3或CH5文件')

