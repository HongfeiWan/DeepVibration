#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
在Physical信号基础上，过滤掉过阈值的事例（max(CH0) > 16382）
筛选条件：既非RT也非Inhibit，且CH0最大值 <= 16382
"""
import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple
import importlib.util

# 添加路径以便导入 utils 模块和 select.py
current_dir = os.path.dirname(os.path.abspath(__file__))
# current_dir = python/data/physical/cut
parent_dir = os.path.dirname(current_dir)  # python/data/physical
grandparent_dir = os.path.dirname(parent_dir)  # python/data
python_dir = os.path.dirname(grandparent_dir)  # python
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from utils.visualize import get_h5_files

# 导入 select.py 模块
select_py_path = os.path.join(parent_dir, 'select.py')
spec = importlib.util.spec_from_file_location("physical_select", select_py_path)
physical_select = importlib.util.module_from_spec(spec)
spec.loader.exec_module(physical_select)
select_physical_events = physical_select.select_physical_events


def select_physical_events_no_overthreshold(ch0_3_file: str = None,
                                            ch5_file: str = None,
                                            rt_cut: float = 6000.0,
                                            ch0_threshold: float = 16382.0,
                                            ch0_idx: int = 0,
                                            ch5_idx: int = 0) -> Dict:
    """
    筛选既非RT也非Inhibit，且CH0最大值不超过阈值的Physical信号
    
    参数:
        ch0_3_file: CH0-3文件路径，如果为None则自动获取
        ch5_file: CH5文件路径，如果为None则自动获取
        rt_cut: RT信号截断阈值（CH5最大值 > rt_cut 为RT信号）
        ch0_threshold: CH0最大值阈值（max(CH0) > ch0_threshold 为过阈值信号）
        ch0_idx: CH0通道索引
        ch5_idx: CH5通道索引
    
    返回:
        包含筛选结果的字典
    """
    print('=' * 70)
    print('筛选Physical信号（既非RT也非Inhibit，且不过阈值）')
    print('=' * 70)
    
    # 1. 首先使用 select_physical_events 筛选出物理事例
    print('\n第一步：筛选既非RT也非Inhibit的Physical信号...')
    physical_result = select_physical_events(ch0_3_file, ch5_file, rt_cut, ch0_idx, ch5_idx)
    
    ch0_3_file = physical_result['ch0_3_file']
    ch5_file = physical_result['ch5_file']
    physical_indices = physical_result['selected_indices']
    physical_count_before = physical_result['physical_count']
    
    if physical_count_before == 0:
        print('未发现Physical信号，无法继续筛选')
        return {
            'ch0_3_file': ch0_3_file,
            'ch5_file': ch5_file,
            'selected_indices': np.array([], dtype=np.int64),
            'trigger_times': np.array([]),
            'rt_count': physical_result['rt_count'],
            'inhibit_count': physical_result['inhibit_count'],
            'physical_count': 0,
            'overthreshold_count': 0,
            'final_physical_count': 0,
            'total_events': physical_result['total_events']
        }
    
    # 2. 在物理事例基础上，计算CH0最大值并过滤过阈值的事例
    print(f'\n第二步：过滤掉CH0最大值 > {ch0_threshold} 的过阈值事例...')
    
    batch_size = 1000
    ch0_max_values = np.zeros(len(physical_indices), dtype=np.float64)
    
    with h5py.File(ch0_3_file, 'r') as f_ch0:
        ch0_channel_data = f_ch0['channel_data']
        
        # 批量计算CH0最大值
        for i in range(0, len(physical_indices), batch_size):
            end_idx = min(i + batch_size, len(physical_indices))
            batch_indices = physical_indices[i:end_idx]
            batch_data = ch0_channel_data[:, ch0_idx, batch_indices]
            ch0_max_values[i:end_idx] = np.max(batch_data, axis=0)
    
    # 判断是否过阈值
    overthreshold_mask = ch0_max_values > ch0_threshold
    valid_mask = ~overthreshold_mask
    final_indices = physical_indices[valid_mask]
    
    overthreshold_count = np.sum(overthreshold_mask)
    final_physical_count = len(final_indices)
    
    print(f'  筛选完成:')
    print(f'    初始Physical信号: {physical_count_before} 个')
    print(f'    过阈值信号 (max(CH0) > {ch0_threshold}): {overthreshold_count} 个')
    print(f'    最终Physical信号 (不过阈值): {final_physical_count} 个')
    
    # 读取触发时间
    with h5py.File(ch0_3_file, 'r') as f_ch0:
        ch0_time_data = f_ch0['time_data']
        trigger_times = ch0_time_data[final_indices]
    
    return {
        'ch0_3_file': ch0_3_file,
        'ch5_file': ch5_file,
        'selected_indices': final_indices,
        'trigger_times': trigger_times,
        'rt_count': physical_result['rt_count'],
        'inhibit_count': physical_result['inhibit_count'],
        'physical_count': physical_count_before,
        'overthreshold_count': overthreshold_count,
        'final_physical_count': final_physical_count,
        'total_events': physical_result['total_events']
    }

def plot_physical_waveforms_no_overthreshold(ch0_3_file: str = None,
                                            ch5_file: str = None,
                                            rt_cut: float = 6000.0,
                                            ch0_threshold: float = 16382.0,
                                            ch0_idx: int = 0,
                                            ch5_idx: int = 0,
                                            max_events_to_plot: Optional[int] = 10,
                                            save_path: Optional[str] = None,
                                            show_plot: bool = True,
                                            figsize: Tuple[int, int] = (16, 12)) -> None:
    """
    筛选并可视化Physical信号（既非RT也非Inhibit，且不过阈值）的四个通道原始波形
    
    参数:
        ch0_3_file: CH0-3文件路径，如果为None则自动获取
        ch5_file: CH5文件路径，如果为None则自动获取
        rt_cut: RT信号截断阈值（CH5最大值 > rt_cut 为RT信号）
        ch0_threshold: CH0最大值阈值（max(CH0) > ch0_threshold 为过阈值信号）
        ch0_idx: CH0通道索引
        ch5_idx: CH5通道索引
        max_events_to_plot: 最多绘制的Physical事件数量，None表示绘制所有
        save_path: 保存图片路径，如果为None则不保存
        show_plot: 是否显示图片
        figsize: 图片大小
    """
    # 1. 筛选Physical信号（不过阈值）
    selection_result = select_physical_events_no_overthreshold(
        ch0_3_file, ch5_file, rt_cut, ch0_threshold, ch0_idx, ch5_idx
    )
    
    ch0_3_file = selection_result['ch0_3_file']
    all_selected_indices = selection_result['selected_indices']  # 所有满足条件的信号
    final_physical_count = selection_result['final_physical_count']
    
    if final_physical_count == 0:
        print('未发现Physical信号（不过阈值），无法绘制')
        return
    
    # 确定要绘制的事件数量
    if max_events_to_plot is None:
        num_to_plot = final_physical_count
    else:
        num_to_plot = min(max_events_to_plot, final_physical_count)
    
    selected_indices = all_selected_indices[:num_to_plot]
    print(f'\n将绘制 {num_to_plot} 个Physical信号（不过阈值）的CH0-3和CH5通道波形')
    
    # 获取CH5文件路径
    ch5_file = selection_result['ch5_file']
    
    # 1.5. 计算所有满足条件的信号的基线水平（开头500点和末尾500点的平均值）
    print(f'\n计算所有 {final_physical_count} 个满足条件信号的基线水平...')
    baseline_points = 500  # 2μs = 500个点（采样间隔4ns，500*4ns = 2μs）
    baselines = []
    
    # 2. 读取文件获取波形数据并计算基线
    with h5py.File(ch0_3_file, 'r') as f_ch0:
        ch0_channel_data = f_ch0['channel_data']
        time_samples_ch0, num_channels_ch0, num_events_ch0 = ch0_channel_data.shape
        
        # 计算所有满足条件的信号的基线水平
        if time_samples_ch0 < baseline_points * 2:
            print(f'警告: 波形长度 {time_samples_ch0} 小于 {baseline_points * 2}，无法计算基线')
            baselines = np.array([])
        else:
            batch_size = 1000
            for i in range(0, len(all_selected_indices), batch_size):
                end_idx = min(i + batch_size, len(all_selected_indices))
                batch_indices = all_selected_indices[i:end_idx]
                batch_data = ch0_channel_data[:, ch0_idx, batch_indices].astype(np.float64)
                
                # 计算开头500点和末尾500点的平均值
                start_baseline = np.mean(batch_data[:baseline_points, :], axis=0)
                end_baseline = np.mean(batch_data[-baseline_points:, :], axis=0)
                
                # 基线 = (开头平均值 + 末尾平均值) / 2
                batch_baselines = (start_baseline + end_baseline) / 2.0
                baselines.extend(batch_baselines)
            
            baselines = np.array(baselines)
            
            if len(baselines) > 0:
                print(f'基线计算完成: {len(baselines)} 个信号')
                print(f'  基线范围: {np.min(baselines):.2f} - {np.max(baselines):.2f} ADC')
                print(f'  基线平均值: {np.mean(baselines):.2f} ADC')
                print(f'  基线标准差: {np.std(baselines):.2f} ADC')
        
        # 确保有4个通道（CH0-3）
        if num_channels_ch0 < 4:
            print(f'警告: CH0-3文件只有 {num_channels_ch0} 个通道，将只绘制前 {num_channels_ch0} 个通道')
            num_channels_to_plot = num_channels_ch0
        else:
            num_channels_to_plot = 4
        
        # 读取CH5数据
        with h5py.File(ch5_file, 'r') as f_ch5:
            ch5_channel_data = f_ch5['channel_data']
            time_samples_ch5, num_channels_ch5, num_events_ch5 = ch5_channel_data.shape
            
            # 参数设置
            sampling_interval_ns = 4.0  # 4ns per sample
            sampling_interval_s = sampling_interval_ns * 1e-9
            time_axis_us_ch0 = np.arange(time_samples_ch0) * sampling_interval_ns / 1000.0  # 转换为微秒
            time_axis_us_ch5 = np.arange(time_samples_ch5) * sampling_interval_ns / 1000.0  # 转换为微秒
            
            # 通道名称（CH0-3 + CH5）
            channel_names = ['CH0', 'CH1', 'CH2', 'CH3', 'CH5']
            channel_colors = ['b', 'g', 'r', 'm', 'orange']  # 蓝色、绿色、红色、品红色、橙色
            
            # 3. 创建图形
            # 每个事件一行，每行5个子图（CH0-3 + CH5）
            n_cols = 5  # 4个CH0-3通道 + 1个CH5通道
            n_rows = num_to_plot  # 每个事件一行
            
            # 调整图形大小以适应5列
            fig_width = figsize[0] * (5 / 4)  # 按比例增加宽度
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, figsize[1]))
            
            # 如果只有一个事件，确保axes是二维数组
            if num_to_plot == 1:
                axes = axes.reshape(1, -1)
            
            # 4. 绘制每个Physical信号的CH0-3和CH5通道波形
            for plot_idx, event_idx in enumerate(selected_indices):
                # 绘制CH0-3的4个通道
                for ch_idx in range(num_channels_to_plot):
                    ax = axes[plot_idx, ch_idx] if n_rows > 1 else axes[ch_idx]
                    
                    # 获取CH0-3波形数据
                    waveform = ch0_channel_data[:, ch_idx, event_idx].astype(np.float64)
                    min_val = np.min(waveform)
                    max_val = np.max(waveform)
                    mean_val = np.mean(waveform)
                    std_val = np.std(waveform)
                    
                    # 如果是CH0，先计算y轴范围（在绘制之前）
                    if ch_idx == 0:
                        # 只根据波形数据本身设置y轴范围，不强制包含阈值线
                        y_min = min_val
                        y_max = max_val
                        
                        # 计算数据范围并添加边距
                        data_range = y_max - y_min
                        if data_range > 0:
                            # 添加10%的边距
                            margin = data_range * 0.1
                            y_min = y_min - margin
                            y_max = y_max + margin
                        else:
                            # 如果数据范围很小或为0，使用基于值的边距
                            center = (y_min + y_max) / 2
                            margin = max(abs(center) * 0.1, 100)
                            y_min = center - margin
                            y_max = center + margin
                    
                    # 绘制波形
                    color = channel_colors[ch_idx] if ch_idx < len(channel_colors) else 'b'
                    ax.plot(time_axis_us_ch0, waveform, color=color, linewidth=0.8, alpha=0.8)
                    
                    # 如果是CH0，设置y轴范围
                    if ch_idx == 0:
                        ax.set_ylim(y_min, y_max)
                    
                    # 标注关键统计信息
                    ax.axhline(mean_val, color='gray', linestyle='--', linewidth=0.5, alpha=0.5, label=f'Mean: {mean_val:.1f}')
                    
                    # 如果是CH0，标注阈值线
                    if ch_idx == 0:
                        ax.axhline(ch0_threshold, color='red', linestyle=':', linewidth=1.0, alpha=0.7, label=f'Threshold: {ch0_threshold}')
                    
                    # 设置标题和标签
                    if plot_idx == 0:
                        # 第一行显示通道名称
                        ax.set_title(f'{channel_names[ch_idx]}', fontsize=11, fontweight='bold')
                    else:
                        ax.set_title('', fontsize=9)
                    
                    if ch_idx == 0:
                        # 第一列显示事件编号
                        ax.set_ylabel(f'Event #{event_idx}\nAmplitude (ADC)', fontsize=9)
                    else:
                        ax.set_ylabel('Amplitude (ADC)', fontsize=9)
                    
                    if plot_idx == n_rows - 1:
                        # 最后一行显示时间轴标签
                        ax.set_xlabel('Time (μs)', fontsize=9)
                    else:
                        ax.set_xlabel('', fontsize=9)
                    
                    # 在图上显示统计信息
                    info_text = f'Min: {min_val:.1f}\nMax: {max_val:.1f}\nMean: {mean_val:.1f}\nStd: {std_val:.1f}'
                    ax.text(0.98, 0.98, info_text, transform=ax.transAxes,
                           verticalalignment='top', horizontalalignment='right',
                           fontsize=7, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    
                    ax.grid(True, alpha=0.3)
                
                # 绘制CH5通道（第5列）
                ch5_col_idx = 4
                ax = axes[plot_idx, ch5_col_idx] if n_rows > 1 else axes[ch5_col_idx]
                
                # 获取CH5波形数据
                ch5_waveform = ch5_channel_data[:, ch5_idx, event_idx].astype(np.float64)
                ch5_min_val = np.min(ch5_waveform)
                ch5_max_val = np.max(ch5_waveform)
                ch5_mean_val = np.mean(ch5_waveform)
                ch5_std_val = np.std(ch5_waveform)
                
                # 绘制CH5波形
                ch5_color = channel_colors[ch5_col_idx] if ch5_col_idx < len(channel_colors) else 'orange'
                ax.plot(time_axis_us_ch5, ch5_waveform, color=ch5_color, linewidth=0.8, alpha=0.8)
                
                # 标注关键统计信息
                ax.axhline(ch5_mean_val, color='gray', linestyle='--', linewidth=0.5, alpha=0.5, label=f'Mean: {ch5_mean_val:.1f}')
                
                # 设置标题和标签
                if plot_idx == 0:
                    # 第一行显示通道名称
                    ax.set_title(f'{channel_names[ch5_col_idx]}', fontsize=11, fontweight='bold')
                else:
                    ax.set_title('', fontsize=9)
                
                if plot_idx == n_rows - 1:
                    # 最后一行显示时间轴标签
                    ax.set_xlabel('Time (μs)', fontsize=9)
                else:
                    ax.set_xlabel('', fontsize=9)
                
                ax.set_ylabel('Amplitude (ADC)', fontsize=9)
                
                # 在图上显示统计信息
                ch5_info_text = f'Min: {ch5_min_val:.1f}\nMax: {ch5_max_val:.1f}\nMean: {ch5_mean_val:.1f}\nStd: {ch5_std_val:.1f}'
                ax.text(0.98, 0.98, ch5_info_text, transform=ax.transAxes,
                       verticalalignment='top', horizontalalignment='right',
                       fontsize=7, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                ax.grid(True, alpha=0.3)
        
        # 设置整体标题
        overthreshold_count = selection_result['overthreshold_count']
        fig.suptitle(f'Physical Signals (No Overthreshold, max(CH0) <= {ch0_threshold}) - CH0-3 and CH5 Waveforms\n'
                    f'Total Physical: {selection_result["physical_count"]} events, '
                    f'Overthreshold: {overthreshold_count} events, '
                    f'Final: {final_physical_count} events, Displaying: {num_to_plot} events',
                    fontsize=13, fontweight='bold', y=0.995)
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'\n图片已保存: {save_path}')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        # 3. 绘制基线分布图
        if len(baselines) > 0:
            print(f'\n绘制基线分布图...')
            fig_baseline, ax_baseline = plt.subplots(figsize=(10, 6))
            
            # 绘制直方图
            n_bins = min(50, max(20, int(np.sqrt(len(baselines)))))  # 根据数据量自适应选择bin数量
            counts, bins, patches = ax_baseline.hist(baselines, bins=n_bins, 
                                                      color='#2E86AB', alpha=0.7, 
                                                      edgecolor='black', linewidth=0.5)
            
            # 添加统计信息
            mean_baseline = np.mean(baselines)
            std_baseline = np.std(baselines)
            median_baseline = np.median(baselines)
            min_baseline = np.min(baselines)
            max_baseline = np.max(baselines)
            
            # 标注平均值和标准差
            ax_baseline.axvline(mean_baseline, color='red', linestyle='--', linewidth=2, 
                               label=f'Mean: {mean_baseline:.2f} ADC')
            ax_baseline.axvline(median_baseline, color='green', linestyle='--', linewidth=2, 
                               label=f'Median: {median_baseline:.2f} ADC')
            
            # 添加标准差范围（±3σ）
            ax_baseline.axvspan(mean_baseline - 3*std_baseline, mean_baseline + 3*std_baseline, 
                               alpha=0.2, color='red', label=f'±3σ: {std_baseline:.2f} ADC')
            
            ax_baseline.set_xlabel('Baseline Level (ADC)', fontsize=12, fontweight='normal')
            ax_baseline.set_ylabel('Count', fontsize=12, fontweight='normal')
            ax_baseline.set_title(f'Baseline Distribution (CH0)\n'
                                 f'Total Events: {len(baselines)}, '
                                 f'Mean: {mean_baseline:.2f} ± {std_baseline:.2f} ADC, '
                                 f'Range: [{min_baseline:.2f}, {max_baseline:.2f}] ADC',
                                 fontsize=13, fontweight='bold')
            ax_baseline.legend(loc='upper right', fontsize=10)
            ax_baseline.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            # 添加统计信息文本框
            stats_text = (f'Statistics:\n'
                         f'  N = {len(baselines)}\n'
                         f'  Min = {min_baseline:.2f} ADC\n'
                         f'  Max = {max_baseline:.2f} ADC\n'
                         f'  Mean = {mean_baseline:.2f} ADC\n'
                         f'  Median = {median_baseline:.2f} ADC\n'
                         f'  Std = {std_baseline:.2f} ADC')
            ax_baseline.text(0.98, 0.98, stats_text, transform=ax_baseline.transAxes,
                           verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                                    edgecolor='gray', alpha=0.8, linewidth=0.8),
                           fontsize=10, family='monospace')
            
            plt.tight_layout()
            
            if save_path:
                # 保存基线分布图
                baseline_save_path = save_path.replace('.png', '_baseline.png').replace('.jpg', '_baseline.jpg')
                if baseline_save_path == save_path:
                    baseline_save_path = save_path + '_baseline.png'
                plt.savefig(baseline_save_path, dpi=150, bbox_inches='tight')
                print(f'基线分布图已保存: {baseline_save_path}')
            
            if show_plot:
                plt.show()
            else:
                plt.close()


if __name__ == '__main__':
    try:
        plot_physical_waveforms_no_overthreshold(
            ch0_3_file=None,  # 自动选择匹配的文件对
            ch5_file=None,
            rt_cut=6000.0,
            ch0_threshold=16382.0,  # CH0最大值阈值
            ch0_idx=0,
            ch5_idx=0,
            max_events_to_plot=5,  # 绘制前20个Physical信号
            show_plot=True
        )
    except Exception as e:
        print(f'分析失败: {e}')
        import traceback
        traceback.print_exc()
