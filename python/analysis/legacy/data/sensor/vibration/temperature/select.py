#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
振动传感器温度数据读取和筛选脚本
从detector_2_YYYY-MM-DD.h5文件中读取数据，并按日期范围筛选
从data/vibration/hdf5目录读取HDF5格式的数据文件
"""
import os
import glob
import numpy as np
import pandas as pd
import h5py
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker


def read_vibration_data(file_path: str, downsample_factor: int = 1) -> pd.DataFrame:
    """
    读取振动传感器数据文件（HDF5格式），支持降采样
    
    参数:
        file_path: HDF5数据文件路径
        downsample_factor: 降采样因子，每隔N个点读取一个（默认为1，不降采样）
                          例如：downsample_factor=10 表示每隔10个点读取1个
    
    返回:
        DataFrame，包含所有列数据，以及datetime列
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'文件不存在: {file_path}')
    
    if downsample_factor < 1:
        raise ValueError(f'降采样因子必须 >= 1，当前值: {downsample_factor}')
    
    try:
        with h5py.File(file_path, 'r') as h5f:
            # 检查必需的数据集是否存在
            required_datasets = ['time', 'datetime_str', 'Temperature']
            for ds_name in required_datasets:
                if ds_name not in h5f:
                    raise ValueError(f'文件中缺少必需的数据集: {ds_name}')
            
            # 确定要读取的索引范围（降采样）
            dataset_size = h5f['time'].shape[0]
            if downsample_factor > 1:
                # 使用切片而不是数组索引，效率更高
                indices = slice(0, dataset_size, downsample_factor)
                expected_size = (dataset_size + downsample_factor - 1) // downsample_factor
            else:
                indices = slice(None)  # 读取所有数据
                expected_size = dataset_size
            
            # 读取datetime字符串（先读取，因为需要解码）
            # 使用切片索引更高效
            datetime_str_data = h5f['datetime_str'][indices]
            
            # 将字节字符串转换为普通字符串
            if datetime_str_data.dtype.kind == 'S':  # 字节字符串
                # 使用列表推导式解码字节字符串
                datetime_str_data = [s.decode('utf-8') if isinstance(s, bytes) else s for s in datetime_str_data]
            
            # 读取时间数据（从epoch开始的秒数）- 实际上这里读取了但没有使用
            # if downsample_factor > 1:
            #     time_data = h5f['time'][indices]
            # else:
            #     time_data = h5f['time'][:]
            
            # 解析datetime
            datetime_series = pd.to_datetime(datetime_str_data, format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
            
            # 读取所有数据列
            data_dict = {'datetime': datetime_series}
            
            # 读取所有数值列
            numeric_columns = [
                'Temperature',
                'Velocity_x', 'Velocity_y', 'Velocity_z',
                'Acceleration_x', 'Acceleration_y', 'Acceleration_z',
                'Displacement_x', 'Displacement_y', 'Displacement_z',
                'Frequency_x', 'Frequency_y', 'Frequency_z'
            ]
            
            for col_name in numeric_columns:
                if col_name in h5f:
                    if downsample_factor > 1:
                        data_dict[col_name] = h5f[col_name][indices].astype(float)
                    else:
                        data_dict[col_name] = h5f[col_name][:].astype(float)
                else:
                    # 如果列不存在，用NaN填充
                    data_dict[col_name] = np.full(len(datetime_series), np.nan, dtype=float)
            
            # 创建DataFrame
            df = pd.DataFrame(data_dict)
            
            # 过滤掉datetime为NaN的行
            df = df[df['datetime'].notna()].copy()
            
            # 按时间排序
            df = df.sort_values('datetime').reset_index(drop=True)
            
            if df.empty:
                raise ValueError(f'文件中没有有效数据')
            
            return df
        
    except Exception as e:
        raise ValueError(f'读取文件 {file_path} 时出错: {e}')

def select_by_date_range_vibration(data_dir: str,
                                   detector_num = 2,
                                   start_date: str = None,
                                   end_date: str = None,
                                   start_time: Optional[str] = None,
                                   end_time: Optional[str] = None,
                                   downsample_factor: int = 1) -> Dict[str, np.ndarray]:
    """
    按日期范围筛选振动传感器数据
    参数:
        data_dir: 数据文件夹路径
        detector_num: 探测器编号，可以是单个整数（如2）或整数列表（如[1,2,3,4,5]），默认为2
        start_date: 起始日期，格式 'YYYY-MM-DD'
        end_date: 终止日期，格式 'YYYY-MM-DD'
        start_time: 起始时间（可选），格式 'HH:MM:SS'
        end_time: 终止时间（可选），格式 'HH:MM:SS'
        downsample_factor: 降采样因子，每隔N个点读取一个（默认为1，不降采样）
                          例如：downsample_factor=10 表示每隔10个点读取1个
    返回:
        字典，包含所有列的numpy数组，键名为列名。如果detector_num是列表，还会包含'detector_num'列
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f'目录不存在: {data_dir}')
    
    # 将detector_num转换为列表以便统一处理
    if isinstance(detector_num, (int, float)):
        detector_list = [int(detector_num)]
    elif isinstance(detector_num, (list, tuple)):
        detector_list = [int(d) for d in detector_num]
    else:
        raise TypeError(f'detector_num 必须是整数或整数列表，当前类型: {type(detector_num)}')
    
    # 解析起始和终止日期
    def parse_date(date_str: str) -> datetime:
        return datetime.strptime(date_str, '%Y-%m-%d')
    
    if start_date is None or end_date is None:
        # 如果没有指定日期，使用所有文件
        start_dt = None
        end_dt = None
    else:
        start_dt = parse_date(start_date)
        end_dt = parse_date(end_date)
        
        # 如果提供了时间，则添加到日期中
        if start_time:
            time_parts = start_time.split(':')
            if len(time_parts) == 3:
                start_dt = start_dt.replace(hour=int(time_parts[0]), 
                                           minute=int(time_parts[1]), 
                                           second=int(time_parts[2]))
        else:
            start_dt = start_dt.replace(hour=0, minute=0, second=0)
        
        if end_time:
            time_parts = end_time.split(':')
            if len(time_parts) == 3:
                end_dt = end_dt.replace(hour=int(time_parts[0]), 
                                       minute=int(time_parts[1]), 
                                       second=int(time_parts[2]))
        else:
            end_dt = end_dt.replace(hour=23, minute=59, second=59)
    
    # 读取所有探测器的数据
    all_dataframes = []
    total_files_read = 0
    
    print(f'\n开始读取 {len(detector_list)} 个探测器的数据: {detector_list}')
    if downsample_factor > 1:
        print(f'使用降采样因子: {downsample_factor} (每隔 {downsample_factor} 个点读取1个)')
    
    for det_idx, det_num in enumerate(detector_list, 1):
        print(f'\n[{det_idx}/{len(detector_list)}] 处理探测器 {det_num}...')
        # 构建文件模式（查找.h5文件）
        if start_date and end_date:
            # 生成日期范围内的所有日期
            date_list = []
            current_date = start_dt.date()
            end_date_obj = end_dt.date()
            
            while current_date <= end_date_obj:
                filename = f'detector_{det_num}_{current_date.strftime("%Y-%m-%d")}.h5'
                file_path = os.path.join(data_dir, filename)
                if os.path.exists(file_path):
                    date_list.append(file_path)
                current_date += timedelta(days=1)
        else:
            # 读取所有匹配的文件
            pattern = os.path.join(data_dir, f'detector_{det_num}_*.h5')
            date_list = sorted(glob.glob(pattern))
        
        # 读取当前探测器的所有文件
        print(f'  开始读取探测器 {det_num} 的文件，共 {len(date_list)} 个文件...')
        for file_idx, file_path in enumerate(date_list, 1):
            try:
                print(f'    [{file_idx}/{len(date_list)}] 正在读取: {os.path.basename(file_path)}...', end='', flush=True)
                df = read_vibration_data(file_path, downsample_factor=downsample_factor)
                
                # 如果指定了时间范围，筛选数据
                if start_dt and end_dt:
                    mask = (df['datetime'] >= start_dt) & (df['datetime'] <= end_dt)
                    df = df[mask].copy()
                
                if not df.empty:
                    # 添加探测器编号列
                    df['detector_num'] = det_num
                    all_dataframes.append(df)
                    total_files_read += 1
                    if downsample_factor > 1:
                        print(f' ✓ {len(df)} 个数据点 (降采样因子: {downsample_factor})')
                    else:
                        print(f' ✓ {len(df)} 个数据点')
                else:
                    print(f' ⚠ 数据为空（可能不在指定时间范围内）')
            except Exception as e:
                print(f' ✗ 错误: {e}')
                continue
    
    if not all_dataframes:
        print(f'警告：没有读取到有效数据')
        return {}
    
    # 合并所有DataFrame
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    combined_df = combined_df.sort_values('datetime').reset_index(drop=True)
    
    # 转换为字典，每个列对应一个numpy数组
    result = {}
    for col in combined_df.columns:
        if col == 'datetime':
            result['datetime'] = combined_df[col].values
        elif col == 'detector_num':
            result['detector_num'] = combined_df[col].values.astype(int)
        else:
            # 数值列转换为float数组
            result[col] = combined_df[col].values.astype(float)
    
    print(f'\n筛选完成：共读取 {total_files_read} 个文件，合并后共 {len(combined_df)} 条记录')
    if len(detector_list) > 1:
        print(f'包含 {len(detector_list)} 个探测器的数据: {detector_list}')
    if not combined_df.empty:
        print(f'日期范围：{combined_df["datetime"].min()} 到 {combined_df["datetime"].max()}')
    
    return result

def plot_temp_vs_datetime_vibration(data_dict: Dict[str, np.ndarray],
                                    save_path: Optional[str] = None,
                                    show_plot: bool = True,
                                    figsize: Tuple[int, int] = (12, 6)) -> None:
    """
    绘制振动传感器温度-时间图
    如果数据中包含多个探测器（detector_num列），会绘制多条曲线
    
    参数:
        data_dict: 包含数据的字典，必须包含 'datetime' 和 'Temperature' 列
                  如果包含 'detector_num' 列，会按探测器分组绘制多条曲线
        save_path: 保存图片的路径，如果为None则不保存
        show_plot: 是否显示图片
        figsize: 图片大小 (宽度, 高度)
    """
    if 'datetime' not in data_dict:
        raise ValueError("数据字典中必须包含 'datetime' 列")
    
    if 'Temperature' not in data_dict:
        raise ValueError("数据字典中必须包含 'Temperature' 列")
    
    # 获取数据
    datetime_arr = data_dict['datetime']
    temp_arr = data_dict['Temperature']
    
    # 检查是否有多个探测器
    has_multiple_detectors = 'detector_num' in data_dict
    if has_multiple_detectors:
        detector_num_arr = data_dict['detector_num']
        unique_detectors = np.unique(detector_num_arr)
        
        # 过滤NaN值，同时保留detector_num
        valid_mask = ~np.isnan(temp_arr)
        datetime_arr = datetime_arr[valid_mask]
        temp_arr = temp_arr[valid_mask]
        detector_num_arr = detector_num_arr[valid_mask]
    else:
        # 过滤NaN值
        valid_mask = ~np.isnan(temp_arr)
        datetime_arr = datetime_arr[valid_mask]
        temp_arr = temp_arr[valid_mask]
        unique_detectors = None
    
    if len(datetime_arr) == 0:
        raise ValueError("'Temperature' 列中没有有效数据（过滤后）")
    
    # 设置matplotlib参数以获得更好的科研图表样式
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'axes.linewidth': 1.2,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.minor.width': 0.8,
        'ytick.minor.width': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'figure.dpi': 100
    })
    
    # 定义颜色列表（用于多个探测器）
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4749', '#9B5DE5', '#F15BB5']
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 根据是否有多个探测器选择绘制方式
    if has_multiple_detectors and len(unique_detectors) > 1:
        # 绘制多条曲线（每个探测器一条）
        legend_labels = []
        all_stats = []
        
        for i, det_num in enumerate(sorted(unique_detectors)):
            det_mask = detector_num_arr == det_num
            det_datetime = datetime_arr[det_mask]
            det_temp = temp_arr[det_mask]
            
            if len(det_datetime) > 0:
                color = colors[i % len(colors)]
                ax.plot(det_datetime, det_temp,
                        linewidth=1.5,
                        alpha=0.85,
                        color=color,
                        label=f'Detector {det_num}')
                legend_labels.append(f'Detector {det_num}')
                
                # 计算统计信息
                mean_val = np.mean(det_temp)
                std_val = np.std(det_temp)
                min_val = np.min(det_temp)
                max_val = np.max(det_temp)
                all_stats.append((det_num, len(det_temp), min_val, max_val, mean_val, std_val))
        
        # 添加图例
        ax.legend(legend_labels, loc='upper right', framealpha=0.9, edgecolor='gray',
                  frameon=True, fancybox=False, shadow=False)
        
        # 添加统计信息框（显示所有探测器的统计信息）
        stats_text = 'Statistics:\n'
        for det_num, n, min_val, max_val, mean_val, std_val in all_stats:
            stats_text += (f'Det {det_num}: N={n}, '
                          f'Mean={mean_val:.2f}°C, '
                          f'Std={std_val:.2f}°C\n')
        stats_text = stats_text.rstrip('\n')
        
        # 计算整体的温度范围用于y轴
        all_temps = temp_arr
    else:
        # 单个探测器或没有detector_num列的情况，绘制单条曲线
        color = colors[0] if has_multiple_detectors else '#2E86AB'
        label = f'Detector {unique_detectors[0]}' if has_multiple_detectors else 'Temperature'
        ax.plot(datetime_arr, temp_arr,
                linewidth=1.5,
                alpha=0.85,
                color=color,
                label=label)
        
        # 计算统计信息
        mean_val = np.mean(temp_arr)
        std_val = np.std(temp_arr)
        min_val = np.min(temp_arr)
        max_val = np.max(temp_arr)
        
        # 添加图例
        ax.legend([label], loc='upper right', framealpha=0.9, edgecolor='gray',
                  frameon=True, fancybox=False, shadow=False)
        
        # 添加统计信息框
        stats_text = (f'N = {len(temp_arr)}\n'
                     f'Min = {min_val:.2f} °C\n'
                     f'Max = {max_val:.2f} °C\n'
                     f'Mean = {mean_val:.2f} °C\n'
                     f'Std = {std_val:.2f} °C')
        
        all_temps = temp_arr
    
    # 设置标签和标题（使用英文）
    ax.set_xlabel('Time', fontsize=13, fontweight='normal')
    ax.set_ylabel('Temperature (°C)', fontsize=13, fontweight='normal')
    
    # 格式化x轴日期 - 根据时间范围自动选择格式
    time_span = datetime_arr[-1] - datetime_arr[0]
    
    # 转换为timedelta（处理numpy datetime类型）
    if hasattr(time_span, 'days'):
        days = time_span.days
        total_hours = time_span.total_seconds() / 3600
    else:
        # 如果是numpy timedelta64
        days = time_span.astype('timedelta64[D]').astype(int)
        total_hours = time_span.astype('timedelta64[h]').astype(int)
    
    if days > 30:
        # 超过30天，只显示日期
        date_format = '%Y-%m-%d'
        locator = mdates.DayLocator(interval=max(1, days // 10))
        minor_locator = mdates.HourLocator(interval=6)  # 每6小时一个次要刻度
    elif days > 1:
        # 1-30天，显示日期和时间
        date_format = '%m-%d %H:%M'
        major_interval = max(1, int(total_hours / 10))
        locator = mdates.HourLocator(interval=major_interval)
        minor_locator = mdates.MinuteLocator(interval=30)  # 每30分钟一个次要刻度
    else:
        # 小于1天，只显示时间
        date_format = '%H:%M'
        major_interval = max(1, int(total_hours / 8))
        locator = mdates.HourLocator(interval=major_interval)
        minor_locator = mdates.MinuteLocator(interval=15)  # 每15分钟一个次要刻度
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_minor_locator(minor_locator)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 设置y轴格式，确保显示真实值而不是偏移量
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
    ax.yaxis.get_major_formatter().set_scientific(False)
    
    # 添加y轴的次要刻度
    temp_range = np.max(all_temps) - np.min(all_temps)
    if temp_range > 0:
        # 根据温度范围自动设置次要刻度间隔
        minor_interval = temp_range / 20
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(minor_interval))
    
    # 改进网格线样式
    ax.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.3, color='gray')
    ax.grid(True, which='minor', linestyle='--', linewidth=0.5, alpha=0.2, color='gray')
    
    # 设置坐标轴边框样式
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_color('gray')
    ax.spines['right'].set_color('gray')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    
    # 在图的角落添加统计信息框
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                     edgecolor='gray', alpha=0.8, linewidth=0.8),
            fontsize=9,
            family='monospace')
    
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'图片已保存至: {save_path}')
    
    # 显示图片
    if show_plot:
        plt.show()
    else:
        plt.close()

# 示例使用
if __name__ == '__main__':
    # 获取项目根目录
    # 从 python/data/sensor/vibration/temperature/select.py 向上5层到达项目根目录
    # temperature -> vibration -> sensor -> data -> python -> DeepVibration (项目根目录)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))))
    data_dir = os.path.join(project_root, 'data', 'vibration', 'hdf5')
    
    print('=' * 70)
    print('振动传感器温度数据读取和筛选示例')
    print('=' * 70)
    
    # 示例: 读取并筛选数据（多个探测器）
    print('\n按日期范围筛选数据（多个探测器）')
    print('-' * 70)
    try:
        # 筛选指定日期范围的数据（多个探测器）
        # downsample_factor=10 表示每隔10个点读取1个，可以显著减少数据量
        data_multi = select_by_date_range_vibration(data_dir,
                                                    detector_num=[1, 2, 3, 4, 5],
                                                    start_date='2025-05-28',
                                                    end_date='2025-06-10',
                                                    downsample_factor=100)  # 降采样，每隔10个点读取1个
        if data_multi:
            print(f'\n筛选结果包含以下列: {list(data_multi.keys())}')
            print(f'数据点数量: {len(data_multi["datetime"])}')
            if 'detector_num' in data_multi:
                unique_detectors = np.unique(data_multi['detector_num'])
                print(f'包含探测器: {unique_detectors}')
                print(f'各探测器数据点数量:')
                for det_num in sorted(unique_detectors):
                    det_count = np.sum(data_multi['detector_num'] == det_num)
                    print(f'  探测器 {det_num}: {det_count} 个数据点')
            
            # 绘制多个探测器的温度图
            print('\n绘制多个探测器的温度-时间图')
            print('-' * 70)
            plot_temp_vs_datetime_vibration(data_multi, show_plot=True)
        else:
            print('警告：未能读取到任何数据')
    
    except Exception as e:
        print(f'错误: {e}')
        import traceback
        traceback.print_exc()

