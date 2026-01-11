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


def read_vibration_data(file_path: str) -> pd.DataFrame:
    """
    读取振动传感器数据文件（HDF5格式）
    
    参数:
        file_path: HDF5数据文件路径
    
    返回:
        DataFrame，包含所有列数据，以及datetime列
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'文件不存在: {file_path}')
    
    try:
        with h5py.File(file_path, 'r') as h5f:
            # 检查必需的数据集是否存在
            required_datasets = ['time', 'datetime_str', 'Temperature']
            for ds_name in required_datasets:
                if ds_name not in h5f:
                    raise ValueError(f'文件中缺少必需的数据集: {ds_name}')
            
            # 读取时间数据（从epoch开始的秒数）
            time_data = h5f['time'][:]
            
            # 读取datetime字符串
            datetime_str_data = h5f['datetime_str'][:]
            # 将字节字符串转换为普通字符串
            if datetime_str_data.dtype.kind == 'S':  # 字节字符串
                datetime_str_data = [s.decode('utf-8') for s in datetime_str_data]
            
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
                                   detector_num: int = 2,
                                   start_date: str = None,
                                   end_date: str = None,
                                   start_time: Optional[str] = None,
                                   end_time: Optional[str] = None) -> Dict[str, np.ndarray]:
    """
    按日期范围筛选振动传感器数据
    
    参数:
        data_dir: 数据文件夹路径
        detector_num: 探测器编号，默认为2
        start_date: 起始日期，格式 'YYYY-MM-DD'
        end_date: 终止日期，格式 'YYYY-MM-DD'
        start_time: 起始时间（可选），格式 'HH:MM:SS'
        end_time: 终止时间（可选），格式 'HH:MM:SS'
    
    返回:
        字典，包含所有列的numpy数组，键名为列名
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f'目录不存在: {data_dir}')
    
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
    
    # 构建文件模式（查找.h5文件）
    if start_date and end_date:
        # 生成日期范围内的所有日期
        date_list = []
        current_date = start_dt.date()
        end_date_obj = end_dt.date()
        
        while current_date <= end_date_obj:
            filename = f'detector_{detector_num}_{current_date.strftime("%Y-%m-%d")}.h5'
            file_path = os.path.join(data_dir, filename)
            if os.path.exists(file_path):
                date_list.append(file_path)
            current_date += timedelta(days=1)
    else:
        # 读取所有匹配的文件
        pattern = os.path.join(data_dir, f'detector_{detector_num}_*.h5')
        date_list = sorted(glob.glob(pattern))
    
    if not date_list:
        print(f'警告：在指定日期范围内没有找到匹配的文件')
        return {}
    
    # 读取所有文件并合并数据
    all_dataframes = []
    
    for file_path in date_list:
        try:
            df = read_vibration_data(file_path)
            
            # 如果指定了时间范围，筛选数据
            if start_dt and end_dt:
                mask = (df['datetime'] >= start_dt) & (df['datetime'] <= end_dt)
                df = df[mask].copy()
            
            if not df.empty:
                all_dataframes.append(df)
                print(f'已读取文件: {os.path.basename(file_path)}, 数据点数: {len(df)}')
        except Exception as e:
            print(f'警告：读取文件 {os.path.basename(file_path)} 时出错: {e}')
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
        else:
            # 数值列转换为float数组
            result[col] = combined_df[col].values.astype(float)
    
    print(f'\n筛选完成：共读取 {len(date_list)} 个文件，合并后共 {len(combined_df)} 条记录')
    if not combined_df.empty:
        print(f'日期范围：{combined_df["datetime"].min()} 到 {combined_df["datetime"].max()}')
    
    return result

def plot_temp_vs_datetime_vibration(data_dict: Dict[str, np.ndarray],
                                    save_path: Optional[str] = None,
                                    show_plot: bool = True,
                                    figsize: Tuple[int, int] = (12, 6)) -> None:
    """
    绘制振动传感器温度-时间图
    模仿 compressor/select.py 中的 plot_temp_vs_datetime 函数
    
    参数:
        data_dict: 包含数据的字典，必须包含 'datetime' 和 'Temperature' 列
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
    
    # 过滤NaN值和温度小于20的点
    valid_mask = (~np.isnan(temp_arr)) & (temp_arr >= 20.0)
    datetime_arr = datetime_arr[valid_mask]
    temp_arr = temp_arr[valid_mask]
    
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
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制数据，使用更专业的颜色和线型
    ax.plot(datetime_arr, temp_arr,
            linewidth=1.5,
            alpha=0.85,
            color='#2E86AB',
            label='Temperature')
    
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
    temp_range = np.max(temp_arr) - np.min(temp_arr)
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
    
    # 计算统计信息
    mean_val = np.mean(temp_arr)
    std_val = np.std(temp_arr)
    min_val = np.min(temp_arr)
    max_val = np.max(temp_arr)
    
    # 添加图例（只显示数据系列名称）
    ax.legend(['Temperature'], loc='upper right', framealpha=0.9, edgecolor='gray',
              frameon=True, fancybox=False, shadow=False)
    
    # 在图的角落添加统计信息框
    stats_text = (f'N = {len(temp_arr)}\n'
                 f'Min = {min_val:.2f} °C\n'
                 f'Max = {max_val:.2f} °C\n'
                 f'Mean = {mean_val:.2f} °C\n'
                 f'Std = {std_val:.2f} °C')
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
    
    # 示例1: 读取并筛选数据
    print('\n示例1: 按日期范围筛选数据')
    print('-' * 70)
    try:
        # 筛选指定日期范围的数据
        data = select_by_date_range_vibration(data_dir,
                                             detector_num=2,
                                             start_date='2025-05-28',
                                             end_date='2025-06-10')
        if data:
            print(f'\n筛选结果包含以下列: {list(data.keys())}')
            print(f'数据点数量: {len(data["datetime"])}')
            
            # 示例2: 绘制温度图
            print('\n示例2: 绘制温度-时间图')
            print('-' * 70)
            plot_temp_vs_datetime_vibration(data, show_plot=True)
            
    except Exception as e:
        print(f'错误: {e}')
        import traceback
        traceback.print_exc()

