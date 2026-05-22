#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bin文件Inhibit事件计数率（CH0 min == 0）和制冷机Controller温度联合绘制脚本
从bin文件读取Inhibit事件（CH0最小值等于0）的时间，计算计数率，并与制冷机Controller温度绘制在一起
"""
import os
import sys
import struct
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from datetime import datetime
from typing import Optional, Tuple, List
import importlib.util
import h5py
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor, as_completed

# 添加路径以便导入其他模块
current_dir = os.path.dirname(os.path.abspath(__file__))
# current_dir = python/data/temperature/countrate/inhibit
parent_dir = os.path.dirname(current_dir)  # python/data/temperature/countrate
grandparent_dir = os.path.dirname(parent_dir)  # python/data/temperature
great_grandparent_dir = os.path.dirname(grandparent_dir)  # python/data
project_root = os.path.dirname(os.path.dirname(great_grandparent_dir))  # 项目根目录

# 导入time.py模块（使用sys.path）
python_dir = os.path.dirname(great_grandparent_dir)  # python
utils_dir = os.path.join(python_dir, 'utils')
if utils_dir not in sys.path:
    sys.path.insert(0, utils_dir)
# 注意：这里只是参考time.py的算法，不需要导入read_bin_time_span

# 导入制冷机模块
compressor_select_path = os.path.join(great_grandparent_dir, 'compressor', 'select.py')
spec_compressor = importlib.util.spec_from_file_location("compressor_select", compressor_select_path)
compressor_select = importlib.util.module_from_spec(spec_compressor)
spec_compressor.loader.exec_module(compressor_select)
select_compressor_by_date_range = compressor_select.select_by_date_range


def read_hdf5_event_times(hdf5_file_path: str,
                          epoch_offset: float = 2.082816000000000e+09) -> np.ndarray:
    """
    从HDF5文件读取所有event的时间并转换为datetime
    
    参数:
        hdf5_file_path: HDF5文件路径
        epoch_offset: 时间戳偏移量（默认值来自MATLAB代码）
    
    返回:
        datetime数组，包含所有event的时间
    """
    if not os.path.exists(hdf5_file_path):
        raise FileNotFoundError(f'文件不存在: {hdf5_file_path}')
    
    with h5py.File(hdf5_file_path, 'r') as f:
        if 'time_data' not in f:
            raise ValueError(f'HDF5文件中缺少 time_data 数据集')
        
        time_data = f['time_data'][:]
        
        if len(time_data) == 0:
            raise ValueError(f'HDF5文件中 time_data 为空')
        
        # 转换为datetime
        epoch_start = datetime(1970, 1, 1)
        event_times = []
        
        for time_val in time_data:
            eventtime = time_val - epoch_offset
            event_datetime = epoch_start + pd.Timedelta(seconds=eventtime)
            event_times.append(event_datetime)
        
        return np.array(event_times)

def read_bin_event_times(bin_file_path: str,
                         epoch_offset: float = 2.082816000000000e+09,
                         max_events: Optional[int] = None) -> np.ndarray:
    """
    从bin文件读取所有event的时间并转换为datetime
    
    参数:
        bin_file_path: bin文件路径
        epoch_offset: 时间戳偏移量（默认值来自MATLAB代码）
        max_events: 最多读取的事件数，None表示读取所有
    
    返回:
        datetime数组，包含所有event的时间
    """
    if not os.path.exists(bin_file_path):
        raise FileNotFoundError(f'文件不存在: {bin_file_path}')
    
    fid = None
    try:
        fid = open(bin_file_path, 'rb')
        
        # 读取 Run Header
        pstt = struct.unpack('<d', fid.read(8))[0]  # Program Start Time (double, 8 bytes)
        
        # 跳过 V1725-1 Channel DAC (16个uint32，每个4字节)
        fid.read(16 * 4)
        
        # 读取 Time Window
        V1725_1_twd = struct.unpack('<I', fid.read(4))[0]  # Time Window (uint32, 4 bytes)
        
        # 跳过其他 Run Header 信息
        fid.read(4)  # Pre Trigger
        fid.read(4)  # Opened Channel
        
        # 读取 Run Start Time
        rstt = struct.unpack('<d', fid.read(8))[0]  # Run Start Time (double, 8 bytes)
        
        # 读取所有event的时间
        event_times = []
        event_header_size = 4 * 5  # 5个uint32字段
        event_data_size = 16 * V1725_1_twd * 2  # 16个通道的数据
        
        event_count = 0
        while True:
            if max_events is not None and event_count >= max_events:
                break
            
            # 读取 Hit_pat
            hit_pat_bytes = fid.read(4)
            if len(hit_pat_bytes) < 4:
                break
            
            # 跳过 V1729_tg_rec
            fid.read(4)
            
            # 读取 Evt_endtime
            evt_endtime_bytes = fid.read(4)
            if len(evt_endtime_bytes) < 4:
                break
            evt_endtime = struct.unpack('<I', evt_endtime_bytes)[0]
            
            # 跳过 V1725_1_tgno 和 V1725_1_tag
            fid.read(8)
            
            # 跳过所有通道的数据
            channel_data_bytes = fid.read(event_data_size)
            if len(channel_data_bytes) < event_data_size:
                break
            
            # 计算时间（按照time.py的逻辑）
            # time_data = evt_endtime / 1000.0 + pstt
            time_data = evt_endtime / 1000.0 + pstt
            
            # 转换为datetime
            # eventtime = time_data - epoch_offset
            # time_transformed = epochStart + seconds(eventtime)
            epoch_start = datetime(1970, 1, 1)
            eventtime = time_data - epoch_offset
            event_datetime = epoch_start + pd.Timedelta(seconds=eventtime)
            
            event_times.append(event_datetime)
            event_count += 1
        
        if len(event_times) == 0:
            raise ValueError('未能读取到任何event时间')
        
        return np.array(event_times)
        
    finally:
        if fid is not None:
            fid.close()

def calculate_count_rate(event_times: np.ndarray,
                        time_bin_size: str = '1h') -> Tuple[np.ndarray, np.ndarray]:
    """
    计算事件计数率（按时间bin分组）
    
    参数:
        event_times: event时间数组（datetime）
        time_bin_size: 时间bin大小，pandas频率字符串（如'1h', '30min', '1D'）
    
    返回:
        (bin_centers, counts): bin中心时间数组和计数数组
    """
    # 使用pandas的resample功能
    df = pd.DataFrame({'datetime': event_times})
    df.set_index('datetime', inplace=True)
    df['count'] = 1  # 每个event计数为1
    
    # 按时间bin分组并计数
    resampled = df.resample(time_bin_size).count()
    
    bin_centers = resampled.index.to_pydatetime()
    counts = resampled['count'].values
    
    return bin_centers, counts

def read_hdf5_inhibit_event_times(hdf5_file_path: str,
                                  ch0_idx: int = 0,
                                  epoch_offset: float = 2.082816000000000e+09) -> np.ndarray:
    """
    从HDF5文件读取Inhibit事件（CH0 min == 0）的时间并转换为datetime
    
    参数:
        hdf5_file_path: HDF5文件路径
        ch0_idx: CH0通道索引，默认0
        epoch_offset: 时间戳偏移量（默认值来自MATLAB代码）
    
    返回:
        datetime数组，包含所有Inhibit事件的时间
    """
    if not os.path.exists(hdf5_file_path):
        raise FileNotFoundError(f'文件不存在: {hdf5_file_path}')
    
    with h5py.File(hdf5_file_path, 'r') as f:
        if 'time_data' not in f:
            raise ValueError(f'HDF5文件中缺少 time_data 数据集')
        if 'channel_data' not in f:
            raise ValueError(f'HDF5文件中缺少 channel_data 数据集')
        
        time_data = f['time_data'][:]
        channel_data = f['channel_data']
        time_samples, num_channels, num_events = channel_data.shape
        
        if ch0_idx >= num_channels:
            raise IndexError(f'CH0通道索引 {ch0_idx} 超出范围 [0, {num_channels-1}]')
        
        # 计算所有事件的CH0最小值（批量处理以提高效率）
        ch0_min_values = np.zeros(num_events, dtype=np.float64)
        batch_size = 5000  # 增大batch_size以提高效率
        
        for i in range(0, num_events, batch_size):
            end_idx = min(i + batch_size, num_events)
            batch_data = channel_data[:, ch0_idx, i:end_idx]
            ch0_min_values[i:end_idx] = np.min(batch_data, axis=0)
        
        # 筛选Inhibit信号（CH0 min == 0）
        inhibit_mask = ch0_min_values == 0
        inhibit_indices = np.where(inhibit_mask)[0]
        inhibit_count = len(inhibit_indices)
        
        if inhibit_count == 0:
            return np.array([])
        
        # 获取Inhibit事件的时间
        inhibit_time_data = time_data[inhibit_indices]
        
        # 批量转换为datetime（使用向量化操作，比循环快得多）
        epoch_start = datetime(1970, 1, 1)
        eventtime = inhibit_time_data - epoch_offset
        # 使用pandas的to_datetime进行批量转换
        event_times = epoch_start + pd.to_timedelta(eventtime, unit='s')
        
        # 转换为numpy datetime64数组以确保类型一致
        if hasattr(event_times, 'values'):
            return pd.to_datetime(event_times.values).values
        else:
            return pd.to_datetime(event_times).values


def _process_single_file_inhibit(bin_file_path: str,
                                 hdf5_dir: Optional[str],
                                 ch0_idx: int,
                                 epoch_offset: float) -> Tuple[str, Optional[np.ndarray]]:
    """
    处理单个文件的辅助函数（用于多进程并行处理）
    
    参数:
        bin_file_path: bin文件路径
        hdf5_dir: HDF5文件目录（如果提供，会优先从HDF5读取）
        ch0_idx: CH0通道索引
        epoch_offset: 时间戳偏移量
    
    返回:
        (bin_filename, event_times): 文件名和事件时间数组，如果失败则返回None
    """
    bin_filename = os.path.basename(bin_file_path)
    
    # 尝试从HDF5文件读取（如果提供hdf5_dir）
    event_times = None
    if hdf5_dir is not None:
        # 构造HDF5文件路径：{bin_filename}_processed.h5
        hdf5_filename = f'{os.path.splitext(bin_filename)[0]}_processed.h5'
        hdf5_path = os.path.join(hdf5_dir, 'CH0-3', hdf5_filename)
        
        if os.path.exists(hdf5_path):
            try:
                event_times = read_hdf5_inhibit_event_times(hdf5_path, ch0_idx, epoch_offset)
            except Exception:
                # 如果HDF5读取失败，尝试从bin文件读取
                try:
                    event_times = read_bin_inhibit_event_times(bin_file_path, ch0_idx, epoch_offset)
                except Exception:
                    return (bin_filename, None)
    
    # 如果HDF5读取失败或不存在，从bin文件读取
    if event_times is None or len(event_times) == 0:
        try:
            event_times = read_bin_inhibit_event_times(bin_file_path, ch0_idx, epoch_offset)
        except Exception:
            return (bin_filename, None)
    
    return (bin_filename, event_times if len(event_times) > 0 else None)


def read_bin_inhibit_event_times(bin_file_path: str,
                                 ch0_idx: int = 0,
                                 epoch_offset: float = 2.082816000000000e+09,
                                 max_events: Optional[int] = None) -> np.ndarray:
    """
    从bin文件读取Inhibit事件（CH0 min == 0）的时间并转换为datetime
    
    参数:
        bin_file_path: bin文件路径
        ch0_idx: CH0通道索引，默认0（在bin文件中，CH0通常是第一个通道）
        epoch_offset: 时间戳偏移量（默认值来自MATLAB代码）
        max_events: 最多读取的事件数，None表示读取所有
    
    返回:
        datetime数组，包含所有Inhibit事件的时间
    """
    if not os.path.exists(bin_file_path):
        raise FileNotFoundError(f'文件不存在: {bin_file_path}')
    
    fid = None
    try:
        fid = open(bin_file_path, 'rb')
        
        # 读取 Run Header
        pstt = struct.unpack('<d', fid.read(8))[0]  # Program Start Time (double, 8 bytes)
        
        # 跳过 V1725-1 Channel DAC (16个uint32，每个4字节)
        fid.read(16 * 4)
        
        # 读取 Time Window
        V1725_1_twd = struct.unpack('<I', fid.read(4))[0]  # Time Window (uint32, 4 bytes)
        
        # 跳过其他 Run Header 信息
        fid.read(4)  # Pre Trigger
        fid.read(4)  # Opened Channel
        
        # 读取 Run Start Time
        rstt = struct.unpack('<d', fid.read(8))[0]  # Run Start Time (double, 8 bytes)
        
        # 读取所有event的时间并筛选Inhibit信号
        event_times = []
        event_header_size = 4 * 5  # 5个uint32字段
        event_data_size = 16 * V1725_1_twd * 2  # 16个通道的数据
        
        event_count = 0
        while True:
            if max_events is not None and event_count >= max_events:
                break
            
            # 读取 Hit_pat
            hit_pat_bytes = fid.read(4)
            if len(hit_pat_bytes) < 4:
                break
            
            # 跳过 V1729_tg_rec
            fid.read(4)
            
            # 读取 Evt_endtime
            evt_endtime_bytes = fid.read(4)
            if len(evt_endtime_bytes) < 4:
                break
            evt_endtime = struct.unpack('<I', evt_endtime_bytes)[0]
            
            # 跳过 V1725_1_tgno 和 V1725_1_tag
            fid.read(8)
            
            # 读取所有通道的数据
            channel_data_bytes = fid.read(event_data_size)
            if len(channel_data_bytes) < event_data_size:
                break
            
            # 解析通道数据（uint16，小端）
            channel_data = np.frombuffer(channel_data_bytes, dtype='<u2')
            # 重新整形为 (16, V1725_1_twd)
            channel_data = channel_data.reshape(16, V1725_1_twd)
            
            # 检查CH0的最小值是否等于0
            ch0_data = channel_data[ch0_idx, :]
            ch0_min = np.min(ch0_data)
            
            # 只保存Inhibit事件（CH0 min == 0）
            if ch0_min == 0:
                # 计算时间
                time_data = evt_endtime / 1000.0 + pstt
                
                # 转换为datetime
                epoch_start = datetime(1970, 1, 1)
                eventtime = time_data - epoch_offset
                event_datetime = epoch_start + pd.Timedelta(seconds=eventtime)
                
                event_times.append(event_datetime)
            
            event_count += 1
        
        if len(event_times) == 0:
            print(f'  警告：未找到任何Inhibit事件（CH0 min == 0）')
            return np.array([])
        
        return np.array(event_times)
        
    finally:
        if fid is not None:
            fid.close()


def read_all_bin_files_inhibit_event_times(bin_dir: str,
                                           hdf5_dir: Optional[str] = None,
                                           ch0_idx: int = 0,
                                           epoch_offset: float = 2.082816000000000e+09,
                                           max_workers: Optional[int] = None) -> np.ndarray:
    """
    读取bin文件夹下所有bin文件的Inhibit事件时间（CH0 min == 0）
    使用多进程并行处理以加速
    
    参数:
        bin_dir: bin文件目录
        hdf5_dir: HDF5文件目录（如果提供，会优先从HDF5读取）
        ch0_idx: CH0通道索引，默认0
        epoch_offset: 时间戳偏移量（默认值来自MATLAB代码）
        max_workers: 最大并行进程数，None表示使用所有CPU核心
    
    返回:
        合并后的datetime数组，包含所有bin文件的Inhibit事件时间
    """
    # 查找所有bin文件
    bin_files = glob.glob(os.path.join(bin_dir, '*.bin'))
    bin_files.sort()  # 按文件名排序
    
    if len(bin_files) == 0:
        raise ValueError(f'在目录 {bin_dir} 中未找到任何bin文件')
    
    print(f'找到 {len(bin_files)} 个bin文件')
    
    # 确定并行进程数
    if max_workers is None:
        max_workers = os.cpu_count()
    print(f'使用 {max_workers} 个CPU核心进行并行处理')
    
    all_event_times = []
    success_count = 0
    fail_count = 0
    
    # 使用多进程并行处理
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_file = {
            executor.submit(_process_single_file_inhibit, bin_file_path, hdf5_dir, ch0_idx, epoch_offset): bin_file_path
            for bin_file_path in bin_files
        }
        
        # 收集结果
        for future in as_completed(future_to_file):
            bin_file_path = future_to_file[future]
            bin_filename = os.path.basename(bin_file_path)
            try:
                filename, event_times = future.result()
                if event_times is not None and len(event_times) > 0:
                    all_event_times.extend(event_times)
                    success_count += 1
                    print(f'[{success_count}/{len(bin_files)}] ✓ 成功: {bin_filename} - {len(event_times)} 个Inhibit events')
                else:
                    fail_count += 1
                    print(f'[{success_count + fail_count}/{len(bin_files)}] ✗ 跳过: {bin_filename} - 未找到Inhibit事件')
            except Exception as e:
                fail_count += 1
                print(f'[{success_count + fail_count}/{len(bin_files)}] ✗ 错误: {bin_filename} - {e}')
    
    if len(all_event_times) == 0:
        raise ValueError('未能从任何文件中读取到Inhibit事件时间')
    
    # 转换为numpy datetime64数组并排序（确保类型一致）
    all_event_times = pd.to_datetime(all_event_times).values
    all_event_times = np.sort(all_event_times)
    
    print(f'\n处理完成统计:')
    print(f'  成功: {success_count} 个文件')
    print(f'  失败/跳过: {fail_count} 个文件')
    print(f'  总共读取到 {len(all_event_times)} 个Inhibit events (CH0 min == 0)')
    print(f'  时间范围: {all_event_times[0]} 到 {all_event_times[-1]}')
    
    return all_event_times

def plot_countrate_and_temperature(bin_dir: str,
                                   compressor_file_path: str,
                                   hdf5_dir: Optional[str] = None,
                                   ch0_idx: int = 0,
                                   time_bin_size: str = '1h',
                                   start_date: Optional[str] = None,
                                   end_date: Optional[str] = None,
                                   start_time: Optional[str] = None,
                                   end_time: Optional[str] = None,
                                   save_path: Optional[str] = None,
                                   show_plot: bool = True,
                                   figsize: Tuple[int, int] = (14, 7),
                                   epoch_offset: float = 2.082816000000000e+09) -> None:
    """
    绘制bin文件Inhibit事件计数率（CH0 min == 0）和制冷机Controller温度
    
    参数:
        bin_dir: bin文件目录
        compressor_file_path: 制冷机数据文件路径
        hdf5_dir: HDF5文件目录（如果提供，会优先从HDF5读取）
        ch0_idx: CH0通道索引，默认0
        time_bin_size: 时间bin大小，pandas频率字符串（如'1h', '30min', '1D'）
        start_date: 起始日期，格式 'YYYY-MM-DD'
        end_date: 终止日期，格式 'YYYY-MM-DD'
        start_time: 起始时间（可选），格式 'HH:MM:SS'
        end_time: 终止时间（可选），格式 'HH:MM:SS'
        save_path: 保存图片的路径，如果为None则不保存
        show_plot: 是否显示图片
        figsize: 图片大小 (宽度, 高度)
        epoch_offset: 时间戳偏移量（默认值来自MATLAB代码）
    """
    print('=' * 70)
    print('读取所有bin文件的Inhibit事件时间（CH0 min == 0）...')
    print('-' * 70)
    
    # 读取所有bin文件的Inhibit事件时间
    all_event_times = read_all_bin_files_inhibit_event_times(bin_dir, hdf5_dir, ch0_idx, epoch_offset)
    
    # 按日期范围筛选事件时间
    if start_date is not None or end_date is not None:
        print(f'\n按日期范围筛选: {start_date} 到 {end_date}')
        start_datetime = None
        end_datetime = None
        
        if start_date is not None:
            start_date_str = start_date
            if start_time is not None:
                start_date_str = f'{start_date} {start_time}'
            start_datetime = pd.to_datetime(start_date_str)
        
        if end_date is not None:
            end_date_str = end_date
            if end_time is not None:
                end_date_str = f'{end_date} {end_time}'
            end_datetime = pd.to_datetime(end_date_str)
        
        # 筛选时间范围
        mask = np.ones(len(all_event_times), dtype=bool)
        if start_datetime is not None:
            mask = mask & (all_event_times >= start_datetime)
        if end_datetime is not None:
            mask = mask & (all_event_times <= end_datetime)
        
        all_event_times = all_event_times[mask]
        print(f'筛选后剩余 {len(all_event_times)} 个events')
        
        if len(all_event_times) == 0:
            raise ValueError('筛选后没有剩余的事件')
    
    # 计算计数率
    print(f'\n计算计数率（时间bin: {time_bin_size}）...')
    bin_centers, counts = calculate_count_rate(all_event_times, time_bin_size)
    print(f'生成了 {len(bin_centers)} 个时间bin')
    
    # 读取制冷机数据
    print('\n' + '=' * 70)
    print('读取制冷机数据...')
    print('-' * 70)
    compressor_data = select_compressor_by_date_range(
        compressor_file_path,
        start_date=start_date,
        end_date=end_date,
        start_time=start_time,
        end_time=end_time
    )
    
    if not compressor_data:
        raise ValueError('未能读取到制冷机数据')
    
    # 获取Controller温度数据
    compressor_datetime = compressor_data['datetime']
    controller_temp = compressor_data['Controller temp']
    
    # 过滤NaN值
    controller_valid_mask = ~np.isnan(controller_temp)
    compressor_datetime = compressor_datetime[controller_valid_mask]
    controller_temp = controller_temp[controller_valid_mask]
    
    if len(compressor_datetime) == 0:
        raise ValueError("制冷机Controller温度数据中没有有效数据")
    
    # 设置matplotlib参数
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
    
    # 创建图形和左y轴（计数率）
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # 绘制计数率（柱状图）
    ax1.bar(bin_centers, counts, width=pd.Timedelta(time_bin_size) * 0.8,
            color='#2E86AB', alpha=0.7, label='Inhibit Count (CH0 min == 0)')
    ax1.set_xlabel('Time', fontsize=13, fontweight='normal')
    ax1.set_ylabel('Inhibit Count', fontsize=13, fontweight='normal', color='#2E86AB')
    ax1.tick_params(axis='y', labelcolor='#2E86AB')
    
    # 创建右y轴（温度）
    ax2 = ax1.twinx()
    ax2.plot(compressor_datetime, controller_temp,
             linewidth=1.5, alpha=0.85, color='#C73E1D',
             label='Controller Temperature')
    ax2.set_ylabel('Controller Temperature (°C)', fontsize=13, fontweight='normal', color='#C73E1D')
    ax2.tick_params(axis='y', labelcolor='#C73E1D')
    
    # 格式化x轴日期
    # 使用所有数据集的共同时间范围
    # 确保所有datetime都是相同类型（转换为numpy datetime64）
    bin_centers_dt64 = pd.to_datetime(bin_centers).values
    compressor_datetime_dt64 = pd.to_datetime(compressor_datetime).values
    
    all_datetime = np.concatenate([bin_centers_dt64, compressor_datetime_dt64])
    all_datetime = np.sort(all_datetime)
    
    if len(all_datetime) > 1:
        time_span = all_datetime[-1] - all_datetime[0]
        
        # 转换为timedelta
        if hasattr(time_span, 'days'):
            days = time_span.days
            total_hours = time_span.total_seconds() / 3600
        else:
            days = time_span.astype('timedelta64[D]').astype(int)
            total_hours = time_span.astype('timedelta64[h]').astype(int)
        
        if days > 30:
            date_format = '%m-%d %H:%M'
            locator = mdates.HourLocator(byhour=[6, 12, 18])
            minor_locator = mdates.HourLocator(interval=3)
        elif days > 1:
            date_format = '%m-%d %H:%M'
            locator = mdates.HourLocator(byhour=[6, 12, 18])
            minor_locator = mdates.HourLocator(interval=3)
        else:
            date_format = '%H:%M'
            locator = mdates.HourLocator(byhour=[0, 6, 12, 18])
            minor_locator = mdates.HourLocator(interval=3)
        
        ax1.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_minor_locator(minor_locator)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 设置y轴格式
    ax1.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
    ax1.yaxis.get_major_formatter().set_scientific(False)
    ax2.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
    ax2.yaxis.get_major_formatter().set_scientific(False)
    
    # 网格
    ax1.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.3, color='gray')
    ax1.grid(True, which='minor', linestyle='--', linewidth=0.5, alpha=0.2, color='gray')
    
    # 坐标轴边框
    ax1.spines['top'].set_visible(True)
    ax1.spines['right'].set_visible(True)
    ax1.spines['top'].set_color('gray')
    ax1.spines['right'].set_color('gray')
    
    # 计算统计信息
    count_mean = np.mean(counts)
    count_std = np.std(counts)
    count_min = np.min(counts)
    count_max = np.max(counts)
    total_events = np.sum(counts)
    
    controller_mean = np.mean(controller_temp)
    controller_std = np.std(controller_temp)
    controller_min = np.min(controller_temp)
    controller_max = np.max(controller_temp)
    
    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
              framealpha=0.9, edgecolor='gray', frameon=True, fancybox=False, shadow=False)
    
    # 添加统计信息框
    stats_text = (f'Inhibit Count (CH0 min == 0):\n'
                 f'  Total = {total_events}\n'
                 f'  Mean = {count_mean:.1f}\n'
                 f'  Std = {count_std:.1f}\n'
                 f'  Min = {count_min:.0f}\n'
                 f'  Max = {count_max:.0f}\n'
                 f'\nController Temp:\n'
                 f'  Mean = {controller_mean:.2f} °C\n'
                 f'  Std = {controller_std:.2f} °C\n'
                 f'  Min = {controller_min:.2f} °C\n'
                 f'  Max = {controller_max:.2f} °C')
    
    ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                     edgecolor='gray', alpha=0.8, linewidth=0.8),
            fontsize=9,
            family='monospace')
    
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'\n图片已保存至: {save_path}')
    
    # 显示图片
    if show_plot:
        plt.show()
    else:
        plt.close()

# 示例使用
if __name__ == '__main__':
    # 获取项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # current_dir = python/data/temperature/countrate/inhibit
    parent_dir = os.path.dirname(current_dir)  # python/data/temperature/countrate
    grandparent_dir = os.path.dirname(parent_dir)  # python/data/temperature
    great_grandparent_dir = os.path.dirname(grandparent_dir)  # python/data
    project_root = os.path.dirname(os.path.dirname(great_grandparent_dir))  # 项目根目录
    
    # 设置数据路径
    bin_dir = os.path.join(project_root, 'data', 'bin')
    hdf5_dir = os.path.join(project_root, 'data', 'hdf5', 'raw_pulse')
    compressor_file_path = os.path.join(project_root, 'data', 'compressor', 'txt', 'EC1CP5.txt')
    
    print('=' * 70)
    print('bin文件Inhibit事件计数率（CH0 min == 0）和制冷机Controller温度联合绘制')
    print('=' * 70)
    
    try:
        # 绘制
        plot_countrate_and_temperature(
            bin_dir=bin_dir,
            compressor_file_path=compressor_file_path,
            hdf5_dir=hdf5_dir,  # 优先从HDF5文件读取
            ch0_idx=0,  # CH0通道索引
            time_bin_size='1h',  # 1小时一个bin
            start_date='2025-05-28',
            end_date='2025-06-10',
            show_plot=True
        )
        
    except Exception as e:
        print(f'错误: {e}')
        import traceback
        traceback.print_exc()
