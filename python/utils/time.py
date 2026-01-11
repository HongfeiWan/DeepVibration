#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从bin文件读取时间信息并计算时间跨度
"""
import os
import struct
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Tuple
import h5py


def read_bin_time_span(bin_file_path: str, 
                       epoch_offset: float = 2.082816000000000e+09) -> Tuple[datetime, datetime]:
    """
    从bin文件读取时间信息并计算时间跨度
    
    参数:
        bin_file_path: bin文件路径
        epoch_offset: 时间戳偏移量（默认值来自MATLAB代码）
    
    返回:
        (start_time, end_time): 起始时间和结束时间的datetime对象
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
        
        # 读取第一个事件的时间
        # 跳过 Event Header 的前几个字段
        fid.read(4)  # Hit_pat
        fid.read(4)  # V1729_tg_rec
        first_evt_endtime_bytes = fid.read(4)  # Evt_endtime
        if len(first_evt_endtime_bytes) < 4:
            raise ValueError('文件格式错误：无法读取第一个事件的时间')
        first_evt_endtime = struct.unpack('<I', first_evt_endtime_bytes)[0]
        
        # 跳过第一个事件的剩余数据
        fid.read(4)  # V1725_1_tgno
        fid.read(4)  # V1725_1_tag
        fid.read(16 * V1725_1_twd * 2)  # 16个通道的数据
        
        # 顺序读取所有事件头来找到最后一个事件
        # 每个事件的结构：
        # - Hit_pat (4 bytes)
        # - V1729_tg_rec (4 bytes)
        # - Evt_endtime (4 bytes)
        # - V1725_1_tgno (4 bytes)
        # - V1725_1_tag (4 bytes)
        # - 16个通道的数据，每个通道 V1725_1_twd * 2 字节
        event_header_size = 4 * 5  # 5个uint32字段
        event_data_size = 16 * V1725_1_twd * 2  # 16个通道的数据
        event_size = event_header_size + event_data_size
        
        last_evt_endtime = first_evt_endtime  # 默认使用第一个事件的时间
        max_events_to_check = 100000  # 最多检查100000个事件（避免无限循环）
        
        for j in range(1, max_events_to_check):
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
            
            # 更新最后一个事件的时间
            if evt_endtime > 0:
                last_evt_endtime = evt_endtime
        
        # 计算时间（按照MATLAB代码的逻辑）
        # time_data = evt_endtime / 1000.0 + pstt
        first_time_data = first_evt_endtime / 1000.0 + pstt
        last_time_data = last_evt_endtime / 1000.0 + pstt
        
        # 转换为datetime（按照MATLAB代码的逻辑）
        # eventtime = time_data - epoch_offset
        # time_transformed = epochStart + seconds(eventtime)
        epoch_start = datetime(1970, 1, 1)
        first_eventtime = first_time_data - epoch_offset
        last_eventtime = last_time_data - epoch_offset
        
        start_time = epoch_start + pd.Timedelta(seconds=first_eventtime)
        end_time = epoch_start + pd.Timedelta(seconds=last_eventtime)
        
        return start_time, end_time
        
    finally:
        if fid is not None:
            fid.close()

def read_hdf5_time_span(hdf5_file_path: str,
                        epoch_offset: float = 2.082816000000000e+09) -> Tuple[datetime, datetime]:
    """
    从HDF5文件读取时间信息并计算时间跨度（如果bin文件已处理为HDF5）
    
    参数:
        hdf5_file_path: HDF5文件路径
        epoch_offset: 时间戳偏移量（默认值来自MATLAB代码）
    
    返回:
        (start_time, end_time): 起始时间和结束时间的datetime对象
    """
    if not os.path.exists(hdf5_file_path):
        raise FileNotFoundError(f'文件不存在: {hdf5_file_path}')
    
    with h5py.File(hdf5_file_path, 'r') as f:
        if 'time_data' not in f:
            raise ValueError(f'HDF5文件中缺少 time_data 数据集')
        
        time_data = f['time_data'][:]
        
        if len(time_data) == 0:
            raise ValueError(f'HDF5文件中 time_data 为空')
        
        # 按照MATLAB代码的逻辑转换时间
        first_time_data = time_data[0]
        last_time_data = time_data[-1]
        
        # 转换为datetime
        epoch_start = datetime(1970, 1, 1)
        first_eventtime = first_time_data - epoch_offset
        last_eventtime = last_time_data - epoch_offset
        
        start_time = epoch_start + pd.Timedelta(seconds=first_eventtime)
        end_time = epoch_start + pd.Timedelta(seconds=last_eventtime)
        
        return start_time, end_time

def get_bin_file_time_span(bin_file_path: str,
                           hdf5_dir: Optional[str] = None,
                           epoch_offset: float = 2.082816000000000e+09) -> Tuple[datetime, datetime]:
    """
    获取bin文件的时间跨度（优先从HDF5文件读取，如果不存在则从bin文件读取）
    
    参数:
        bin_file_path: bin文件路径
        hdf5_dir: HDF5文件目录（如果提供，会先尝试从HDF5读取）
        epoch_offset: 时间戳偏移量（默认值来自MATLAB代码）
    
    返回:
        (start_time, end_time): 起始时间和结束时间的datetime对象
    """
    # 如果提供了HDF5目录，先尝试从HDF5文件读取
    if hdf5_dir is not None:
        bin_filename = os.path.basename(bin_file_path)
        hdf5_filename = f'{os.path.splitext(bin_filename)[0]}_processed.h5'
        hdf5_path = os.path.join(hdf5_dir, 'CH0-3', hdf5_filename)
        
        if os.path.exists(hdf5_path):
            try:
                return read_hdf5_time_span(hdf5_path, epoch_offset)
            except Exception as e:
                print(f'警告：从HDF5文件读取时间失败: {e}，尝试从bin文件读取')
    
    # 从bin文件读取
    return read_bin_time_span(bin_file_path, epoch_offset)

def process_bin_files_time_span(bin_dir: str,
                                 filename_input: str,
                                 run_start: int = 0,
                                 run_end: int = 999,
                                 hdf5_dir: Optional[str] = None,
                                 save_path: Optional[str] = None,
                                 epoch_offset: float = 2.082816000000000e+09) -> pd.DataFrame:
    """
    处理多个bin文件，获取每个文件的时间跨度
    
    参数:
        bin_dir: bin文件目录
        filename_input: 文件名前缀（不包含运行编号和扩展名）
        run_start: 起始运行编号
        run_end: 结束运行编号
        hdf5_dir: HDF5文件目录（可选，如果提供会优先从HDF5读取）
        save_path: 保存结果的路径（可选）
        epoch_offset: 时间戳偏移量（默认值来自MATLAB代码）
    
    返回:
        DataFrame: 包含运行编号、起始时间、结束时间的表格
    """
    run_numbers = []
    start_times = []
    end_times = []
    
    print('=' * 70)
    print('处理bin文件时间跨度')
    print('=' * 70)
    print(f'文件目录: {bin_dir}')
    print(f'文件名前缀: {filename_input}')
    print(f'运行编号范围: {run_start} - {run_end}')
    print('=' * 70)
    
    file_count = 0
    
    for i in range(run_start, run_end + 1):
        # 构造文件名
        bin_filename = f'{filename_input}FADC_RAW_Data_{i}.bin'
        bin_file_path = os.path.join(bin_dir, bin_filename)
        
        # 检查文件是否存在
        if os.path.exists(bin_file_path):
            file_count += 1
            try:
                start_time, end_time = get_bin_file_time_span(
                    bin_file_path, hdf5_dir, epoch_offset
                )
                
                run_numbers.append(i)
                start_times.append(start_time)
                end_times.append(end_time)
                
                print(f'处理完成: 运行编号 {i}, 起始时间: {start_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]}, '
                      f'结束时间: {end_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]}')
            except Exception as e:
                print(f'警告: 处理文件 {bin_filename} 时出错: {e}')
        else:
            print(f'警告: 文件 {bin_filename} 不存在')
    
    # 创建DataFrame
    if len(run_numbers) > 0:
        date_list = pd.DataFrame({
            'i': run_numbers,
            'starttime': start_times,
            'endtime': end_times
        })
        
        # 保存结果
        if save_path is not None:
            # 确保保存目录存在
            os.makedirs(save_path, exist_ok=True)
            
            # 构造输出文件名（使用文件名前缀的前8个字符，类似MATLAB代码）
            output_filename = os.path.join(save_path, f'{filename_input[:8]}_time.csv')
            date_list.to_csv(output_filename, index=False)
            print(f'\n时间数据已保存到: {output_filename}')
        
        print(f'\n处理完成: 共处理 {len(run_numbers)} 个文件')
        return date_list
    else:
        print('\n未找到任何文件')
        return pd.DataFrame(columns=['i', 'starttime', 'endtime'])

if __name__ == '__main__':
    # 示例用法
    import sys
    
    # 获取项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))  # python/utils
    project_root = os.path.dirname(os.path.dirname(current_dir))  # 从 python/utils 向上两级到项目根目录
    
    # 设置路径
    bin_dir = os.path.join(project_root, 'data', 'bin')
    hdf5_dir = os.path.join(project_root, 'data', 'hdf5', 'raw_pulse')
    save_path = project_root  # 保存到项目根目录
    
    # 文件名前缀（根据实际情况修改）
    filename_input = '20250520_CEvNS_DZL_sm_pre10000_tri10mV_SA6us0.8x50_SA12us0.8x50_TAout10us1.2x100_TAout10us0.5x3_RT50mHz_NaISA1us1.0x20_plasticsci1-10_bkg'
    
    # 运行编号范围
    run_start = 465
    run_end = 465
    
    # 处理文件
    date_list = process_bin_files_time_span(
        bin_dir=bin_dir,
        filename_input=filename_input,
        run_start=run_start,
        run_end=run_end,
        hdf5_dir=hdf5_dir,
        save_path=save_path
    )
    
    # 显示结果
    if len(date_list) > 0:
        print('\n时间跨度统计:')
        print(date_list)
