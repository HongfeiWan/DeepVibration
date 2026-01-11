#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
振动传感器数据预处理脚本
将 data/vibration/txt 文件夹下的 txt 文件转换为 HDF5 格式
保存到同层级的 hdf5 文件夹中
"""
import os
import sys
import glob
import time
import gc
import numpy as np
import pandas as pd
import h5py
from datetime import datetime
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

# 添加父目录到路径，以便导入utils模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.save import save_hdf5


def read_vibration_data(file_path: str) -> pd.DataFrame:
    """
    读取振动传感器数据文件
    参数:
        file_path: 数据文件路径
    返回:
        DataFrame，包含所有列数据
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'文件不存在: {file_path}')
    
    # 数据列名
    column_names = [
        'datetime',
        'Temperature',  # 温度
        'Velocity_x', 'Velocity_y', 'Velocity_z',  # x, y, z 振动速度
        'Acceleration_x', 'Acceleration_y', 'Acceleration_z',  # x, y, z 振动加速度*g
        'Displacement_x', 'Displacement_y', 'Displacement_z',  # x, y, z 振动位移μm
        'Frequency_x', 'Frequency_y', 'Frequency_z'  # x, y, z 振动频率
    ]
    
    # 逐行读取并解析
    data_rows = []
    skipped_lines = 0
    total_lines = 0
    
    # 检查文件大小
    file_size = os.path.getsize(file_path)
    file_size_MB = file_size / (1024 * 1024)
    is_large_file = file_size > 100 * 1024 * 1024  # 大于100MB
    
    # 注释掉文件大小输出，避免多进程输出混乱
    # print(f'  文件大小: {file_size_MB:.1f} MB')
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            
        # 注释掉进度输出，避免多进程输出混乱
        # if is_large_file and line_num % 100000 == 0:
        #     print(f'  已处理 {line_num:,} 行...', end='\r')
            
            line = line.strip()
            if not line:
                continue
            
            # 按逗号分割
            parts = line.split(',')
            
            # 只取前15列
            if len(parts) >= 2:  # 至少要有时间和温度
                row_data = parts[:15] if len(parts) >= 15 else parts + [''] * (15 - len(parts))
                
                try:
                    # 尝试解析时间
                    datetime_str = row_data[0].strip()
                    if datetime_str:
                        dt = pd.to_datetime(datetime_str, format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
                        if pd.notna(dt):
                            # 解析数值列
                            row_dict = {'datetime': dt}
                            for i, col_name in enumerate(column_names[1:], 1):
                                try:
                                    val_str = row_data[i].strip() if i < len(row_data) else ''
                                    val = float(val_str) if val_str else np.nan
                                    row_dict[col_name] = val
                                except (ValueError, IndexError):
                                    row_dict[col_name] = np.nan
                            
                            data_rows.append(row_dict)
                        else:
                            skipped_lines += 1
                    else:
                        skipped_lines += 1
                except Exception:
                    skipped_lines += 1
                    continue
            else:
                skipped_lines += 1
    
    # 注释掉详细输出，避免多进程输出混乱
    # if is_large_file:
    #     print()  # 换行
    # 
    # if skipped_lines > 0:
    #     print(f'  警告：跳过了 {skipped_lines:,} 行格式不正确的数据（共 {total_lines:,} 行）')
    
    if not data_rows:
        raise ValueError(f'文件中没有有效数据')
    
    # 转换为DataFrame
    df = pd.DataFrame(data_rows)
    
    # 按时间排序
    df = df.sort_values('datetime').reset_index(drop=True)
    
    return df


def parse_line(line: str, column_names: List[str]) -> Dict:
    """
    解析单行数据
    
    参数:
        line: 文本行
        column_names: 列名列表
    
    返回:
        解析后的数据字典，如果解析失败返回None
    """
    line = line.strip()
    if not line:
        return None
    
    # 按逗号分割
    parts = line.split(',')
    
    # 只取前15列
    if len(parts) < 2:  # 至少要有时间和温度
        return None
    
    row_data = parts[:15] if len(parts) >= 15 else parts + [''] * (15 - len(parts))
    
    try:
        # 尝试解析时间
        datetime_str = row_data[0].strip()
        if not datetime_str:
            return None
        
        dt = pd.to_datetime(datetime_str, format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
        if pd.isna(dt):
            return None
        
        # 解析数值列
        row_dict = {'datetime': dt}
        for i, col_name in enumerate(column_names[1:], 1):
            try:
                val_str = row_data[i].strip() if i < len(row_data) else ''
                val = float(val_str) if val_str else np.nan
                row_dict[col_name] = val
            except (ValueError, IndexError):
                row_dict[col_name] = np.nan
        
        return row_dict
    except Exception:
        return None


def parse_lines_batch(lines: List[str], column_names: List[str]) -> List[Dict]:
    """
    批量解析多行数据（优化版本，减少函数调用开销）
    
    参数:
        lines: 文本行列表
        column_names: 列名列表
    
    返回:
        解析后的数据字典列表
    """
    batch_rows = []
    for line in lines:
        row_dict = parse_line(line, column_names)
        if row_dict is not None:
            batch_rows.append(row_dict)
    return batch_rows


def txt_to_hdf5_streaming(txt_file_path: str, hdf5_dir: str, overwrite: bool = False, 
                          batch_size: int = 50000, global_sort: bool = False) -> bool:
    """
    将单个txt文件转换为HDF5格式（流式处理，避免内存爆炸）
    采用批量读取、批量解析、批量写入的方式，大幅提升速度
    
    参数:
        txt_file_path: txt文件路径
        hdf5_dir: HDF5文件保存目录
        overwrite: 如果HDF5文件已存在，是否覆盖
        batch_size: 每批处理的行数，默认50000行（增大批次可提升速度）
        global_sort: 是否进行全局排序，False（默认）不排序，只按批次排序，避免内存爆炸
    
    返回:
        成功返回True，失败返回False
    """
    try:
        # 获取文件名（不含扩展名）
        base_name = os.path.splitext(os.path.basename(txt_file_path))[0]
        hdf5_filename = f'{base_name}.h5'
        hdf5_path = os.path.join(hdf5_dir, hdf5_filename)
        
        # 检查文件是否已存在
        if os.path.exists(hdf5_path) and not overwrite:
            sys.stderr.write(f'[{base_name}] 文件已存在，跳过处理\n')
            sys.stderr.flush()
            return True
        
        sys.stderr.write(f'[{base_name}] 开始处理文件\n')
        sys.stderr.flush()
        
        # 数据列名
        column_names = [
            'datetime',
            'Temperature',  # 温度
            'Velocity_x', 'Velocity_y', 'Velocity_z',  # x, y, z 振动速度
            'Acceleration_x', 'Acceleration_y', 'Acceleration_z',  # x, y, z 振动加速度*g
            'Displacement_x', 'Displacement_y', 'Displacement_z',  # x, y, z 振动位移μm
            'Frequency_x', 'Frequency_y', 'Frequency_z'  # x, y, z 振动频率
        ]
        
        # 使用可扩展的HDF5数据集，不需要预先知道大小
        epoch = pd.Timestamp('1970-01-01')
        current_size = 0  # 当前数据集大小
        chunk_size = batch_size  # HDF5 chunk大小
        
        # 预分配扩展大小，减少resize次数（每次扩展更大的块）
        resize_growth_factor = 2  # 每次扩展时增长2倍
        
        # 初始化统计变量（在with块外定义，确保在函数结束时可用）
        allocated_size = 0  # 预分配的大小
        batch_count = 0  # 批次计数器
        total_lines_read = 0  # 总读取行数
        total_valid_rows = 0  # 总有效行数
        
        # 打开HDF5文件，创建可扩展的数据集
        with h5py.File(hdf5_path, 'w') as h5f:
            # 创建可扩展的数据集（使用maxshape=(None,)表示可以无限扩展）
            datasets = {
                'time': h5f.create_dataset('time', shape=(0,), maxshape=(None,), 
                                          dtype=np.float64, chunks=(chunk_size,)),
                'datetime_str': h5f.create_dataset('datetime_str', shape=(0,), maxshape=(None,), 
                                                  dtype='S26', chunks=(chunk_size,)),
                'Temperature': h5f.create_dataset('Temperature', shape=(0,), maxshape=(None,), 
                                                  dtype=np.float32, chunks=(chunk_size,)),
                'Velocity_x': h5f.create_dataset('Velocity_x', shape=(0,), maxshape=(None,), 
                                                 dtype=np.float32, chunks=(chunk_size,)),
                'Velocity_y': h5f.create_dataset('Velocity_y', shape=(0,), maxshape=(None,), 
                                                 dtype=np.float32, chunks=(chunk_size,)),
                'Velocity_z': h5f.create_dataset('Velocity_z', shape=(0,), maxshape=(None,), 
                                                 dtype=np.float32, chunks=(chunk_size,)),
                'Acceleration_x': h5f.create_dataset('Acceleration_x', shape=(0,), maxshape=(None,), 
                                                     dtype=np.float32, chunks=(chunk_size,)),
                'Acceleration_y': h5f.create_dataset('Acceleration_y', shape=(0,), maxshape=(None,), 
                                                     dtype=np.float32, chunks=(chunk_size,)),
                'Acceleration_z': h5f.create_dataset('Acceleration_z', shape=(0,), maxshape=(None,), 
                                                     dtype=np.float32, chunks=(chunk_size,)),
                'Displacement_x': h5f.create_dataset('Displacement_x', shape=(0,), maxshape=(None,), 
                                                     dtype=np.float32, chunks=(chunk_size,)),
                'Displacement_y': h5f.create_dataset('Displacement_y', shape=(0,), maxshape=(None,), 
                                                     dtype=np.float32, chunks=(chunk_size,)),
                'Displacement_z': h5f.create_dataset('Displacement_z', shape=(0,), maxshape=(None,), 
                                                     dtype=np.float32, chunks=(chunk_size,)),
                'Frequency_x': h5f.create_dataset('Frequency_x', shape=(0,), maxshape=(None,), 
                                                 dtype=np.float32, chunks=(chunk_size,)),
                'Frequency_y': h5f.create_dataset('Frequency_y', shape=(0,), maxshape=(None,), 
                                                 dtype=np.float32, chunks=(chunk_size,)),
                'Frequency_z': h5f.create_dataset('Frequency_z', shape=(0,), maxshape=(None,), 
                                                  dtype=np.float32, chunks=(chunk_size,))
            }
            
            # 批量读取文件，边读边写
            
            # 先统计文件总行数（用于显示进度）
            # 对于大文件，统计可能很慢，所以先尝试快速估算
            try:
                file_size = os.path.getsize(txt_file_path)
                file_size_mb = file_size / (1024 * 1024)
                sys.stderr.write(f'[{base_name}] 文件大小: {file_size_mb:.2f} MB\n')
                sys.stderr.flush()
                
                # 对于大文件（>100MB），跳过详细统计，使用估算
                if file_size > 100 * 1024 * 1024:
                    # 估算：假设平均每行100字节（实际可能更少）
                    estimated_lines = int(file_size / 100)
                    total_file_lines = None  # 不统计，使用估算值显示进度
                    sys.stderr.write(f'[{base_name}] 文件较大，跳过详细统计，估算约 {estimated_lines:,} 行\n')
                    sys.stderr.flush()
                else:
                    with open(txt_file_path, 'r', encoding='utf-8') as temp_f:
                        total_file_lines = sum(1 for _ in temp_f)
                    sys.stderr.write(f'[{base_name}] 文件总行数: {total_file_lines:,} 行\n')
                    sys.stderr.flush()
            except Exception as e:
                total_file_lines = None
                sys.stderr.write(f'[{base_name}] 无法统计文件总行数，将实时显示进度 (错误: {str(e)[:50]})\n')
                sys.stderr.flush()
            
            with open(txt_file_path, 'r', encoding='utf-8') as f:
                lines_buffer = []  # 行缓冲区
                
                # 使用迭代器方式读取，确保读完所有内容
                for line in f:
                    total_lines_read += 1
                    lines_buffer.append(line)
                    
                    # 每读取10万行打印一次进度（用于大文件）
                    if total_lines_read % 100000 == 0:
                        if total_file_lines:
                            progress_pct = (total_lines_read / total_file_lines * 100) if total_file_lines > 0 else 0
                            sys.stderr.write(f'[{base_name}] 读取进度: {total_lines_read:,}/{total_file_lines:,} 行 ({progress_pct:.1f}%)\n')
                        else:
                            sys.stderr.write(f'[{base_name}] 读取进度: {total_lines_read:,} 行\n')
                        sys.stderr.flush()
                    
                    # 当缓冲区达到批次大小时，处理这一批
                    if len(lines_buffer) >= batch_size:
                        batch_count += 1
                        try:
                            # 批量解析所有行
                            batch_rows = parse_lines_batch(lines_buffer, column_names)
                            # 清空缓冲区，准备读取下一批
                            lines_buffer = []
                            
                            # 如果解析后没有有效数据，继续读取下一批
                            if not batch_rows:
                                if total_file_lines:
                                    sys.stderr.write(f'[{base_name}] 批次 {batch_count}: 读取 {total_lines_read:,}/{total_file_lines:,} 行 ({(total_lines_read/total_file_lines*100):.1f}%), 有效数据: 0 行，跳过\n')
                                else:
                                    sys.stderr.write(f'[{base_name}] 批次 {batch_count}: 已读取 {total_lines_read:,} 行, 有效数据: 0 行，跳过\n')
                                sys.stderr.flush()
                                continue
                            
                            # 处理这一批数据
                            # 转换为DataFrame并排序
                            df_batch = pd.DataFrame(batch_rows)
                            df_batch = df_batch.sort_values('datetime').reset_index(drop=True)
                            
                            # 准备数据
                            datetime_series = pd.to_datetime(df_batch['datetime'])
                            time_data = (datetime_series - epoch).dt.total_seconds().values.astype(np.float64)
                            datetime_str_array = datetime_series.dt.strftime('%Y-%m-%d %H:%M:%S.%f').values.astype('S26')
                            
                            # 计算新的大小（预分配更大的空间，减少resize次数）
                            batch_size_actual = len(df_batch)
                            new_size = current_size + batch_size_actual
                            # 预分配策略：如果当前大小较小，按倍数增长；否则按固定增量增长
                            if current_size == 0:
                                allocated_size = new_size
                            elif new_size > allocated_size:
                                # 每次扩展时，预分配更多空间以减少resize次数
                                allocated_size = max(new_size, int(current_size * resize_growth_factor))
                            
                            # 批量扩展所有数据集（一次性扩展，减少开销）
                            if allocated_size > current_size:
                                for dset in datasets.values():
                                    dset.resize((allocated_size,))
                            
                            # 批量写入HDF5
                            datasets['time'][current_size:new_size] = time_data
                            datasets['datetime_str'][current_size:new_size] = datetime_str_array
                            datasets['Temperature'][current_size:new_size] = df_batch['Temperature'].values.astype(np.float32)
                            datasets['Velocity_x'][current_size:new_size] = df_batch['Velocity_x'].values.astype(np.float32)
                            datasets['Velocity_y'][current_size:new_size] = df_batch['Velocity_y'].values.astype(np.float32)
                            datasets['Velocity_z'][current_size:new_size] = df_batch['Velocity_z'].values.astype(np.float32)
                            datasets['Acceleration_x'][current_size:new_size] = df_batch['Acceleration_x'].values.astype(np.float32)
                            datasets['Acceleration_y'][current_size:new_size] = df_batch['Acceleration_y'].values.astype(np.float32)
                            datasets['Acceleration_z'][current_size:new_size] = df_batch['Acceleration_z'].values.astype(np.float32)
                            datasets['Displacement_x'][current_size:new_size] = df_batch['Displacement_x'].values.astype(np.float32)
                            datasets['Displacement_y'][current_size:new_size] = df_batch['Displacement_y'].values.astype(np.float32)
                            datasets['Displacement_z'][current_size:new_size] = df_batch['Displacement_z'].values.astype(np.float32)
                            datasets['Frequency_x'][current_size:new_size] = df_batch['Frequency_x'].values.astype(np.float32)
                            datasets['Frequency_y'][current_size:new_size] = df_batch['Frequency_y'].values.astype(np.float32)
                            datasets['Frequency_z'][current_size:new_size] = df_batch['Frequency_z'].values.astype(np.float32)
                            
                            # 更新当前大小
                            current_size = new_size
                            total_valid_rows += batch_size_actual
                            
                            # 打印进度信息
                            if total_file_lines:
                                progress_pct = (total_lines_read / total_file_lines * 100) if total_file_lines > 0 else 0
                                sys.stderr.write(f'[{base_name}] 批次 {batch_count}: 读取 {total_lines_read:,}/{total_file_lines:,} 行 ({progress_pct:.1f}%), 有效数据: {batch_size_actual:,} 行, 累计: {total_valid_rows:,} 行\n')
                            else:
                                sys.stderr.write(f'[{base_name}] 批次 {batch_count}: 已读取 {total_lines_read:,} 行, 有效数据: {batch_size_actual:,} 行, 累计: {total_valid_rows:,} 行\n')
                            sys.stderr.flush()
                            
                            # 清空批次数据，释放内存
                            del df_batch, datetime_series, time_data, datetime_str_array, batch_rows
                        except Exception as batch_error:
                            # 如果处理这一批时出错，记录错误但继续处理下一批
                            sys.stderr.write(f'[{base_name}] 批次 {batch_count}: 处理出错 - {str(batch_error)[:100]}\n')
                            sys.stderr.flush()
                            # 清空缓冲区，避免重复处理
                            lines_buffer = []
                            # 继续读取下一批，不中断整个处理过程
                            continue
                
                # 文件读取循环结束
                sys.stderr.write(f'[{base_name}] 文件读取循环结束，共读取 {total_lines_read:,} 行\n')
                sys.stderr.flush()
                
                # 处理最后一批数据（文件末尾剩余的数据）
                if lines_buffer:
                    batch_count += 1
                    if batch_count == 1:
                        # 如果这是第一批（也是最后一批），说明文件较小，所有数据都在这一批
                        sys.stderr.write(f'[{base_name}] 文件较小，所有数据将在单批处理: {len(lines_buffer):,} 行\n')
                    else:
                        sys.stderr.write(f'[{base_name}] 处理最后一批数据: {len(lines_buffer):,} 行\n')
                    sys.stderr.flush()
                    # 批量解析所有行
                    batch_rows = parse_lines_batch(lines_buffer, column_names)
                    del lines_buffer
                    
                    if batch_rows:
                        # 转换为DataFrame并排序
                        df_batch = pd.DataFrame(batch_rows)
                        df_batch = df_batch.sort_values('datetime').reset_index(drop=True)
                        
                        # 准备数据
                        datetime_series = pd.to_datetime(df_batch['datetime'])
                        time_data = (datetime_series - epoch).dt.total_seconds().values.astype(np.float64)
                        datetime_str_array = datetime_series.dt.strftime('%Y-%m-%d %H:%M:%S.%f').values.astype('S26')
                        
                        # 计算新的大小
                        batch_size_actual = len(df_batch)
                        new_size = current_size + batch_size_actual
                        if new_size > allocated_size:
                            allocated_size = max(new_size, int(current_size * resize_growth_factor)) if current_size > 0 else new_size
                        
                        # 批量扩展所有数据集
                        if allocated_size > current_size:
                            for dset in datasets.values():
                                dset.resize((allocated_size,))
                        
                        # 批量写入HDF5
                        datasets['time'][current_size:new_size] = time_data
                        datasets['datetime_str'][current_size:new_size] = datetime_str_array
                        datasets['Temperature'][current_size:new_size] = df_batch['Temperature'].values.astype(np.float32)
                        datasets['Velocity_x'][current_size:new_size] = df_batch['Velocity_x'].values.astype(np.float32)
                        datasets['Velocity_y'][current_size:new_size] = df_batch['Velocity_y'].values.astype(np.float32)
                        datasets['Velocity_z'][current_size:new_size] = df_batch['Velocity_z'].values.astype(np.float32)
                        datasets['Acceleration_x'][current_size:new_size] = df_batch['Acceleration_x'].values.astype(np.float32)
                        datasets['Acceleration_y'][current_size:new_size] = df_batch['Acceleration_y'].values.astype(np.float32)
                        datasets['Acceleration_z'][current_size:new_size] = df_batch['Acceleration_z'].values.astype(np.float32)
                        datasets['Displacement_x'][current_size:new_size] = df_batch['Displacement_x'].values.astype(np.float32)
                        datasets['Displacement_y'][current_size:new_size] = df_batch['Displacement_y'].values.astype(np.float32)
                        datasets['Displacement_z'][current_size:new_size] = df_batch['Displacement_z'].values.astype(np.float32)
                        datasets['Frequency_x'][current_size:new_size] = df_batch['Frequency_x'].values.astype(np.float32)
                        datasets['Frequency_y'][current_size:new_size] = df_batch['Frequency_y'].values.astype(np.float32)
                        datasets['Frequency_z'][current_size:new_size] = df_batch['Frequency_z'].values.astype(np.float32)
                        
                        # 更新当前大小
                        current_size = new_size
                        total_valid_rows += batch_size_actual
                        
                        if batch_count == 1:
                            # 单批处理的情况
                            sys.stderr.write(f'[{base_name}] 单批处理完成: 有效数据: {batch_size_actual:,} 行\n')
                        else:
                            sys.stderr.write(f'[{base_name}] 最后一批: 有效数据: {batch_size_actual:,} 行, 累计: {total_valid_rows:,} 行\n')
                        sys.stderr.flush()
                        
                        del df_batch, datetime_series, time_data, datetime_str_array, batch_rows
                    else:
                        sys.stderr.write(f'[{base_name}] 最后一批: 无有效数据\n')
                        sys.stderr.flush()
                
                # 打印总结信息
                invalid_lines = total_lines_read - total_valid_rows
                if total_file_lines:
                    sys.stderr.write(f'[{base_name}] 文件读取完成: 总行数 {total_file_lines:,}, 已读取 {total_lines_read:,} 行, 有效数据 {total_valid_rows:,} 行, 无效/跳过 {invalid_lines:,} 行, 共处理 {batch_count} 个批次\n')
                else:
                    sys.stderr.write(f'[{base_name}] 文件读取完成: 已读取 {total_lines_read:,} 行, 有效数据 {total_valid_rows:,} 行, 无效/跳过 {invalid_lines:,} 行, 共处理 {batch_count} 个批次\n')
                sys.stderr.flush()
                
                # 如果预分配的空间大于实际使用的空间，裁剪数据集
                if allocated_size > current_size:
                    for dset in datasets.values():
                        dset.resize((current_size,))
        
        # 如果需要全局排序（不推荐，会占用大量内存）
        if global_sort:
            try:
                with h5py.File(hdf5_path, 'r+') as h5f:
                    time_data = h5f['time'][:]
                    sorted_indices = np.argsort(time_data)
                    
                    # 如果排序后的索引与原始索引相同，说明已经是有序的，跳过重排
                    if not np.array_equal(sorted_indices, np.arange(len(time_data))):
                        # 需要重排所有数据集
                        datasets = ['time', 'datetime_str', 'Temperature', 'Velocity_x', 'Velocity_y', 'Velocity_z',
                                   'Acceleration_x', 'Acceleration_y', 'Acceleration_z',
                                   'Displacement_x', 'Displacement_y', 'Displacement_z',
                                   'Frequency_x', 'Frequency_y', 'Frequency_z']
                        
                        # 读取所有数据，排序后写回（这会占用内存）
                        data_dict = {}
                        for dset_name in datasets:
                            data_dict[dset_name] = h5f[dset_name][:]
                        
                        # 按排序索引重排
                        for dset_name in datasets:
                            h5f[dset_name][:] = data_dict[dset_name][sorted_indices]
                        
                        del data_dict, time_data, sorted_indices
                        gc.collect()
            except MemoryError:
                # 如果内存不足，跳过全局排序
                pass
        
        # 检查文件是否成功保存
        if os.path.exists(hdf5_path):
            file_size = os.path.getsize(hdf5_path) / (1024 * 1024)  # MB
            sys.stderr.write(f'[{base_name}] 文件处理完成: {base_name}.h5 已保存 ({file_size:.2f} MB), 共 {total_valid_rows:,} 行有效数据\n')
            sys.stderr.flush()
            return True
        else:
            sys.stderr.write(f'[{base_name}] 错误: HDF5文件未成功创建\n')
            sys.stderr.flush()
            return False
            
    except Exception as e:
        # 在多进程环境中，异常信息会被主进程捕获
        raise Exception(f'处理文件 {os.path.basename(txt_file_path)} 时出错: {e}')


def txt_to_hdf5(txt_file_path: str, hdf5_dir: str, overwrite: bool = False) -> bool:
    """
    将单个txt文件转换为HDF5格式（使用流式处理，边读边写，避免内存爆炸）
    
    参数:
        txt_file_path: txt文件路径
        hdf5_dir: HDF5文件保存目录
        overwrite: 如果HDF5文件已存在，是否覆盖
    
    返回:
        成功返回True，失败返回False
    """
    return txt_to_hdf5_streaming(txt_file_path, hdf5_dir, overwrite, batch_size=50000, global_sort=False)


def preprocess_all_txt_files(txt_dir: str, hdf5_dir: str, overwrite: bool = False, max_workers: int = None):
    """
    预处理所有txt文件，转换为HDF5格式（使用多进程并行处理）
    
    参数:
        txt_dir: txt文件所在目录
        hdf5_dir: HDF5文件保存目录
        overwrite: 如果HDF5文件已存在，是否覆盖
        max_workers: 最大并行进程数，如果为None则使用所有CPU核心
    """
    if not os.path.exists(txt_dir):
        raise FileNotFoundError(f'目录不存在: {txt_dir}')
    
    # 创建hdf5目录（如果不存在）
    os.makedirs(hdf5_dir, exist_ok=True)
    
    # 查找所有txt文件
    pattern = os.path.join(txt_dir, 'detector_*.txt')
    txt_files = sorted(glob.glob(pattern))
    
    if not txt_files:
        print(f'未找到任何txt文件: {pattern}')
        return
    
    print(f'找到 {len(txt_files)} 个txt文件')
    
    # 获取CPU核心数
    if max_workers is None:
        max_workers = os.cpu_count()
    print(f'使用 {max_workers} 个CPU核心进行并行处理')
    print('=' * 70)
    
    # 准备任务列表
    tasks = [(txt_file, hdf5_dir, overwrite) for txt_file in txt_files]
    
    # 使用多进程并行处理
    start_time = time.time()
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_file = {executor.submit(txt_to_hdf5, txt_file, hdf5_dir, overwrite): txt_file 
                          for txt_file, _, _ in tasks}
        
        # 收集结果
        for i, future in enumerate(as_completed(future_to_file), 1):
            txt_file = future_to_file[future]
            base_name = os.path.splitext(os.path.basename(txt_file))[0]
            
            try:
                result = future.result()
                if result:
                    success_count += 1
                    print(f'[{i}/{len(txt_files)}] ✓ 成功: {base_name}.txt')
                else:
                    # 检查是否因为已存在而跳过
                    hdf5_path = os.path.join(hdf5_dir, f'{base_name}.h5')
                    if os.path.exists(hdf5_path) and not overwrite:
                        skip_count += 1
                        print(f'[{i}/{len(txt_files)}] ⊘ 跳过（已存在）: {base_name}.txt')
                    else:
                        fail_count += 1
                        print(f'[{i}/{len(txt_files)}] ✗ 失败: {base_name}.txt')
            except Exception as e:
                fail_count += 1
                print(f'[{i}/{len(txt_files)}] ✗ 错误: {base_name}.txt - {e}')
    
    elapsed_time = time.time() - start_time
    
    # 显示统计信息
    print('\n' + '=' * 70)
    print('处理完成统计:')
    print(f'  成功: {success_count} 个文件')
    print(f'  跳过: {skip_count} 个文件（已存在）')
    print(f'  失败: {fail_count} 个文件')
    print(f'  总计: {len(txt_files)} 个文件')
    print(f'  总耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)')
    print('=' * 70)


# 示例使用
if __name__ == '__main__':
    # 获取项目根目录
    # 从 python/data/sensor/vibration/preprocess.py 向上4层到达项目根目录
    # vibration -> sensor -> data -> python -> DeepVibration (项目根目录)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
    
    # 设置输入和输出目录
    txt_dir = os.path.join(project_root, 'data', 'vibration', 'txt')
    hdf5_dir = os.path.join(project_root, 'data', 'vibration', 'hdf5')
    
    print('=' * 70)
    print('振动传感器数据预处理 - TXT转HDF5')
    print('=' * 70)
    print(f'输入目录: {txt_dir}')
    print(f'输出目录: {hdf5_dir}')
    print('=' * 70)
    
    # 预处理所有文件
    # overwrite=False: 如果HDF5文件已存在，则跳过
    # overwrite=True: 如果HDF5文件已存在，则覆盖
    preprocess_all_txt_files(txt_dir, hdf5_dir, overwrite=False)

