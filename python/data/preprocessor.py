#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
二进制文件预处理脚本
将V1725数据采集卡的.bin文件转换为.mat格式的原始脉冲数据
"""
import os
import sys
import struct
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py


# 添加父目录到路径，以便导入utils模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from utils.save import save_hdf5
from utils.fit import _compute_fast_fit_params
from utils.frequency import _compute_fast_highfreq_energy_ratio

# 定义文件路径和文件名（相对于项目根目录）
# 从 python/data/ 目录到项目根目录的 data/ 目录
project_root = os.path.dirname(os.path.dirname(current_dir))
read_path = os.path.join(project_root, 'data', 'bin')

amp_save_path = os.path.join(project_root, 'data', 'hdf5', 'raw_pulse', 'CH0-3')
NAI_save_path = os.path.join(project_root, 'data', 'hdf5', 'raw_pulse', 'CH4')
trigger_save_path = os.path.join(project_root, 'data', 'hdf5', 'raw_pulse', 'CH5')

ch0parameters_save_path = os.path.join(project_root, 'data', 'hdf5', 'raw_pulse', 'CH0_parameters')
ch1parameters_save_path = os.path.join(project_root, 'data', 'hdf5', 'raw_pulse', 'CH1_parameters')
ch2parameters_save_path = os.path.join(project_root, 'data', 'hdf5', 'raw_pulse', 'CH2_parameters')
ch3parameters_save_path = os.path.join(project_root, 'data', 'hdf5', 'raw_pulse', 'CH3_parameters')
ch4parameters_save_path = os.path.join(project_root, 'data', 'hdf5', 'raw_pulse', 'CH4_parameters')
ch5parameters_save_path = os.path.join(project_root, 'data', 'hdf5', 'raw_pulse', 'CH5_parameters')

filename_input = '20250520_CEvNS_DZL_sm_pre10000_tri10mV_SA6us0.8x50_SA12us0.8x50_TAout10us1.2x100_TAout10us0.5x3_RT50mHz_NaISA1us1.0x20_plasticsci1-10_bkg'

RUN_Start_NUMBER = 0    # 起始运行编号
RUN_End_NUMBER = 511    # 结束运行编号

# 定义通道数和事件数
AMP_CHANNEL_LIST = [0, 1, 2, 3]     # 指定要保存的通道索引（0=CH0, 1=CH1, 2=CH2, 3=CH3）
TRIGGER_CHANNEL_LIST = [5]          # 指定要保存的随机触发通道索引（5=CH5）
NAI_CHANNEL_LIST = [4]              # 指定要保存的NAI通道索引（4=CH4）
EVENT_NUMBER = 10000    # 每个bin文件中的理论上事件数
MAX_WINDOWS = 30000     # 时间窗 120μs （30000个时间点 x 4ns）

def bin2rawpulse(run_filename, channel_list, event_number, save_path, ch0parameters_save_dir=None):
    """
    处理bin文件中对应channel_list通道的原始波形并且保存。
    若 channel_list 包含 CH0(0) 且 ch0parameters_save_dir 给定，则顺带计算并保存 CH0 的特征参数。

    参数:
        run_filename: 输入文件路径
        channel_list: 要保存的通道索引列表（例如 [0, 1, 2, 3] 表示CH0-CH3）
        event_number: 预期事件数
        save_path: 保存路径
        ch0parameters_save_dir: CH0max 输出目录；为 None 则不写 CH0max
    """
    print('=' * 45)
    print(f'Opening {run_filename}')

    # 打开文件
    fid = None
    try:
        fid = open(run_filename, 'rb')
    except (IOError, OSError, FileNotFoundError) as e:
        raise OSError(f'Failed to open file: {run_filename}') from e
    
    try:
        # 读取 Run Header
        # 使用小端字节序（'<'），因为MATLAB默认使用小端
        pstt = struct.unpack('<d', fid.read(8))[0]  # Program Start Time (double, 8 bytes)
        print(f'Program Start Time: {pstt:.6f} s.')
        
        # 读取 V1725-1 Channel DAC (读取16个通道的DAC值)
        # numpy.frombuffer默认使用系统字节序，通常是小端，与MATLAB一致
        V1725_1_DAC = np.frombuffer(fid.read(16 * 4), dtype='<u4')  # 16个uint32，小端
        print('V1725-1 Channel DAC:', ' '.join(str(x) for x in V1725_1_DAC))
        
        # 读取其他 Run Header 信息
        V1725_1_twd = struct.unpack('<I', fid.read(4))[0]  # Time Window (uint32, 4 bytes)
        print(f'V1725-1 Time Window: {V1725_1_twd}')
        
        V1725_1_pretg = struct.unpack('<I', fid.read(4))[0]  # Pre Trigger (uint32, 4 bytes)
        print(f'V1725-1 Pre Trigger: {V1725_1_pretg}')
        
        V1725_1_opch = struct.unpack('<I', fid.read(4))[0]  # Opened Channel (uint32, 4 bytes)
        print(f'V1725-1 Opened Channel: {V1725_1_opch}')
        
        # 读取 Run Start Time
        rstt = struct.unpack('<d', fid.read(8))[0]  # Run Start Time (double, 8 bytes)
        print(f'Run Start Time: {rstt:.6f} s.')
        
        # 初始化事件数据结构
        idevt = np.zeros(event_number, dtype=np.float64)
        trig = np.zeros(event_number, dtype=np.float64)
        time_array = np.zeros(event_number, dtype=np.float64)
        deadtime = np.zeros(event_number, dtype=np.float64)
        
        # 预分配事件数据数组
        hit_pat_array = np.zeros(event_number, dtype=np.uint32)
        v1729_tg_rec_array = np.zeros(event_number, dtype=np.uint32)
        evt_endtime_array = np.zeros(event_number, dtype=np.uint32)
        v1725_1_tgno_array = np.zeros(event_number, dtype=np.uint32)
        v1725_1_tag_array = np.zeros(event_number, dtype=np.uint32)
        
        # 初始化数据结构 - 保存原始脉冲数据
        # 只保存指定通道的数据
        num_channels_to_save = len(channel_list)
        channel_data = np.zeros((V1725_1_twd, num_channels_to_save, event_number), dtype=np.uint16)
        time_data = np.zeros(event_number, dtype=np.float64)
        
        # 验证通道索引有效性
        max_channel_index = max(channel_list) if channel_list else -1
        if max_channel_index >= 16:
            raise ValueError(f'通道索引超出范围！最大支持通道索引为15，但指定了通道 {max_channel_index}')
        if min(channel_list) < 0:
            raise ValueError(f'通道索引不能为负数！')
        
        # 首先读取所有事件的数据
        actual_events = 0
        for j in range(event_number):
            # 检查是否到达文件末尾
            current_pos = fid.tell()
            fid.seek(0, 2)  # 移动到文件末尾
            file_size = fid.tell()
            fid.seek(current_pos)  # 恢复位置
            
            if current_pos >= file_size:
                print(f'警告：文件在事件 {j+1} 处结束，实际读取了 {actual_events} 个事件')
                break
            
            # 读取 Event Header
            hit_pat_bytes = fid.read(4)  # Hit_pat (uint32, 4 bytes)
            if len(hit_pat_bytes) < 4:
                print(f'警告：文件在事件 {j+1} 处结束，实际读取了 {actual_events} 个事件')
                break
            hit_pat_data = struct.unpack('<I', hit_pat_bytes)[0]
            hit_pat_array[j] = hit_pat_data
            
            v1729_tg_rec_bytes = fid.read(4)  # V1729_tg_rec (uint32, 4 bytes)
            if len(v1729_tg_rec_bytes) < 4:
                print(f'警告：文件在事件 {j+1} 处结束，实际读取了 {actual_events} 个事件')
                break
            v1729_tg_rec_data = struct.unpack('<I', v1729_tg_rec_bytes)[0]
            v1729_tg_rec_array[j] = v1729_tg_rec_data
            
            evt_endtime_bytes = fid.read(4)  # Evt_endtime (uint32, 4 bytes)
            if len(evt_endtime_bytes) < 4:
                print(f'警告：文件在事件 {j+1} 处结束，实际读取了 {actual_events} 个事件')
                break
            evt_endtime_data = struct.unpack('<I', evt_endtime_bytes)[0]
            evt_endtime_array[j] = evt_endtime_data
            
            v1725_1_tgno_bytes = fid.read(4)  # V1725_1_tgno (uint32, 4 bytes)
            if len(v1725_1_tgno_bytes) < 4:
                print(f'警告：文件在事件 {j+1} 处结束，实际读取了 {actual_events} 个事件')
                break
            v1725_1_tgno_data = struct.unpack('<I', v1725_1_tgno_bytes)[0]
            v1725_1_tgno_array[j] = v1725_1_tgno_data
            
            v1725_1_tag_bytes = fid.read(4)  # V1725_1_tag (uint32, 4 bytes)
            if len(v1725_1_tag_bytes) < 4:
                print(f'警告：文件在事件 {j+1} 处结束，实际读取了 {actual_events} 个事件')
                break
            v1725_1_tag_data = struct.unpack('<I', v1725_1_tag_bytes)[0]
            v1725_1_tag_array[j] = v1725_1_tag_data
            
            # 读取每个通道的数据 (只处理前4个通道，但需要跳过其他通道)
            for k in range(16):  # 仍然需要读取所有16个通道的数据
                channel_data_bytes = fid.read(V1725_1_twd * 2)  # uint16 = 2 bytes
                if len(channel_data_bytes) < V1725_1_twd * 2:
                    print(f'警告：文件在事件 {j+1} 通道 {k+1} 处结束，实际读取了 {actual_events} 个事件')
                    break
                channel_data_temp = np.frombuffer(channel_data_bytes, dtype='<u2')  # uint16，小端
                
                # 只保存指定通道的数据
                if k in channel_list:
                    # 找到k在channel_list中的位置索引
                    channel_idx = channel_list.index(k)
                    channel_data[:, channel_idx, j] = channel_data_temp
            
            # 计算时间信息（将在后续处理中加上pstt）
            time_with_pstt = evt_endtime_data / 1000.0  # 先转换为秒，后续会加上pstt
            time_data[j] = time_with_pstt
            
            actual_events += 1
        
        # 更新实际事件数并调整数组大小
        if actual_events < event_number:
            event_number = actual_events
            print(f'实际读取的事件数：{event_number}')
            
            # 调整数组大小以匹配实际事件数
            hit_pat_array = hit_pat_array[:actual_events]
            v1729_tg_rec_array = v1729_tg_rec_array[:actual_events]
            evt_endtime_array = evt_endtime_array[:actual_events]
            v1725_1_tgno_array = v1725_1_tgno_array[:actual_events]
            v1725_1_tag_array = v1725_1_tag_array[:actual_events]
            channel_data = channel_data[:, :, :actual_events]
            time_data = time_data[:actual_events]
            
            # 重新初始化数组以匹配实际事件数
            idevt = idevt[:event_number]
            trig = trig[:event_number]
            time_array = time_array[:event_number]
            deadtime = deadtime[:event_number]
        
        # 关闭文件
        fid.close()
        
        # 处理所有事件（串行处理）
        for j in range(event_number):
            # 获取预读取的事件头信息
            Hit_pat = hit_pat_array[j]
            V1729_tg_rec = v1729_tg_rec_array[j]
            Evt_endtime = evt_endtime_array[j]
            V1725_1_tgno = v1725_1_tgno_array[j]
            V1725_1_tag = v1725_1_tag_array[j]
            
            # 计算时间
            TTTV1725 = V1725_1_tag & 0x7FFFFFFF
            TimeV1725 = 10.0 * float(TTTV1725)
            
            # 存储事件信息
            idevt[j] = j + 1  # 简化ID计算
            trig[j] = V1725_1_tgno
            time_array[j] = Evt_endtime / 1000.0  # 转换为秒
            deadtime[j] = 0  # 假设 deadtime 为 0，根据实际情况修改
            
            # 处理每个通道的数据（只处理保存的通道）
            for channel_idx, original_channel in enumerate(channel_list):
                V1725_1_pulse = channel_data[:, channel_idx, j]
                # 注释掉所有计算，只保留原始波形读取和时间计算
                # 如果需要计算，可以在这里添加相应的代码
                # original_channel 是原始通道索引（0-15）
                # channel_idx 是在保存数组中的索引（0到len(channel_list)-1）
        
        # 更新时间数据（加上pstt）
        time_data = time_data + pstt
        
        # 保存文件 - 使用优化的保存方法
        filename = os.path.basename(run_filename)
        base_filename = os.path.join(save_path, f'{os.path.splitext(filename)[0]}_processed')
        
        # 确保保存目录存在
        try:
            os.makedirs(save_path, exist_ok=True)
            print(f'保存目录: {save_path}')
        except Exception as e:
            print(f'创建保存目录失败: {save_path}')
            print(f'错误信息: {e}')
            raise
        
        start_time = time.time()
        # 保存所有数据 - 使用v7.3格式，因为channel_data可能超过2GB
        mdict = {
            'channel_data': channel_data,
            'time_data': time_data
        }
        
        # 使用HDF5原生格式（.h5）- 更快，MATLAB可以直接读取
        output_file = f'{base_filename}.h5'
        save_hdf5(output_file, mdict)

        # 若包含 CH0 且指定了 ch0 参数目录，在解析时顺带计算并保存：
        # - max_ch0: 每个事件 CH0 的最大值
        # - tmax_ch0: 达峰时刻的 sample 索引
        # - ch0ped_mean/ch0ped_var: 前 500 点（或不足 500 时全部）的均值和方差
        # - ch0pedt_mean/ch0pedt_var: 后 500 点（或不足 500 时全部）的均值和方差
        # - ch0ped_rms: 前 500 点对线性拟合残差的 RMS（CH0pedRMS）
        # - ch0pedt_rms: 后 500 点对线性拟合残差的 RMS（CH0pedtRMS）
        if ch0parameters_save_dir is not None and 0 in channel_list:
            ch0_idx = channel_list.index(0)
            ch0max_file = os.path.join(ch0parameters_save_dir, os.path.basename(output_file))
            if not os.path.exists(ch0max_file):
                os.makedirs(ch0parameters_save_dir, exist_ok=True)
                ch0_wave = channel_data[:, ch0_idx, :]  # (n_samples, n_events)
                ch0_wave_f32 = np.asarray(ch0_wave, dtype=np.float32)

                # 达峰信息
                max_vals = ch0_wave_f32.max(axis=0)
                tmax_vals = np.argmax(ch0_wave_f32, axis=0).astype(np.uint32)

                # pedestal: 前/后 500 点的均值和方差
                n_samples = ch0_wave_f32.shape[0]
                n_ped = min(500, n_samples)
                front_seg = ch0_wave_f32[:n_ped, :]
                back_seg = ch0_wave_f32[-n_ped:, :]

                ch0ped_mean = front_seg.mean(axis=0)
                ch0ped_var = front_seg.var(axis=0)
                ch0pedt_mean = back_seg.mean(axis=0)
                ch0pedt_var = back_seg.var(axis=0)

                # 线性拟合 RMS（对每个 event，在前/后 500 点上做一次线性拟合 y = a x + b）
                # x 取样本索引 0..n_ped-1，所有 event 共用一套 x
                x = np.arange(n_ped, dtype=np.float32).reshape(-1, 1)           # (n_ped, 1)
                x2_sum = float((x * x).sum())
                x_sum = float(x.sum())
                N = float(n_ped)
                denom = N * x2_sum - x_sum * x_sum
                if denom == 0:
                    # 极端情况（几乎不可能），退化为用均值拟合，RMS ~ 标准差
                    ch0ped_rms = np.sqrt(ch0ped_var)
                    ch0pedt_rms = np.sqrt(ch0pedt_var)
                else:
                    # 前 500 点：front_seg 形状 (n_ped, n_events)
                    y_front = front_seg
                    y_sum_front = y_front.sum(axis=0)
                    xy_sum_front = (x * y_front).sum(axis=0)
                    a_front = (N * xy_sum_front - x_sum * y_sum_front) / denom
                    b_front = (y_sum_front - a_front * x_sum) / N
                    y_fit_front = a_front.reshape(1, -1) * x + b_front.reshape(1, -1)
                    resid_front = y_front - y_fit_front
                    ch0ped_rms = np.sqrt((resid_front * resid_front).mean(axis=0))

                    # 后 500 点：back_seg 形状 (n_ped, n_events)
                    y_back = back_seg
                    y_sum_back = y_back.sum(axis=0)
                    xy_sum_back = (x * y_back).sum(axis=0)
                    a_back = (N * xy_sum_back - x_sum * y_sum_back) / denom
                    b_back = (y_sum_back - a_back * x_sum) / N
                    y_fit_back = a_back.reshape(1, -1) * x + b_back.reshape(1, -1)
                    resid_back = y_back - y_fit_back
                    ch0pedt_rms = np.sqrt((resid_back * resid_back).mean(axis=0))

                with h5py.File(ch0max_file, 'w') as f_dst:
                    f_dst.create_dataset('max_ch0', data=max_vals)
                    f_dst.create_dataset('tmax_ch0', data=tmax_vals)
                    f_dst.create_dataset('ch0ped_mean', data=ch0ped_mean)
                    f_dst.create_dataset('ch0ped_var', data=ch0ped_var)
                    f_dst.create_dataset('ch0pedt_mean', data=ch0pedt_mean)
                    f_dst.create_dataset('ch0pedt_var', data=ch0pedt_var)
                    f_dst.create_dataset('ch0ped_rms', data=ch0ped_rms)
                    f_dst.create_dataset('ch0pedt_rms', data=ch0pedt_rms)
                    f_dst.attrs['source_file'] = str(os.path.abspath(output_file))
                    f_dst.attrs['channel_index'] = int(ch0_idx)
                    f_dst.attrs['description'] = (
                        'Per-event CH0 features: max, argmax(sample index), '
                        'front/back pedestal (first/last samples) mean & var, '
                        'and linear-fit RMS for front/back pedestal.'
                    )
                print(f'CH0_parameters 已写入 {os.path.basename(ch0max_file)}')

        # 对 CH1 做一模一样的参数化存储过程，输出到 CH1_parameters 目录
        if 1 in channel_list:
            ch1_idx = channel_list.index(1)
            ch1_file = os.path.join(ch1parameters_save_path, os.path.basename(output_file))
            if not os.path.exists(ch1_file):
                os.makedirs(ch1parameters_save_path, exist_ok=True)
                ch1_wave = channel_data[:, ch1_idx, :]  # (n_samples, n_events)
                ch1_wave_f32 = np.asarray(ch1_wave, dtype=np.float32)

                # 达峰信息
                max_ch1 = ch1_wave_f32.max(axis=0)
                tmax_ch1 = np.argmax(ch1_wave_f32, axis=0).astype(np.uint32)

                # pedestal: 前/后 500 点的均值和方差
                n_samples_1 = ch1_wave_f32.shape[0]
                n_ped_1 = min(500, n_samples_1)
                front_seg_1 = ch1_wave_f32[:n_ped_1, :]
                back_seg_1 = ch1_wave_f32[-n_ped_1:, :]

                ch1ped_mean = front_seg_1.mean(axis=0)
                ch1ped_var = front_seg_1.var(axis=0)
                ch1pedt_mean = back_seg_1.mean(axis=0)
                ch1pedt_var = back_seg_1.var(axis=0)

                # 线性拟合 RMS（CH1pedRMS / CH1pedtRMS）
                x1 = np.arange(n_ped_1, dtype=np.float32).reshape(-1, 1)
                x1_2_sum = float((x1 * x1).sum())
                x1_sum = float(x1.sum())
                N1 = float(n_ped_1)
                denom1 = N1 * x1_2_sum - x1_sum * x1_sum
                if denom1 == 0:
                    ch1ped_rms = np.sqrt(ch1ped_var)
                    ch1pedt_rms = np.sqrt(ch1pedt_var)
                else:
                    y_front_1 = front_seg_1
                    y_sum_front_1 = y_front_1.sum(axis=0)
                    xy_sum_front_1 = (x1 * y_front_1).sum(axis=0)
                    a_front_1 = (N1 * xy_sum_front_1 - x1_sum * y_sum_front_1) / denom1
                    b_front_1 = (y_sum_front_1 - a_front_1 * x1_sum) / N1
                    y_fit_front_1 = a_front_1.reshape(1, -1) * x1 + b_front_1.reshape(1, -1)
                    resid_front_1 = y_front_1 - y_fit_front_1
                    ch1ped_rms = np.sqrt((resid_front_1 * resid_front_1).mean(axis=0))

                    y_back_1 = back_seg_1
                    y_sum_back_1 = y_back_1.sum(axis=0)
                    xy_sum_back_1 = (x1 * y_back_1).sum(axis=0)
                    a_back_1 = (N1 * xy_sum_back_1 - x1_sum * y_sum_back_1) / denom1
                    b_back_1 = (y_sum_back_1 - a_back_1 * x1_sum) / N1
                    y_fit_back_1 = a_back_1.reshape(1, -1) * x1 + b_back_1.reshape(1, -1)
                    resid_back_1 = y_back_1 - y_fit_back_1
                    ch1pedt_rms = np.sqrt((resid_back_1 * resid_back_1).mean(axis=0))

                with h5py.File(ch1_file, 'w') as f_ch1:
                    f_ch1.create_dataset('max_ch1', data=max_ch1)
                    f_ch1.create_dataset('tmax_ch1', data=tmax_ch1)
                    f_ch1.create_dataset('ch1ped_mean', data=ch1ped_mean)
                    f_ch1.create_dataset('ch1ped_var', data=ch1ped_var)
                    f_ch1.create_dataset('ch1pedt_mean', data=ch1pedt_mean)
                    f_ch1.create_dataset('ch1pedt_var', data=ch1pedt_var)
                    f_ch1.create_dataset('ch1ped_rms', data=ch1ped_rms)
                    f_ch1.create_dataset('ch1pedt_rms', data=ch1pedt_rms)
                    f_ch1.attrs['source_file'] = str(os.path.abspath(output_file))
                    f_ch1.attrs['channel_index'] = int(ch1_idx)
                    f_ch1.attrs['description'] = (
                        'Per-event CH1 features: max, argmax(sample index), '
                        'front/back pedestal (first/last samples) mean & var, '
                        'and linear-fit RMS for front/back pedestal.'
                    )
                print(f'CH1_parameters 已写入 {os.path.basename(ch1_file)}')

        # 对 CH2 和 CH3 使用快速拟合函数进行参数化，并分别写入 CH2_parameters / CH3_parameters
        # 仅计算一次：若对应参数文件已存在则跳过

        # CH2
        if 2 in channel_list:
            ch2_idx = channel_list.index(2)
            ch2_file = os.path.join(ch2parameters_save_path, os.path.basename(output_file))
            if not os.path.exists(ch2_file):
                os.makedirs(ch2parameters_save_path, exist_ok=True)
                ch2_wave = channel_data[:, ch2_idx, :]  # (n_samples, n_events)
                ch2_wave_f32 = np.asarray(ch2_wave, dtype=np.float32)
                n_events_2 = ch2_wave_f32.shape[1]

                tanh_p0_2 = np.empty(n_events_2, dtype=np.float32)
                tanh_p1_2 = np.empty(n_events_2, dtype=np.float32)
                tanh_p2_2 = np.empty(n_events_2, dtype=np.float32)
                tanh_p3_2 = np.empty(n_events_2, dtype=np.float32)
                tanh_rms_2 = np.empty(n_events_2, dtype=np.float32)
                highfreq_energy_ratio_2 = np.empty(n_events_2, dtype=np.float32)

                for ev in range(n_events_2):
                    wave_ev_2 = ch2_wave_f32[:, ev]
                    params = _compute_fast_fit_params(wave_ev_2)
                    tanh_p0_2[ev] = params["tanh_p0"]
                    tanh_p1_2[ev] = params["tanh_p1"]
                    tanh_p2_2[ev] = params["tanh_p2"]
                    tanh_p3_2[ev] = params["tanh_p3"]
                    tanh_rms_2[ev] = params["tanh_rms"]

                    # 计算高频能量占比，并写入 CH2_parameters
                    highfreq_energy_ratio_2[ev] = _compute_fast_highfreq_energy_ratio(
                        wave_ev_2
                    )

                    # 简单进度条：每完成约 5% 或最后一个事件时打印一次
                    if n_events_2 >= 20:
                        step = max(1, n_events_2 // 20)
                        if (ev + 1) % step == 0 or ev + 1 == n_events_2:
                            pct = 100.0 * (ev + 1) / n_events_2
                            print(f'CH2 拟合进度: {ev + 1}/{n_events_2} ({pct:.1f}%)')
                print('CH2 拟合完成')

                with h5py.File(ch2_file, 'w') as f_ch2:
                    f_ch2.create_dataset('tanh_p0', data=tanh_p0_2)
                    f_ch2.create_dataset('tanh_p1', data=tanh_p1_2)
                    f_ch2.create_dataset('tanh_p2', data=tanh_p2_2)
                    f_ch2.create_dataset('tanh_p3', data=tanh_p3_2)
                    f_ch2.create_dataset('tanh_rms', data=tanh_rms_2)
                    f_ch2.create_dataset('highfreq_energy_ratio', data=highfreq_energy_ratio_2)
                    f_ch2.attrs['source_file'] = str(os.path.abspath(output_file))
                    f_ch2.attrs['channel_index'] = int(ch2_idx)
                    f_ch2.attrs['description'] = (
                        'Per-event CH2 fast tanh-fit parameters (p0, p1, p2, p3, rms) '
                        'and high-frequency energy ratio (>0.2 MHz).'
                    )
                print(f'CH2_parameters 已写入 {os.path.basename(ch2_file)}')

        # CH3
        if 3 in channel_list:
            ch3_idx = channel_list.index(3)
            ch3_file = os.path.join(ch3parameters_save_path, os.path.basename(output_file))
            if not os.path.exists(ch3_file):
                os.makedirs(ch3parameters_save_path, exist_ok=True)
                ch3_wave = channel_data[:, ch3_idx, :]  # (n_samples, n_events)
                ch3_wave_f32 = np.asarray(ch3_wave, dtype=np.float32)
                n_events_3 = ch3_wave_f32.shape[1]

                tanh_p0_3 = np.empty(n_events_3, dtype=np.float32)
                tanh_p1_3 = np.empty(n_events_3, dtype=np.float32)
                tanh_p2_3 = np.empty(n_events_3, dtype=np.float32)
                tanh_p3_3 = np.empty(n_events_3, dtype=np.float32)
                tanh_rms_3 = np.empty(n_events_3, dtype=np.float32)
                highfreq_energy_ratio_3 = np.empty(n_events_3, dtype=np.float32)

                print(f'CH3 拟合开始，总事件数: {n_events_3}')
                for ev in range(n_events_3):
                    wave_ev = ch3_wave_f32[:, ev]
                    params = _compute_fast_fit_params(wave_ev)
                    tanh_p0_3[ev] = params["tanh_p0"]
                    tanh_p1_3[ev] = params["tanh_p1"]
                    tanh_p2_3[ev] = params["tanh_p2"]
                    tanh_p3_3[ev] = params["tanh_p3"]
                    tanh_rms_3[ev] = params["tanh_rms"]

                    # 计算快放高频能量占比，并写入 CH3_parameters
                    highfreq_energy_ratio_3[ev] = _compute_fast_highfreq_energy_ratio(
                        wave_ev
                    )

                    # 简单进度条：每完成约 5% 或最后一个事件时打印一次
                    if n_events_3 >= 20:
                        step = max(1, n_events_3 // 20)
                        if (ev + 1) % step == 0 or ev + 1 == n_events_3:
                            pct = 100.0 * (ev + 1) / n_events_3
                            print(f'CH3 拟合进度: {ev + 1}/{n_events_3} ({pct:.1f}%)')
                print('CH3 拟合完成')

                with h5py.File(ch3_file, 'w') as f_ch3:
                    f_ch3.create_dataset('tanh_p0', data=tanh_p0_3)
                    f_ch3.create_dataset('tanh_p1', data=tanh_p1_3)
                    f_ch3.create_dataset('tanh_p2', data=tanh_p2_3)
                    f_ch3.create_dataset('tanh_p3', data=tanh_p3_3)
                    f_ch3.create_dataset('tanh_rms', data=tanh_rms_3)
                    f_ch3.create_dataset('highfreq_energy_ratio', data=highfreq_energy_ratio_3)
                    f_ch3.attrs['source_file'] = str(os.path.abspath(output_file))
                    f_ch3.attrs['channel_index'] = int(ch3_idx)
                    f_ch3.attrs['description'] = (
                        'Per-event CH3 fast tanh-fit parameters (p0, p1, p2, p3, rms) '
                        'and high-frequency energy ratio (>0.2 MHz).'
                    )
                print(f'CH3_parameters 已写入 {os.path.basename(ch3_file)}')

        # CH4：仅保存每个事件的最大值
        if 4 in channel_list:
            ch4_idx = channel_list.index(4)
            ch4_file = os.path.join(ch4parameters_save_path, os.path.basename(output_file))
            if not os.path.exists(ch4_file):
                os.makedirs(ch4parameters_save_path, exist_ok=True)
                ch4_wave = channel_data[:, ch4_idx, :]  # (n_samples, n_events)
                ch4_wave_f32 = np.asarray(ch4_wave, dtype=np.float32)
                max_ch4 = ch4_wave_f32.max(axis=0)

                with h5py.File(ch4_file, 'w') as f_ch4:
                    f_ch4.create_dataset('max_ch4', data=max_ch4)
                    f_ch4.attrs['source_file'] = str(os.path.abspath(output_file))
                    f_ch4.attrs['channel_index'] = int(ch4_idx)
                    f_ch4.attrs['description'] = (
                        'Per-event CH4 feature: maximum value only.'
                    )
                print(f'CH4_parameters 已写入 {os.path.basename(ch4_file)}')

        # CH5：仅保存每个事件的最大值
        if 5 in channel_list:
            ch5_idx = channel_list.index(5)
            ch5_file = os.path.join(ch5parameters_save_path, os.path.basename(output_file))
            if not os.path.exists(ch5_file):
                os.makedirs(ch5parameters_save_path, exist_ok=True)
                ch5_wave = channel_data[:, ch5_idx, :]  # (n_samples, n_events)
                ch5_wave_f32 = np.asarray(ch5_wave, dtype=np.float32)
                max_ch5 = ch5_wave_f32.max(axis=0)

                with h5py.File(ch5_file, 'w') as f_ch5:
                    f_ch5.create_dataset('max_ch5', data=max_ch5)
                    f_ch5.attrs['source_file'] = str(os.path.abspath(output_file))
                    f_ch5.attrs['channel_index'] = int(ch5_idx)
                    f_ch5.attrs['description'] = (
                        'Per-event CH5 feature: maximum value only.'
                    )
                print(f'CH5_parameters 已写入 {os.path.basename(ch5_file)}')
        
        print(f'保存完成，耗时: {time.time() - start_time:.2f}秒')
        #print(f'输出文件路径: {os.path.abspath(output_file)}')
        #print(f'MATLAB读取方式: channel_data = h5read("{os.path.basename(output_file)}", "/channel_data");')
        #print(f'                  time_data = h5read("{os.path.basename(output_file)}", "/time_data");')
        
        # 显示文件大小信息
        if os.path.exists(output_file):
            file_size_GB = os.path.getsize(output_file) / (1024**3)
            print(f'文件大小: {file_size_GB:.2f} GB')
            print(f'文件已确认存在: {os.path.abspath(output_file)}')
        else:
            print(f'警告：文件不存在: {output_file}')
            print(f'完整路径: {os.path.abspath(output_file)}')
        print('数据保存完成！')
        
    except Exception as e:
        if fid is not None:
            fid.close()
        raise

def main():
    """主函数 - 使用多进程并行处理"""
    # 生成所有任务列表
    tasks = []
    for i in range(RUN_Start_NUMBER, RUN_End_NUMBER + 1):
        run_filename = os.path.join(read_path, f'{filename_input}FADC_RAW_Data_{i}.bin')
        filename = os.path.basename(run_filename)
        base_name = os.path.splitext(filename)[0]
        # 为每个文件添加任务：AMP通道、TRIGGER通道和NAI通道
        # 仅当对应输出文件夹下该文件的所有 h5（CH0-3 + CH0/CH1/CH2/CH3 参数）都已存在时才跳过
        amp_output_file = os.path.join(amp_save_path, f'{base_name}_processed.h5')
        ch0param_output_file = os.path.join(ch0parameters_save_path, f'{base_name}_processed.h5')
        ch1param_output_file = os.path.join(ch1parameters_save_path, f'{base_name}_processed.h5')
        ch2param_output_file = os.path.join(ch2parameters_save_path, f'{base_name}_processed.h5')
        ch3param_output_file = os.path.join(ch3parameters_save_path, f'{base_name}_processed.h5')
        amp_complete = (
            os.path.exists(amp_output_file)
            and os.path.exists(ch0param_output_file)
            and os.path.exists(ch1param_output_file)
            and os.path.exists(ch2param_output_file)
            and os.path.exists(ch3param_output_file)
        )
        if not amp_complete:
            tasks.append((run_filename, AMP_CHANNEL_LIST, EVENT_NUMBER, amp_save_path, ch0parameters_save_path))
        else:
            print(
                f'跳过 {run_filename} (通道: {AMP_CHANNEL_LIST})，'
                f'CH0-3 与 CH0/CH1/CH2/CH3 参数文件均已存在'
            )

        trigger_output_file = os.path.join(trigger_save_path, f'{base_name}_processed.h5')
        if not os.path.exists(trigger_output_file):
            tasks.append((run_filename, TRIGGER_CHANNEL_LIST, EVENT_NUMBER, trigger_save_path, None))
        else:
            print(f'跳过 {run_filename} (通道: {TRIGGER_CHANNEL_LIST})，已存在: {trigger_output_file}')

        nai_output_file = os.path.join(NAI_save_path, f'{base_name}_processed.h5')
        if not os.path.exists(nai_output_file):
            tasks.append((run_filename, NAI_CHANNEL_LIST, EVENT_NUMBER, NAI_save_path, None))
        else:
            print(f'跳过 {run_filename} (通道: {NAI_CHANNEL_LIST})，已存在: {nai_output_file}')

    # 获取可用CPU核心数，最多使用所有核心
    max_workers = os.cpu_count()
    print(f'使用 {max_workers} 个CPU核心进行并行处理')
    print(f'共 {len(tasks)} 个任务')

    # 使用进程池并行执行
    start_time = time.time()
    success_count = 0
    fail_count = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务，直接使用 bin2rawpulse 函数
        future_to_task = {executor.submit(bin2rawpulse, *task): task for task in tasks}
        # 收集结果
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            run_filename, channel_list, _, _, _ = task
            try:
                future.result()  # 如果成功，没有返回值也没有异常
                success_count += 1
                print(f'[成功 {success_count}] 成功处理: {run_filename} (通道: {channel_list})')
            except Exception as e:
                fail_count += 1
                print(f'[失败 {fail_count}] 处理文件 {run_filename} (通道: {channel_list}) 时出错: {e}')
    
    elapsed_time = time.time() - start_time
    print('=' * 60)
    print(f'所有任务处理完成！')
    print(f'总耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)')
    print(f'成功: {success_count} 个任务')
    print(f'失败: {fail_count} 个任务')


if __name__ == '__main__':
    main()

