#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
二进制文件预处理脚本
将V1725数据采集卡的.bin文件转换为.mat格式的原始脉冲数据
"""
import os
import sys
import struct
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
import psutil
import gc

# 添加父目录到路径，以便导入utils模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from utils.save import save_hdf5

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


def wait_for_memory_below(
    max_used_percent: float = 95.0,
    check_interval_sec: float = 5.0,) -> None:
    """
    简单的内存守门函数：
    - 当当前进程所在机器的内存占用超过 max_used_percent 时，不继续往下读大文件，
      而是等待一段时间后重试；
    - 一旦内存占用恢复到阈值以下，立刻返回，让后续读取/计算继续。
    """
    while True:
        mem = psutil.virtual_memory()
        used = mem.percent
        if used <= max_used_percent:
            return
        print(
            f"[内存保护] 当前内存占用 {used:.1f}% > 阈值 {max_used_percent:.1f}%，"
            f"暂停 {check_interval_sec:.1f}s 等待释放..."
        )
        time.sleep(check_interval_sec)

def _compute_and_save_parameters(output_file, channel_list, channel_data, ch0parameters_save_dir):
    """
    基于给定的 channel_data / channel_list 和 output_file，
    计算 CH0~CH5 的参数并写入各自的 *parameters 目录。
    该函数既可在 bin2rawpulse 内部调用，也可在已有 h5 文件的基础上单独调用。
    """
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

            # 达峰信息与最小值信息
            max_vals = ch0_wave_f32.max(axis=0)
            tmax_vals = np.argmax(ch0_wave_f32, axis=0).astype(np.uint32)
            ch0_min = ch0_wave_f32.min(axis=0)

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
                f_dst.create_dataset('ch0_min', data=ch0_min)
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

            # 达峰信息与最小值信息
            max_ch1 = ch1_wave_f32.max(axis=0)
            tmax_ch1 = np.argmax(ch1_wave_f32, axis=0).astype(np.uint32)
            ch1_min = ch1_wave_f32.min(axis=0)

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
                f_ch1.create_dataset('ch1_min', data=ch1_min)
                f_ch1.create_dataset('ch1ped_mean', data=ch1ped_mean)
                f_ch1.create_dataset('ch1ped_var', data=ch1ped_var)
                f_ch1.create_dataset('ch1pedt_mean', data=ch1pedt_mean)
                f_ch1.create_dataset('ch1pedt_var', data=ch1pedt_var)
                f_ch1.create_dataset('ch1ped_rms', data=ch1ped_rms)
                f_ch1.create_dataset('ch1pedt_rms', data=ch1pedt_rms)
                f_ch1.attrs['source_file'] = str(os.path.abspath(output_file))
                f_ch1.attrs['channel_index'] = int(ch1_idx)
                f_ch1.attrs['description'] = (
                    'Per-event CH1 features: max, argmax(sample index), min, '
                    'front/back pedestal (first/last samples) mean & var, '
                    'and linear-fit RMS for front/back pedestal.'
                )
            print(f'CH1_parameters 已写入 {os.path.basename(ch1_file)}')

    # CH2/CH3 的拟合和高频能量占比改为在独立脚本中并行处理，
    # 这里不再计算，以便在那个阶段独占全部 CPU。

    # CH4：保存每个事件的最大值及达峰时间
    if 4 in channel_list:
        ch4_idx = channel_list.index(4)
        ch4_file = os.path.join(ch4parameters_save_path, os.path.basename(output_file))
        if not os.path.exists(ch4_file):
            os.makedirs(ch4parameters_save_path, exist_ok=True)
            ch4_wave = channel_data[:, ch4_idx, :]  # (n_samples, n_events)
            ch4_wave_f32 = np.asarray(ch4_wave, dtype=np.float32)
            max_ch4 = ch4_wave_f32.max(axis=0)
            tmax_ch4 = np.argmax(ch4_wave_f32, axis=0).astype(np.uint32)

            with h5py.File(ch4_file, 'w') as f_ch4:
                f_ch4.create_dataset('max_ch4', data=max_ch4)
                f_ch4.create_dataset('tmax_ch4', data=tmax_ch4)
                f_ch4.attrs['source_file'] = str(os.path.abspath(output_file))
                f_ch4.attrs['channel_index'] = int(ch4_idx)
                f_ch4.attrs['description'] = (
                    'Per-event CH4 features: maximum value and argmax(sample index).'
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

def compute_parameters_from_existing_h5(output_file, channel_list, ch0parameters_save_dir):
    """
    当原始 CHx h5 已经存在时，仅基于现有 h5 中的 channel_data 计算并写入参数。
    不再重复读取 bin 文件。

    注意：为降低内存峰值，这里对部分通道采用“流式”计算，避免一次性读入整个
    channel_data 三维数组。
    """
    # 在读取大数据前先检查整体内存占用
    wait_for_memory_below()

    with h5py.File(output_file, 'r') as f_src:
        dset = f_src['channel_data']

        # 仅包含 CH4：需要每个事件的最大值及达峰时间，可按样本维度分块流式计算
        if channel_list == NAI_CHANNEL_LIST:
            time_samples, num_channels, num_events = dset.shape
            if num_channels != 1:
                raise ValueError(
                    f"预期 CH4 原始 h5 只有 1 个通道，但实际 num_channels={num_channels}"
                )

            print(f"[参数计算] 基于已有 CH4 h5 计算 CH4_parameters：{output_file}")
            max_ch4 = np.full(num_events, -np.inf, dtype=np.float32)
            tmax_ch4 = np.zeros(num_events, dtype=np.int64)

            # 按时间样本维度分块，逐步更新每个事件的最大值及对应的 sample 索引
            chunk_size = 1024
            for start in range(0, time_samples, chunk_size):
                end = min(start + chunk_size, time_samples)
                # 只读取 [start:end] 这一小段的 CH4 数据，显著降低内存占用
                chunk = dset[start:end, 0, :]  # 形状 (chunk_len, n_events)
                # 对每个事件在当前块内找到最大值及局部索引
                local_argmax = np.argmax(chunk, axis=0)               # (n_events,)
                chunk_max = chunk[local_argmax, np.arange(num_events)]  # (n_events,)

                # 找到那些通过当前块更新了全局最大值的事件
                mask_update = chunk_max > max_ch4
                # 更新这些事件的全局最大值与对应的全局达峰 sample 索引
                max_ch4[mask_update] = chunk_max[mask_update].astype(np.float32)
                tmax_ch4[mask_update] = start + local_argmax[mask_update]

            ch4_file = output_file.replace(
                os.path.join('raw_pulse', 'CH4'),
                os.path.join('raw_pulse', 'CH4_parameters'),
            )
            ch4_dir = os.path.dirname(ch4_file)
            os.makedirs(ch4_dir, exist_ok=True)

            with h5py.File(ch4_file, 'w') as f_ch4:
                f_ch4.create_dataset('max_ch4', data=max_ch4)
                f_ch4.create_dataset('tmax_ch4', data=tmax_ch4.astype(np.uint32))
                f_ch4.attrs['source_file'] = str(os.path.abspath(output_file))
                f_ch4.attrs['channel_index'] = int(4)
                f_ch4.attrs['description'] = (
                    'Per-event CH4 features: maximum value and argmax(sample index) (streamed).'
                )

            print(f'CH4_parameters 已写入 {os.path.basename(ch4_file)}')
            return

        # 仅包含 CH5：同样只需要每个事件的最大值，按样本维度分块流式计算
        if channel_list == TRIGGER_CHANNEL_LIST:
            time_samples, num_channels, num_events = dset.shape
            if num_channels != 1:
                raise ValueError(
                    f"预期 CH5 原始 h5 只有 1 个通道，但实际 num_channels={num_channels}"
                )

            print(f"[参数计算] 基于已有 CH5 h5 计算 CH5_parameters：{output_file}")
            max_ch5 = np.full(num_events, -np.inf, dtype=np.float32)

            chunk_size = 1024
            for start in range(0, time_samples, chunk_size):
                end = min(start + chunk_size, time_samples)
                chunk = dset[start:end, 0, :]  # (chunk_len, n_events)
                chunk_max = chunk.max(axis=0).astype(np.float32)
                max_ch5 = np.maximum(max_ch5, chunk_max)

            ch5_file = output_file.replace(
                os.path.join('raw_pulse', 'CH5'),
                os.path.join('raw_pulse', 'CH5_parameters'),
            )
            ch5_dir = os.path.dirname(ch5_file)
            os.makedirs(ch5_dir, exist_ok=True)

            with h5py.File(ch5_file, 'w') as f_ch5:
                f_ch5.create_dataset('max_ch5', data=max_ch5)
                f_ch5.attrs['source_file'] = str(os.path.abspath(output_file))
                f_ch5.attrs['channel_index'] = int(5)
                f_ch5.attrs['description'] = 'Per-event CH5 feature: maximum value only (streamed).'

            print(f'CH5_parameters 已写入 {os.path.basename(ch5_file)}')
            return

        # 其他情况（主要是 CH0-3）：仍然需要完整的 channel_data 做 pedestal 等复杂参数，
        # 这里维持一次性读入，但配合外层内存阈值与显式释放，尽量降低整体压力。
        channel_data = dset[...]

    try:
        _compute_and_save_parameters(
            output_file=output_file,
            channel_list=channel_list,
            channel_data=channel_data,
            ch0parameters_save_dir=ch0parameters_save_dir,
        )
    finally:
        # 显式释放大数组引用并触发一次垃圾回收，以尽快归还内存
        del channel_data
        gc.collect()

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

    # 在每个 worker 真正开始读 bin 并分配大数组之前，先进行一次内存水位检查。
    # 当内存占用过高时，当前任务会在这里等待，直到有足够空闲内存再继续。
    wait_for_memory_below()

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

        # 基于当前 channel_data 计算各通道参数并写入对应的 *parameters 目录
        try:
            _compute_and_save_parameters(
                output_file=output_file,
                channel_list=channel_list,
                channel_data=channel_data,
                ch0parameters_save_dir=ch0parameters_save_dir,
            )
        finally:
            # 无论参数计算是否成功，都尽快释放大数组和中间缓存
            del channel_data
            del hit_pat_array, v1729_tg_rec_array, evt_endtime_array
            del v1725_1_tgno_array, v1725_1_tag_array
            del idevt, trig, time_array, deadtime, time_data
            gc.collect()

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
    tasks = []         # 需要从 bin 读取并生成原始 h5 的任务
    param_tasks = []   # 已有原始 h5，仅需补参数的任务
    for i in range(RUN_Start_NUMBER, RUN_End_NUMBER + 1):
        run_filename = os.path.join(read_path, f'{filename_input}FADC_RAW_Data_{i}.bin')
        filename = os.path.basename(run_filename)
        base_name = os.path.splitext(filename)[0]
        # 为每个文件添加任务：AMP通道、TRIGGER通道和NAI通道
        # 对于 AMP（CH0-3）：若原始 h5 不存在则跑 bin2rawpulse；若原始 h5 存在但参数不全，则只基于 h5 计算参数
        amp_output_file = os.path.join(amp_save_path, f'{base_name}_processed.h5')
        ch0param_output_file = os.path.join(ch0parameters_save_path, f'{base_name}_processed.h5')
        ch1param_output_file = os.path.join(ch1parameters_save_path, f'{base_name}_processed.h5')
        ch2param_output_file = os.path.join(ch2parameters_save_path, f'{base_name}_processed.h5')
        ch3param_output_file = os.path.join(ch3parameters_save_path, f'{base_name}_processed.h5')
        amp_raw_exists = os.path.exists(amp_output_file)
        amp_params_complete = (
            os.path.exists(ch0param_output_file)
            and os.path.exists(ch1param_output_file)
            and os.path.exists(ch2param_output_file)
            and os.path.exists(ch3param_output_file)
        )

        if not amp_raw_exists:
            # 原始 CH0-3 h5 不存在：必须从 bin 解析
            tasks.append((run_filename, AMP_CHANNEL_LIST, EVENT_NUMBER, amp_save_path, ch0parameters_save_path))
        elif not amp_params_complete:
            # 原始 h5 已有，但参数不全：只补参数
            param_tasks.append((amp_output_file, AMP_CHANNEL_LIST, ch0parameters_save_path))
        else:
            print(
                f'跳过 {run_filename} (通道: {AMP_CHANNEL_LIST})，'
                f'CH0-3 与 CH0/CH1/CH2/CH3 参数文件均已存在'
            )

        trigger_output_file = os.path.join(trigger_save_path, f'{base_name}_processed.h5')
        ch5param_output_file = os.path.join(ch5parameters_save_path, f'{base_name}_processed.h5')
        trigger_raw_exists = os.path.exists(trigger_output_file)
        ch5param_exists = os.path.exists(ch5param_output_file)
        if not trigger_raw_exists:
            # 随机触发原始 h5 不存在：从 bin 解析
            tasks.append((run_filename, TRIGGER_CHANNEL_LIST, EVENT_NUMBER, trigger_save_path, None))
        elif not ch5param_exists:
            # 仅缺少 CH5_parameters：基于已有 h5 计算
            param_tasks.append((trigger_output_file, TRIGGER_CHANNEL_LIST, None))
        else:
            print(
                f'跳过 {run_filename} (通道: {TRIGGER_CHANNEL_LIST})，'
                f'CH5 与 CH5_parameters 文件均已存在'
            )

        nai_output_file = os.path.join(NAI_save_path, f'{base_name}_processed.h5')
        ch4param_output_file = os.path.join(ch4parameters_save_path, f'{base_name}_processed.h5')
        nai_raw_exists = os.path.exists(nai_output_file)
        ch4param_exists = os.path.exists(ch4param_output_file)
        if not nai_raw_exists:
            # CH4 原始 h5 不存在：从 bin 解析
            tasks.append((run_filename, NAI_CHANNEL_LIST, EVENT_NUMBER, NAI_save_path, None))
        elif not ch4param_exists:
            # 仅缺少 CH4_parameters：基于已有 h5 计算
            param_tasks.append((nai_output_file, NAI_CHANNEL_LIST, None))
        else:
            print(
                f'跳过 {run_filename} (通道: {NAI_CHANNEL_LIST})，'
                f'CH4 与 CH4_parameters 文件均已存在'
            )

    # 根据机器内存和 CPU 情况自适应限制并行 worker 数，避免内存爆掉
    total_mem_gb = psutil.virtual_memory().total / (1024 ** 3)
    # 粗略估计：每个 worker 预留 ~4GB，如果机器内存较小则至少保留 1 个 worker
    max_workers_by_mem = max(1, int(total_mem_gb // 4))
    cpu_count = os.cpu_count() or 1
    max_workers = max(1, min(cpu_count, max_workers_by_mem))

    print(f'CPU 核心数: {cpu_count}，物理内存: {total_mem_gb:.1f} GB，'
          f'限制并行 worker 数为: {max_workers}')
    print(f'共 {len(tasks)} 个 bin 解析任务，{len(param_tasks)} 个仅参数任务')

    start_time = time.time()
    success_count = 0
    fail_count = 0

    # 使用进程池并行执行：既包含 bin2rawpulse 任务，也包含仅参数任务
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_desc = {}

        # 提交 bin 解析任务
        for run_filename, channel_list, event_number, save_path, ch0param_dir in tasks:
            fut = executor.submit(
                bin2rawpulse,
                run_filename,
                channel_list,
                event_number,
                save_path,
                ch0param_dir,
            )
            future_to_desc[fut] = f'bin2rawpulse: {run_filename} (通道: {channel_list})'

        # 提交仅参数任务
        for output_file, channel_list, ch0param_dir in param_tasks:
            fut = executor.submit(
                compute_parameters_from_existing_h5,
                output_file,
                channel_list,
                ch0param_dir,
            )
            future_to_desc[fut] = f'compute_parameters_from_existing_h5: {output_file} (通道: {channel_list})'

        # 收集结果
        for fut in as_completed(future_to_desc):
            desc = future_to_desc[fut]
            try:
                fut.result()
                success_count += 1
                print(f'[成功 {success_count}] {desc}')
            except Exception as e:
                fail_count += 1
                print(f'[失败 {fail_count}] {desc} 时出错: {e}')

    elapsed_time = time.time() - start_time
    print('=' * 60)
    print(f'所有任务处理完成！')
    print(f'总耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)')
    print(f'成功: {success_count} 个任务')
    print(f'失败: {fail_count} 个任务')

if __name__ == '__main__':
    main()

