#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
振动加速度数据和GMM pn cut component 1事件计数率联合绘制脚本
从振动传感器HDF5文件读取加速度数据（5.28-6.10日），
并与gmmpncut.py中component 1对应的事件触发时间计数率绘制在一起
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from datetime import datetime
from typing import Optional, Tuple
import importlib.util
import h5py

# 添加路径以便导入其他模块
current_dir = os.path.dirname(os.path.abspath(__file__))
# current_dir = python/data/coincident
parent_dir = os.path.dirname(current_dir)  # python/data
grandparent_dir = os.path.dirname(parent_dir)  # python
project_root = os.path.dirname(grandparent_dir)  # 项目根目录

# 导入振动加速度模块
accelerate_select_path = os.path.join(
    grandparent_dir, 'data', 'sensor', 'vibration', 'accelerate', 'select.py'
)
spec_accelerate = importlib.util.spec_from_file_location(
    "accelerate_select", accelerate_select_path
)
accelerate_select = importlib.util.module_from_spec(spec_accelerate)
spec_accelerate.loader.exec_module(accelerate_select)
select_by_date_range_vibration = accelerate_select.select_by_date_range_vibration

# 导入gmmpncut模块的逻辑（复用其GMM筛选逻辑）
ge_self_cut_path = os.path.join(
    grandparent_dir, 'data', 'ge-self', 'cut', 'gmmpncut.py'
)

# 导入utils模块（将python目录添加到sys.path）
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)
from utils.visualize import get_h5_files


def read_hdf5_event_times(
    hdf5_file_path: str,
    epoch_offset: float = 2.082816000000000e+09,
) -> np.ndarray:
    """
    从HDF5文件读取所有event的时间并转换为datetime
    
    参数:
        hdf5_file_path: HDF5文件路径
        epoch_offset: 时间戳偏移量（默认值来自MATLAB代码）
    
    返回:
        datetime数组，包含所有event的时间
    """
    if not os.path.exists(hdf5_file_path):
        raise FileNotFoundError(f"文件不存在: {hdf5_file_path}")

    with h5py.File(hdf5_file_path, "r") as f:
        if "time_data" not in f:
            raise ValueError(f"HDF5文件中缺少 time_data 数据集")

        time_data = f["time_data"][:]

        if len(time_data) == 0:
            raise ValueError(f"HDF5文件中 time_data 为空")

        # 转换为datetime
        epoch_start = datetime(1970, 1, 1)
        event_times = []

        for time_val in time_data:
            eventtime = time_val - epoch_offset
            event_datetime = epoch_start + pd.Timedelta(seconds=eventtime)
            event_times.append(event_datetime)

        # 转换为numpy datetime64数组以确保类型一致
        return pd.to_datetime(event_times).values


def _fit_single_line_two_step(
    x: np.ndarray,
    y: np.ndarray,
    x_min: float = 2000.0,
    x_max: float = 14000.0,
) -> Tuple[float, float, float]:
    """
    从gmmpncut.py复制的"两步"最小二乘拟合函数
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()

    if x.shape != y.shape:
        raise ValueError(f"x 与 y 形状不一致: {x.shape} vs {y.shape}")

    mask_range = (x > x_min) & (x < x_max)
    n_range = int(mask_range.sum())
    if n_range < 2:
        raise ValueError(
            f"在区间 ({x_min}, {x_max}) 内有效点数不足 2 个，无法拟合直线 "
            f"(有效点数={n_range})"
        )

    x_sel = x[mask_range]
    y_sel = y[mask_range]

    a1, b1 = np.polyfit(x_sel, y_sel, deg=1)

    y_pred = a1 * x_sel + b1
    residuals = y_sel - y_pred
    sigma1 = residuals.std(ddof=1) if residuals.size > 1 else 0.0

    if sigma1 > 0.0:
        inlier_mask = np.abs(residuals) <= 3.0 * sigma1
        n_inliers = int(inlier_mask.sum())

        if 2 <= n_inliers < x_sel.size:
            x_in = x_sel[inlier_mask]
            y_in = y_sel[inlier_mask]
            a2, b2 = np.polyfit(x_in, y_in, deg=1)

            y_pred2 = a2 * x_in + b2
            residuals2 = y_in - y_pred2
            sigma2 = residuals2.std(ddof=1) if residuals2.size > 1 else 0.0
            return float(a2), float(b2), float(sigma2)

    return float(a1), float(b1), float(sigma1)


def get_component1_event_times(
    hdf5_dir: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    time_bin_size: str = "1h",
    epoch_offset: float = 2.082816000000000e+09,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从所有HDF5文件中获取gmmpncut.py component 1对应的事件触发时间，并直接累加到时间bin
    
    参数:
        hdf5_dir: HDF5文件目录，如果为None则自动获取
        start_date: 起始日期，格式 'YYYY-MM-DD'（用于筛选文件）
        end_date: 终止日期，格式 'YYYY-MM-DD'（用于筛选文件）
        start_time: 起始时间（可选），格式 'HH:MM:SS'
        end_time: 终止时间（可选），格式 'HH:MM:SS'
        time_bin_size: 时间bin大小，pandas频率字符串（如'1h', '30min', '1D'）
        epoch_offset: 时间戳偏移量
    
    返回:
        (bin_centers, counts): bin中心时间数组和计数数组
    """
    from sklearn.mixture import GaussianMixture

    print("=" * 70)
    print("获取GMM component 1事件触发时间（从所有HDF5文件）")
    print("=" * 70)

    # 获取所有HDF5文件
    if hdf5_dir is None:
        h5_files = get_h5_files()
        if "CH0-3" not in h5_files or not h5_files["CH0-3"]:
            raise FileNotFoundError("在 data/hdf5/raw_pulse/CH0-3 目录中未找到 h5 文件")
        if "CH5" not in h5_files or not h5_files["CH5"]:
            raise FileNotFoundError("在 data/hdf5/raw_pulse/CH5 目录中未找到 h5 文件")

        ch0_3_files = h5_files["CH0-3"]
        ch5_files = h5_files["CH5"]
    else:
        ch0_3_dir = os.path.join(hdf5_dir, "CH0-3")
        ch5_dir = os.path.join(hdf5_dir, "CH5")
        
        ch0_3_files = glob.glob(os.path.join(ch0_3_dir, "*.h5"))
        ch5_files = glob.glob(os.path.join(ch5_dir, "*.h5"))
        
        ch0_3_files.sort()
        ch5_files.sort()

    # 匹配文件对
    ch0_3_dict = {os.path.basename(f): f for f in ch0_3_files}
    ch5_dict = {os.path.basename(f): f for f in ch5_files}

    matched_pairs = []
    for filename in ch0_3_dict.keys():
        if filename in ch5_dict:
            matched_pairs.append((ch0_3_dict[filename], ch5_dict[filename]))

    if len(matched_pairs) == 0:
        raise ValueError("未找到匹配的CH0-3和CH5文件对")

    print(f"找到 {len(matched_pairs)} 个匹配的文件对")

    # 如果指定了日期范围，筛选文件（基于文件名或文件内容）
    if start_date is not None or end_date is not None:
        print(f"\n按日期范围筛选文件: {start_date} 到 {end_date}")
        start_datetime = None
        end_datetime = None

        if start_date is not None:
            start_date_str = start_date
            if start_time is not None:
                start_date_str = f"{start_date} {start_time}"
            start_datetime = pd.to_datetime(start_date_str)

        if end_date is not None:
            end_date_str = end_date
            if end_time is not None:
                end_date_str = f"{end_date} {end_time}"
            end_datetime = pd.to_datetime(end_date_str)

    # 步骤1: 先读取少量文件训练GMM模型（避免OOM）
    print("\n步骤1: 读取少量文件训练GMM模型...")
    print("  使用前几个文件的数据训练GMM，然后应用到所有文件")
    
    max_files_for_training = 5  # 最多使用前5个文件训练GMM
    training_waveforms_list = []

    def _process_single_file_for_training(ch0_3_file, ch5_file, start_datetime, end_datetime, epoch_offset):
        """处理单个文件，返回物理事例波形数据（用于训练GMM）"""
        # 如果指定了日期范围，先读取文件的时间范围检查
        if start_datetime is not None or end_datetime is not None:
            try:
                file_event_times = read_hdf5_event_times(ch0_3_file, epoch_offset)
                if len(file_event_times) > 0:
                    file_start = file_event_times[0]
                    file_end = file_event_times[-1]
                    
                    # 检查文件时间范围是否与指定范围有重叠
                    if start_datetime is not None and file_end < start_datetime:
                        return None
                    if end_datetime is not None and file_start > end_datetime:
                        return None
            except Exception as e:
                pass  # 继续处理

        # 筛选物理事例（既非RT也非Inhibit）
        rt_cut = 6000.0
        batch_size = 1000

        try:
            with h5py.File(ch0_3_file, "r") as f_ch0:
                ch0_channel_data = f_ch0["channel_data"]
                _, _, ch0_num_events = ch0_channel_data.shape
                ch0_min_values = np.zeros(ch0_num_events, dtype=np.float64)
                for i in range(0, ch0_num_events, batch_size):
                    end_idx = min(i + batch_size, ch0_num_events)
                    batch_data = ch0_channel_data[:, 0, i:end_idx]
                    ch0_min_values[i:end_idx] = np.min(batch_data, axis=0)

            with h5py.File(ch5_file, "r") as f_ch5:
                ch5_channel_data = f_ch5["channel_data"]
                _, _, ch5_num_events = ch5_channel_data.shape
                ch5_max_values = np.zeros(ch5_num_events, dtype=np.float64)
                for i in range(0, ch5_num_events, batch_size):
                    end_idx = min(i + batch_size, ch5_num_events)
                    batch_data = ch5_channel_data[:, 0, i:end_idx]
                    ch5_max_values[i:end_idx] = np.max(batch_data, axis=0)

                rt_mask = ch5_max_values > rt_cut
                inhibit_mask = ch0_min_values == 0
                neither_mask = ~rt_mask & ~inhibit_mask
                selected_indices = np.where(neither_mask)[0]

            physical_count = len(selected_indices)
            if physical_count == 0:
                return None

            # 读取物理事例的波形数据
            waveforms_list = []
            with h5py.File(ch0_3_file, "r") as f:
                channel_data = f["channel_data"]
                time_samples, num_channels, _ = channel_data.shape

                if num_channels < 3:
                    return None

                for i in range(0, len(selected_indices), batch_size):
                    end_idx = min(i + batch_size, len(selected_indices))
                    batch_indices = selected_indices[i:end_idx]
                    batch_waveforms = channel_data[:, :, batch_indices]
                    waveforms_list.append(batch_waveforms)

            phys_waveforms = np.concatenate(waveforms_list, axis=2)
            return phys_waveforms

        except Exception as e:
            return None

    # 读取前几个文件训练GMM
    training_file_count = 0
    for file_idx, (ch0_3_file, ch5_file) in enumerate(matched_pairs):
        if training_file_count >= max_files_for_training:
            break
            
        print(f"\n[训练 {training_file_count+1}/{max_files_for_training}] 处理: {os.path.basename(ch0_3_file)}")
        phys_waveforms = _process_single_file_for_training(
            ch0_3_file, ch5_file, start_datetime, end_datetime, epoch_offset
        )
        
        if phys_waveforms is not None:
            training_waveforms_list.append(phys_waveforms)
            training_file_count += 1
            print(f"  已读取 {phys_waveforms.shape[2]} 个物理事例的波形数据（用于训练）")

    if len(training_waveforms_list) == 0:
        raise RuntimeError("未能从任何文件中读取到物理事例数据用于训练GMM")

    # 合并训练数据
    print(f"\n合并 {len(training_waveforms_list)} 个训练文件的数据...")
    training_waveforms = np.concatenate(training_waveforms_list, axis=2)
    print(f"训练数据共有 {training_waveforms.shape[2]} 个物理事例")
    
    # 释放训练文件数据
    del training_waveforms_list


    # 步骤2: 使用训练数据计算 CH1 和 CH2 的最大值并做 GMM 拟合
    print("\n步骤2: 计算 CH1 和 CH2 的最大值并进行 GMM 拟合...")
    max_ch1 = training_waveforms[:, 1, :].max(axis=0).astype(np.float64)
    max_ch2 = training_waveforms[:, 2, :].max(axis=0).astype(np.float64)

    a, b, sigma = _fit_single_line_two_step(
        max_ch1,
        max_ch2,
        x_min=2000.0,
        x_max=14000.0,
    )

    mask_window = (
        (max_ch1 > 1100.0)
        & (max_ch1 < 1400.0)
        & (max_ch2 > 1000.0)
        & (max_ch2 < 2200.0)
    )
    if sigma > 0.0:
        residuals_all = max_ch2 - (a * max_ch1 + b)
        mask_band = np.abs(residuals_all) <= sigma
    else:
        mask_band = np.ones_like(max_ch1, dtype=bool)

    mask_sel = mask_window & mask_band
    x_sel = max_ch1[mask_sel]
    y_sel = max_ch2[mask_sel]

    n_sel = int(mask_sel.sum())
    print(f"  满足窗口和 ±1σ 条件的事件数: {n_sel}")
    if n_sel < 10:
        raise RuntimeError("选中事件数过少，无法稳定进行 GMM 拟合")

    sel_indices = np.where(mask_sel)[0]

    data = np.column_stack([x_sel, y_sel])

    # 进行 2 成分 GMM 拟合
    print("\n正在进行 2 成分 GMM 拟合...")
    gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=0,
    )
    gmm.fit(data)

    labels = gmm.predict(data)
    means = gmm.means_

    # 按 y 坐标从小到大排序，component 1 对应 region2（y 更大的成分）
    order = np.argsort(means[:, 1])
    region1_label = int(order[0])
    region2_label = int(order[1])  # component 1

    print("GMM 拟合完成。")
    print(f"  component 0 (区域1): {int(np.sum(labels == region1_label))} events")
    print(f"  component 1 (区域2): {int(np.sum(labels == region2_label))} events")
    
    # 释放训练数据
    del training_waveforms, max_ch1, max_ch2

    # 步骤3: 遍历所有文件，使用训练好的GMM模型筛选component 1事件，并累加到时间bin
    print("\n步骤3: 遍历所有文件，筛选component 1事件并累加到时间bin...")
    
    # 使用字典累加时间bin计数（避免将所有事件时间保存在内存中）
    # key: bin中心时间的timestamp, value: 计数
    time_bin_counts = {}
    
    processed_file_count = 0
    for file_idx, (ch0_3_file, ch5_file) in enumerate(matched_pairs):
        print(f"\n[{file_idx+1}/{len(matched_pairs)}] 处理: {os.path.basename(ch0_3_file)}")
        
        # 如果指定了日期范围，先读取文件的时间范围检查
        skip_file = False
        if start_datetime is not None or end_datetime is not None:
            try:
                file_event_times = read_hdf5_event_times(ch0_3_file, epoch_offset)
                if len(file_event_times) > 0:
                    file_start = file_event_times[0]
                    file_end = file_event_times[-1]
                    
                    if start_datetime is not None and file_end < start_datetime:
                        print(f"  跳过：文件结束时间 {file_end} 早于起始时间 {start_datetime}")
                        skip_file = True
                    elif end_datetime is not None and file_start > end_datetime:
                        print(f"  跳过：文件开始时间 {file_start} 晚于结束时间 {end_datetime}")
                        skip_file = True
            except Exception as e:
                pass  # 继续处理
        
        if skip_file:
            continue

        # 筛选物理事例（既非RT也非Inhibit）
        rt_cut = 6000.0
        batch_size = 1000

        try:
            with h5py.File(ch0_3_file, "r") as f_ch0:
                ch0_channel_data = f_ch0["channel_data"]
                _, _, ch0_num_events = ch0_channel_data.shape
                ch0_min_values = np.zeros(ch0_num_events, dtype=np.float64)
                for i in range(0, ch0_num_events, batch_size):
                    end_idx = min(i + batch_size, ch0_num_events)
                    batch_data = ch0_channel_data[:, 0, i:end_idx]
                    ch0_min_values[i:end_idx] = np.min(batch_data, axis=0)

            with h5py.File(ch5_file, "r") as f_ch5:
                ch5_channel_data = f_ch5["channel_data"]
                _, _, ch5_num_events = ch5_channel_data.shape
                ch5_max_values = np.zeros(ch5_num_events, dtype=np.float64)
                for i in range(0, ch5_num_events, batch_size):
                    end_idx = min(i + batch_size, ch5_num_events)
                    batch_data = ch5_channel_data[:, 0, i:end_idx]
                    ch5_max_values[i:end_idx] = np.max(batch_data, axis=0)

                rt_mask = ch5_max_values > rt_cut
                inhibit_mask = ch0_min_values == 0
                neither_mask = ~rt_mask & ~inhibit_mask
                selected_indices = np.where(neither_mask)[0]

            physical_count = len(selected_indices)
            if physical_count == 0:
                print(f"  跳过：未发现物理事例")
                continue

            print(f"  找到 {physical_count} 个物理事例")

            # 读取物理事例的波形数据（只读取CH1和CH2用于计算最大值）
            max_ch1_list = []
            max_ch2_list = []
            with h5py.File(ch0_3_file, "r") as f:
                channel_data = f["channel_data"]
                time_samples, num_channels, _ = channel_data.shape

                if num_channels < 3:
                    print(f"  跳过：文件只有 {num_channels} 个通道")
                    continue

                for i in range(0, len(selected_indices), batch_size):
                    end_idx = min(i + batch_size, len(selected_indices))
                    batch_indices = selected_indices[i:end_idx]
                    batch_waveforms = channel_data[:, :, batch_indices]
                    # 只计算最大值，不保存完整波形
                    max_ch1_list.append(batch_waveforms[:, 1, :].max(axis=0).astype(np.float64))
                    max_ch2_list.append(batch_waveforms[:, 2, :].max(axis=0).astype(np.float64))

            file_max_ch1 = np.concatenate(max_ch1_list)
            file_max_ch2 = np.concatenate(max_ch2_list)
            
            # 释放临时数据
            del max_ch1_list, max_ch2_list

            # 使用全局的拟合参数筛选
            file_mask_window = (
                (file_max_ch1 > 1100.0)
                & (file_max_ch1 < 1400.0)
                & (file_max_ch2 > 1000.0)
                & (file_max_ch2 < 2200.0)
            )
            if sigma > 0.0:
                file_residuals_all = file_max_ch2 - (a * file_max_ch1 + b)
                file_mask_band = np.abs(file_residuals_all) <= sigma
            else:
                file_mask_band = np.ones_like(file_max_ch1, dtype=bool)

            file_mask_sel = file_mask_window & file_mask_band
            file_sel_indices = selected_indices[file_mask_sel]
            
            if len(file_sel_indices) == 0:
                print(f"  跳过：没有满足窗口和±1σ条件的事件")
                del file_max_ch1, file_max_ch2
                continue

            # 对筛选后的事件做GMM预测
            file_data = np.column_stack([file_max_ch1[file_mask_sel], file_max_ch2[file_mask_sel]])
            file_labels = gmm.predict(file_data)
            
            # 释放数据
            del file_max_ch1, file_max_ch2, file_data

            # 找出component 1的事件
            file_comp1_mask_in_sel = file_labels == region2_label
            file_comp1_sel_indices = file_sel_indices[file_comp1_mask_in_sel]
            
            if len(file_comp1_sel_indices) == 0:
                print(f"  跳过：没有component 1事件")
                continue

            # 从文件中读取这些事件的触发时间并直接累加到时间bin
            with h5py.File(ch0_3_file, "r") as f:
                time_data = f["time_data"][:]
                file_comp1_times = time_data[file_comp1_sel_indices]
                
                # 转换为datetime
                epoch_start = datetime(1970, 1, 1)
                file_event_datetimes = []
                for time_val in file_comp1_times:
                    eventtime = time_val - epoch_offset
                    event_datetime = epoch_start + pd.Timedelta(seconds=eventtime)
                    file_event_datetimes.append(event_datetime)
                
                # 如果指定了日期范围，先筛选
                if start_datetime is not None or end_datetime is not None:
                    file_event_datetimes = pd.to_datetime(file_event_datetimes).values
                    mask = np.ones(len(file_event_datetimes), dtype=bool)
                    if start_datetime is not None:
                        mask = mask & (file_event_datetimes >= start_datetime)
                    if end_datetime is not None:
                        mask = mask & (file_event_datetimes <= end_datetime)
                    file_event_datetimes = file_event_datetimes[mask]
                else:
                    file_event_datetimes = pd.to_datetime(file_event_datetimes).values
                
                # 累加到时间bin（使用指定的time_bin_size）
                if len(file_event_datetimes) > 0:
                    df_temp = pd.DataFrame({'datetime': file_event_datetimes})
                    df_temp.set_index('datetime', inplace=True)
                    df_temp['count'] = 1
                    resampled = df_temp.resample(time_bin_size).count()
                    
                    # 累加到总计数
                    for bin_time, count in zip(resampled.index, resampled['count'].values):
                        bin_timestamp = pd.Timestamp(bin_time).value  # 转换为timestamp用于字典key
                        if bin_timestamp not in time_bin_counts:
                            time_bin_counts[bin_timestamp] = 0
                        time_bin_counts[bin_timestamp] += int(count)
                
                print(f"  找到 {len(file_comp1_sel_indices)} 个 component 1 事件（筛选后: {len(file_event_datetimes)}）")
                processed_file_count += 1

        except Exception as e:
            print(f"  错误：处理文件时出错: {e}，跳过此文件")
            import traceback
            traceback.print_exc()
            continue

    if len(time_bin_counts) == 0:
        raise RuntimeError("未能从任何文件中读取到 component 1 事件的触发时间")
    
    total_events = sum(time_bin_counts.values())
    print(f"\n总共处理了 {processed_file_count} 个文件，累加得到 {total_events} 个 component 1 事件")

    # 步骤4: 将时间bin计数转换为时间数组和计数数组
    print("\n步骤4: 整理时间bin数据...")
    
    # 将字典转换为DataFrame
    if len(time_bin_counts) > 0:
        bin_times = [pd.Timestamp(ts, unit='ns') for ts in time_bin_counts.keys()]
        bin_counts = list(time_bin_counts.values())
        
        # 创建DataFrame并排序
        df_bins = pd.DataFrame({'count': bin_counts}, index=bin_times)
        df_bins = df_bins.sort_index()
        
        # 重新resample到指定的时间bin大小（确保bin对齐）
        resampled = df_bins.resample(time_bin_size).sum()
        
        bin_centers = resampled.index.to_pydatetime()
        counts = resampled['count'].values.astype(int)
    else:
        bin_centers = np.array([], dtype='datetime64[ns]')
        counts = np.array([], dtype=int)

    total_events = int(np.sum(counts)) if len(counts) > 0 else 0
    print(f"\n总共读取到 {total_events} 个 component 1 事件（分布在 {len(bin_centers)} 个时间bin中）")
    if len(bin_centers) > 0:
        print(f"时间范围: {bin_centers[0]} 到 {bin_centers[-1]}")

    return bin_centers, counts


def calculate_count_rate(
    event_times: np.ndarray, time_bin_size: str = "1h"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算事件计数率（按时间bin分组）
    
    参数:
        event_times: event时间数组（datetime）
        time_bin_size: 时间bin大小，pandas频率字符串（如'1h', '30min', '1D'）
    
    返回:
        (bin_centers, counts): bin中心时间数组和计数数组
    """
    df = pd.DataFrame({"datetime": event_times})
    df.set_index("datetime", inplace=True)
    df["count"] = 1

    resampled = df.resample(time_bin_size).count()

    bin_centers = resampled.index.to_pydatetime()
    counts = resampled["count"].values

    return bin_centers, counts


def plot_acceleration_and_countrate(
    vibration_data_dir: str,
    hdf5_dir: Optional[str] = None,
    detector_num=2,
    time_bin_size: str = "1h",
    start_date: str = "2025-05-28",
    end_date: str = "2025-06-10",
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    figsize: Tuple[int, int] = (14, 7),
    epoch_offset: float = 2.082816000000000e+09,
    downsample_factor: int = 100,
) -> None:
    """
    绘制振动加速度数据和GMM component 1事件计数率
    
    参数:
        vibration_data_dir: 振动传感器HDF5数据目录
        hdf5_dir: HDF5文件目录（如果为None则自动获取）
        detector_num: 探测器编号，可以是单个整数或整数列表
        time_bin_size: 时间bin大小，pandas频率字符串
        start_date: 起始日期，格式 'YYYY-MM-DD'
        end_date: 终止日期，格式 'YYYY-MM-DD'
        start_time: 起始时间（可选），格式 'HH:MM:SS'
        end_time: 终止时间（可选），格式 'HH:MM:SS'
        save_path: 保存图片的路径，如果为None则不保存
        show_plot: 是否显示图片
        figsize: 图片大小 (宽度, 高度)
        epoch_offset: 时间戳偏移量
        downsample_factor: 振动数据降采样因子
    """
    print("=" * 70)
    print("振动加速度数据和GMM component 1事件计数率联合绘制")
    print("=" * 70)

    # 1. 获取 component 1 事件计数率（从所有HDF5文件，已按时间bin累加，避免OOM）
    print("\n步骤1: 获取GMM component 1事件计数率（从所有HDF5文件，增量累加模式）...")
    bin_centers, counts = get_component1_event_times(
        hdf5_dir, 
        start_date=start_date,
        end_date=end_date,
        start_time=start_time,
        end_time=end_time,
        epoch_offset=epoch_offset,
        time_bin_size=time_bin_size,  # 直接使用指定的时间bin大小
    )

    if len(bin_centers) == 0 or np.sum(counts) == 0:
        raise ValueError("未找到任何 component 1 事件")

    print(f"生成了 {len(bin_centers)} 个时间bin")

    # 2. 读取振动加速度数据
    print("\n步骤2: 读取振动加速度数据...")
    vibration_data = select_by_date_range_vibration(
        vibration_data_dir,
        detector_num=detector_num,
        start_date=start_date,
        end_date=end_date,
        start_time=start_time,
        end_time=end_time,
        downsample_factor=downsample_factor,
    )

    if not vibration_data or "datetime" not in vibration_data:
        raise ValueError("未能读取到振动加速度数据")

    vibration_datetime = vibration_data["datetime"]
    # 选择加速度分量（优先选择 z 分量，如果没有则选择第一个可用的）
    accel_columns = ["Acceleration_z", "Acceleration_y", "Acceleration_x"]
    accel_data = None
    accel_name = None
    for col in accel_columns:
        if col in vibration_data:
            accel_data = vibration_data[col]
            accel_name = col
            break

    if accel_data is None:
        raise ValueError("未能找到加速度数据列")

    # 过滤NaN值
    valid_mask = ~np.isnan(accel_data)
    vibration_datetime_valid = vibration_datetime[valid_mask]
    accel_data_valid = accel_data[valid_mask]

    if len(vibration_datetime_valid) == 0:
        raise ValueError("振动加速度数据中没有有效数据")

    print(f"读取到 {len(vibration_datetime_valid)} 个有效加速度数据点")

    # 3. 绘图
    print("\n步骤3: 绘制联合图...")

    # 设置matplotlib参数
    plt.rcParams.update(
        {
            "font.size": 11,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "axes.linewidth": 1.2,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.minor.width": 0.8,
            "ytick.minor.width": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.fontsize": 10,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "figure.dpi": 100,
        }
    )

    # 创建图形和左y轴（计数率）
    fig, ax1 = plt.subplots(figsize=figsize)

    # 绘制计数率（柱状图）
    ax1.bar(
        bin_centers,
        counts,
        width=pd.Timedelta(time_bin_size) * 0.8,
        color="#2E86AB",
        alpha=0.7,
        label="Component 1 Event Count",
    )
    ax1.set_xlabel("Time", fontsize=13, fontweight="normal")
    ax1.set_ylabel(
        "Component 1 Event Count", fontsize=13, fontweight="normal", color="#2E86AB"
    )
    ax1.tick_params(axis="y", labelcolor="#2E86AB")

    # 创建右y轴（加速度）
    ax2 = ax1.twinx()
    ax2.plot(
        vibration_datetime_valid,
        accel_data_valid,
        linewidth=1.5,
        alpha=0.85,
        color="#C73E1D",
        label=f"Acceleration ({accel_name})",
    )
    ax2.set_ylabel(
        f"Acceleration ({accel_name}) (g)",
        fontsize=13,
        fontweight="normal",
        color="#C73E1D",
    )
    ax2.tick_params(axis="y", labelcolor="#C73E1D")

    # 格式化x轴日期
    bin_centers_dt64 = pd.to_datetime(bin_centers).values
    vibration_datetime_dt64 = pd.to_datetime(vibration_datetime_valid).values

    all_datetime = np.concatenate([bin_centers_dt64, vibration_datetime_dt64])
    all_datetime = np.sort(all_datetime)

    if len(all_datetime) > 1:
        time_span = all_datetime[-1] - all_datetime[0]

        if hasattr(time_span, "days"):
            days = time_span.days
        else:
            days = time_span.astype("timedelta64[D]").astype(int)

        if days > 30:
            date_format = "%m-%d %H:%M"
            locator = mdates.HourLocator(byhour=[6, 12, 18])
            minor_locator = mdates.HourLocator(interval=3)
        elif days > 1:
            date_format = "%m-%d %H:%M"
            locator = mdates.HourLocator(byhour=[6, 12, 18])
            minor_locator = mdates.HourLocator(interval=3)
        else:
            date_format = "%H:%M"
            locator = mdates.HourLocator(byhour=[0, 6, 12, 18])
            minor_locator = mdates.HourLocator(interval=3)

        ax1.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_minor_locator(minor_locator)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # 设置y轴格式
    ax1.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
    ax1.yaxis.get_major_formatter().set_scientific(False)
    ax2.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
    ax2.yaxis.get_major_formatter().set_scientific(False)

    # 网格
    ax1.grid(True, which="major", linestyle="-", linewidth=0.7, alpha=0.3, color="gray")
    ax1.grid(True, which="minor", linestyle="--", linewidth=0.5, alpha=0.2, color="gray")

    # 坐标轴边框
    ax1.spines["top"].set_visible(True)
    ax1.spines["right"].set_visible(True)
    ax1.spines["top"].set_color("gray")
    ax1.spines["right"].set_color("gray")

    # 计算统计信息
    count_mean = np.mean(counts)
    count_std = np.std(counts)
    count_min = np.min(counts)
    count_max = np.max(counts)
    total_events = np.sum(counts)

    accel_mean = np.mean(accel_data_valid)
    accel_std = np.std(accel_data_valid)
    accel_min = np.min(accel_data_valid)
    accel_max = np.max(accel_data_valid)

    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
        framealpha=0.9,
        edgecolor="gray",
        frameon=True,
        fancybox=False,
        shadow=False,
    )

    # 添加统计信息框
    stats_text = (
        f"Component 1 Events:\n"
        f"  Total = {total_events}\n"
        f"  Mean = {count_mean:.1f}\n"
        f"  Std = {count_std:.1f}\n"
        f"  Min = {count_min:.0f}\n"
        f"  Max = {count_max:.0f}\n"
        f"\nAcceleration ({accel_name}):\n"
        f"  Mean = {accel_mean:.4f} g\n"
        f"  Std = {accel_std:.4f} g\n"
        f"  Min = {accel_min:.4f} g\n"
        f"  Max = {accel_max:.4f} g"
    )

    ax1.text(
        0.98,
        0.98,
        stats_text,
        transform=ax1.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="gray",
            alpha=0.8,
            linewidth=0.8,
        ),
        fontsize=9,
        family="monospace",
    )

    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"\n图片已保存至: {save_path}")

    # 显示图片
    if show_plot:
        plt.show()
    else:
        plt.close()


# 示例使用
if __name__ == "__main__":
    # 获取项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)  # python/data
    grandparent_dir = os.path.dirname(parent_dir)  # python
    project_root = os.path.dirname(grandparent_dir)  # 项目根目录

    # 设置数据路径
    vibration_data_dir = os.path.join(project_root, "data", "vibration", "hdf5")
    hdf5_dir = os.path.join(project_root, "data", "hdf5", "raw_pulse")

    print("=" * 70)
    print("振动加速度数据和GMM component 1事件计数率联合绘制")
    print("=" * 70)

    try:
        # 绘制
        plot_acceleration_and_countrate(
            vibration_data_dir=vibration_data_dir,
            hdf5_dir=hdf5_dir,
            detector_num=2,  # 可以改为 [1,2,3,4,5] 读取多个传感器
            time_bin_size="1h",  # 1小时一个bin
            start_date="2025-05-28",
            end_date="2025-06-10",
            show_plot=True,
            downsample_factor=100,  # 降采样，每隔100个点读取1个
        )

    except Exception as e:
        print(f"错误: {e}")
        import traceback

        traceback.print_exc()
