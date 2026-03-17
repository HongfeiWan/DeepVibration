#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对 CH0-3 原始脉冲 h5 文件的 CH2 / CH3 做全核并行拟合。

功能特点：
- 批处理模式：
  - 自动遍历 data/hdf5/raw_pulse/CH0-3 目录下的所有 h5 文件，
    依次对每一个文件执行 CH2/CH3 的并行拟合计算；
- 单文件模式：
  - 也可以手动指定某一个 CH0-3 h5 文件，仅对该文件进行处理；
- 对每个文件：
  - 读取 channel_data 中 CH2 / CH3 的所有事件波形；
  - 使用全部 CPU 内核 (max_workers=os.cpu_count()) 在“事件维度”并行：
    * utils.fit._compute_fast_fit_params
    * utils.frequency._compute_fast_highfreq_energy_ratio（内部是基于 FFT 的能量比计算）；
  - 将结果分别写入 data/hdf5/raw_pulse/CH2_parameters / CH3_parameters 目录中，
    文件名与输入 h5 保持一致。

推荐使用方式：
1) 先用 python/data/preprocessor.py 生成原始 CH0-3 h5；
2) 再运行本脚本批量处理全部文件：
   python fit_ch2_ch3_parallel.py
   （或指定单个文件：python fit_ch2_ch3_parallel.py /path/to/CH0-3/xxxx_processed.h5）

同一时刻只跑一个这样的脚本，即可在拟合阶段让每个 h5 文件都独占整机 CPU。
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple

import h5py
import numpy as np


# 目录推断：本脚本与 preprocessor.py 同目录，位于 .../python/data
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)          # .../python
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

project_root = os.path.dirname(parent_dir)         # 项目根目录

from utils.fit import _compute_fast_fit_params_with_npoints  # type: ignore  # noqa: E402
from utils.frequency import (  # type: ignore  # noqa: E402
    _compute_fast_highfreq_energy_ratio,
    _compute_spectral_centroid_mhz,
)


ch0_3_dir = os.path.join(project_root, "data", "hdf5", "raw_pulse", "CH0-3")
ch2parameters_save_path = os.path.join(project_root, "data", "hdf5", "raw_pulse", "CH2_parameters")
ch3parameters_save_path = os.path.join(project_root, "data", "hdf5", "raw_pulse", "CH3_parameters")


# -----------------------------------------------------------------------------
# 独立的 cut 函数：输入为对应参数数组，输出为 bool 掩码
# -----------------------------------------------------------------------------

def cut_ch0_min_positive(ch0_min: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    条件：ch0_min > threshold（排除抑制信号）。
    输入: ch0_min 数组
    输出: bool 掩码
    """
    return ch0_min > threshold

def cut_ch0_max_saturation(max_ch0: np.ndarray, max_val: float = 16382.0) -> np.ndarray:
    """
    条件：max_ch0 <= max_val（排除饱和事例）。
    输入: max_ch0 数组
    输出: bool 掩码
    """
    return max_ch0 <= max_val

def cut_ch5_self_trigger(max_ch5: np.ndarray, rt_threshold: float = 6000.0) -> np.ndarray:
    """
    条件：max_ch5 <= rt_threshold（排除随机触发）。
    输入: max_ch5 数组
    输出: bool 掩码
    """
    return max_ch5 <= rt_threshold


def _process_one_event(waveform: np.ndarray) -> Tuple[float, float, float, float, float, float, float, int]:
    """
    对单个事件波形做 tanh 拟合和高频能量占比计算。
    返回: (tanh_p0, tanh_p1, tanh_p2, tanh_p3, tanh_rms, highfreq_energy_ratio, spectral_centroid_mhz, n_fit_points)
    """
    wf = np.asarray(waveform, dtype=np.float32)
    try:
        params, n_points = _compute_fast_fit_params_with_npoints(wf)
        highfreq_ratio = _compute_fast_highfreq_energy_ratio(wf)
        spectral_centroid_mhz = _compute_spectral_centroid_mhz(wf, sampling_interval_ns=4.0)
    except Exception:
        # 对拟合失败的事件，统一返回异常值标记
        bad = np.float32(1e6)
        return (bad, bad, bad, bad, bad, bad, bad, 0)
    return (
        float(params["tanh_p0"]),
        float(params["tanh_p1"]),
        float(params["tanh_p2"]),
        float(params["tanh_p3"]),
        float(params["tanh_rms"]),
        float(highfreq_ratio),
        float(spectral_centroid_mhz),
        n_points,
    )

def _fit_one_channel_for_file(
    ch0_3_file: str,
    channel_idx: int,
    out_dir: str,) -> None:
    """
    对给定 CH0-3 h5 文件中的指定通道 (channel_idx=2 或 3) 做全核事件并行拟合。
    将结果写入 out_dir 下、与输入 h5 同名的 h5 文件。
    """
    base_name = os.path.basename(ch0_3_file)
    out_path = os.path.join(out_dir, base_name)

    if os.path.exists(out_path):
        print(f"{out_path} 已存在，跳过通道 {channel_idx} 的拟合。")
        return

    os.makedirs(out_dir, exist_ok=True)

    with h5py.File(ch0_3_file, "r") as f:
        channel_data = f["channel_data"]
        time_samples, num_channels, num_events = channel_data.shape

        if channel_idx >= num_channels:
            raise ValueError(
                f"channel_idx={channel_idx} 超过通道数 {num_channels}，"
                f"文件: {ch0_3_file}"
            )

        print(
            f"开始对文件 {base_name} 的通道 {channel_idx} 拟合，"
            f"时间点数={time_samples}，事件数={num_events}"
        )

        # 读取为 (time_samples, num_events)，在事件维度并行处理
        waveforms = channel_data[:, channel_idx, :].astype(np.float32)

    n_events = waveforms.shape[1]

    # 统一的异常值标记
    bad_val = np.float32(1e6)

    # 结果数组初始化为异常值，后续仅对“通过基础 cut 且拟合成功”的事件覆盖
    tanh_p0 = np.full(n_events, bad_val, dtype=np.float32)
    tanh_p1 = np.full(n_events, bad_val, dtype=np.float32)
    tanh_p2 = np.full(n_events, bad_val, dtype=np.float32)
    tanh_p3 = np.full(n_events, bad_val, dtype=np.float32)
    tanh_rms = np.full(n_events, bad_val, dtype=np.float32)
    highfreq_ratio = np.full(n_events, bad_val, dtype=np.float32)
    spectral_centroid_mhz = np.full(n_events, bad_val, dtype=np.float32)
    # 拟合使用的点数，对所有事件初始化为 0；仅对实际拟合的事件填入正数
    n_fit_points = np.zeros(n_events, dtype=np.int32)

    # 同时读取 CH0 / CH5 的关键参数，用于基础物理 cut
    ch0_param_dir = os.path.join(project_root, "data", "hdf5", "raw_pulse", "CH0_parameters")
    ch5_param_dir = os.path.join(project_root, "data", "hdf5", "raw_pulse", "CH5_parameters")
    ch0_param_path = os.path.join(ch0_param_dir, base_name)
    ch5_param_path = os.path.join(ch5_param_dir, base_name)

    # 必须找到对应的 CH0/CH5 参数文件及所需数据集，否则认为数据不完整，直接报错
    if not os.path.exists(ch0_param_path):
        raise FileNotFoundError(f"缺少 CH0 参数文件: {ch0_param_path}")
    if not os.path.exists(ch5_param_path):
        raise FileNotFoundError(f"缺少 CH5 参数文件: {ch5_param_path}")

    with h5py.File(ch0_param_path, "r") as f_ch0:
        if "max_ch0" not in f_ch0 or "ch0_min" not in f_ch0:
            raise KeyError(
                f"CH0 参数文件 {ch0_param_path} 中缺少 max_ch0 或 ch0_min 数据集"
            )
        max_ch0 = np.asarray(f_ch0["max_ch0"][...], dtype=np.float32)
        ch0_min = np.asarray(f_ch0["ch0_min"][...], dtype=np.float32)

    with h5py.File(ch5_param_path, "r") as f_ch5:
        if "max_ch5" not in f_ch5:
            raise KeyError(
                f"CH5 参数文件 {ch5_param_path} 中缺少 max_ch5 数据集"
            )
        max_ch5 = np.asarray(f_ch5["max_ch5"][...], dtype=np.float32)

    # 计算基础物理 cut 掩码：
    # - ch0_min > 0
    # - max_ch0 <= 16382
    # - max_ch5 <= 6000
    #
    # 仅对通过上述三个 cut 的事件提交拟合任务；
    # 未通过的事件保持初始化的异常值（bad_val），完全不进入拟合流程。
    if max_ch0 is not None and ch0_min is not None and max_ch5 is not None:
        mask_valid = np.zeros(n_events, dtype=bool)
        n_cut = min(len(max_ch0), len(ch0_min), len(max_ch5), n_events)
        mask_valid[:n_cut] = (
            cut_ch0_min_positive(ch0_min[:n_cut])
            & cut_ch0_max_saturation(max_ch0[:n_cut])
            & cut_ch5_self_trigger(max_ch5[:n_cut])
        )
    else:
        # 若无法计算 cut，则认为全部事件通过（mask 全 True）
        mask_valid = np.ones(n_events, dtype=bool)

    # 基础 cut 通过的事件数量统计
    n_kept = int(mask_valid.sum())
    print(f"通道 {channel_idx}: 基础 cut 通过事件数 = {n_kept} / {n_events}")

    max_workers = os.cpu_count() or 1
    print(f"使用 {max_workers} 个 CPU 核并行拟合通道 {channel_idx}。")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for ev in range(n_events):
            # 仅对通过基础 cut 的事件进行拟合；
            # 未通过的事件保持初始化的异常值（bad_val）
            if not mask_valid[ev]:
                continue
            wf_ev = waveforms[:, ev]
            fut = executor.submit(_process_one_event, wf_ev)
            futures[fut] = ev

        # 简单进度统计：以“实际提交拟合的事件数”作为分母
        total_to_fit = len(futures)
        if total_to_fit == 0:
            print(f"通道 {channel_idx}: 无事件通过基础 cut，跳过拟合。")
        else:
            finished = 0
            for fut in as_completed(futures):
                ev = futures[fut]
                p0, p1, p2, p3, rms, hfr, sc, n_pts = fut.result()
                tanh_p0[ev] = p0
                tanh_p1[ev] = p1
                tanh_p2[ev] = p2
                tanh_p3[ev] = p3
                tanh_rms[ev] = rms
                highfreq_ratio[ev] = hfr
                spectral_centroid_mhz[ev] = sc
                n_fit_points[ev] = int(n_pts)

                finished += 1

            # 仅在完成时打印一次进度，节约输出和计算资源
            print(
                f"通道 {channel_idx} 拟合进度: "
                f"{finished}/{total_to_fit} (100.0%)"
            )

    with h5py.File(out_path, "w") as f_out:
        f_out.create_dataset("tanh_p0", data=tanh_p0)
        f_out.create_dataset("tanh_p1", data=tanh_p1)
        f_out.create_dataset("tanh_p2", data=tanh_p2)
        f_out.create_dataset("tanh_p3", data=tanh_p3)
        f_out.create_dataset("tanh_rms", data=tanh_rms)
        f_out.create_dataset("highfreq_energy_ratio", data=highfreq_ratio)
        f_out.create_dataset("spectral_centroid_mhz", data=spectral_centroid_mhz)
        f_out.create_dataset("n_fit_points", data=n_fit_points)
        f_out.attrs["source_file"] = str(os.path.abspath(ch0_3_file))
        f_out.attrs["channel_index"] = int(channel_idx)
        f_out.attrs["description"] = (
            "Per-event fast tanh-fit parameters (p0, p1, p2, p3, rms), "
            "high-frequency energy ratio (>0.2 MHz), and spectral centroid (MHz)."
        )

    print(f"通道 {channel_idx} 拟合完成，结果已写入 {out_path}")

def fit_ch2_ch3_for_file(ch0_3_file: str) -> None:
    """
    对单个 CH0-3 h5 文件的 CH2 (index=2) 和 CH3 (index=3) 做全核事件并行拟合。
    """
    if not os.path.exists(ch0_3_file):
        raise FileNotFoundError(f"未找到 CH0-3 文件: {ch0_3_file}")

    # 先拟合 CH2，再拟合 CH3；同一时刻只有一个通道在用满 CPU。
    _fit_one_channel_for_file(
        ch0_3_file=ch0_3_file,
        channel_idx=2,
        out_dir=ch2parameters_save_path,
    )
    _fit_one_channel_for_file(
        ch0_3_file=ch0_3_file,
        channel_idx=3,
        out_dir=ch3parameters_save_path,
    )

def _backfill_spectral_centroid_for_param_dir(param_dir: str, channel_idx: int) -> None:
    """
    为已有的 CH2/CH3 参数文件补写 spectral_centroid_mhz：
        - 扫描 param_dir 下的所有 h5 文件；
        - 若已存在 "spectral_centroid_mhz" 数据集，则跳过该文件；
        - 若不存在，则：
            * 通过 attrs["source_file"] 找到对应的 CH0-3 源文件；
            * 读取该通道的所有波形；
            * 使用全部 CPU 内核并行计算每个事件的频谱质心（MHz）；
            * 将结果追加写入参数 h5 文件中。
    """
    if not os.path.isdir(param_dir):
        print(f"[补写 spectral_centroid_mhz] 目录不存在，跳过: {param_dir}")
        return

    files = [
        os.path.join(param_dir, name)
        for name in sorted(os.listdir(param_dir))
        if name.lower().endswith(".h5")
    ]
    if not files:
        print(f"[补写 spectral_centroid_mhz] 目录 {param_dir} 下未找到任何 h5 文件。")
        return

    print(f"[补写 spectral_centroid_mhz] 在 {param_dir} 中找到 {len(files)} 个参数文件，开始检查。")

    for param_path in files:
        with h5py.File(param_path, "r") as f_param:
            if "spectral_centroid_mhz" in f_param:
                continue
            source_file = f_param.attrs.get("source_file", None)
            ch_idx_attr = f_param.attrs.get("channel_index", None)
            # 若缺少必要的属性，则无法恢复，直接跳过并报警
            if source_file is None or ch_idx_attr is None:
                print(f"[补写 spectral_centroid_mhz] 文件缺少 source_file/channel_index 属性，跳过: {param_path}")
                continue
            ch_idx_attr = int(ch_idx_attr)
            if ch_idx_attr != channel_idx:
                print(
                    f"[补写 spectral_centroid_mhz] 文件 {param_path} 中 channel_index={ch_idx_attr} "
                    f"与期望通道 {channel_idx} 不一致，跳过。"
                )
                continue
            try:
                source_file_str = source_file.decode("utf-8") if isinstance(source_file, bytes) else str(source_file)
            except Exception:
                source_file_str = str(source_file)

        if not os.path.exists(source_file_str):
            print(f"[补写 spectral_centroid_mhz] 找不到源 CH0-3 文件，跳过: {source_file_str}")
            continue

        print(f"[补写 spectral_centroid_mhz] 处理参数文件: {os.path.basename(param_path)} (通道 {channel_idx})")

        # 从源 CH0-3 文件读取该通道的所有事件波形
        with h5py.File(source_file_str, "r") as f_ch:
            ch_data = f_ch["channel_data"]
            time_samples, num_channels, num_events = ch_data.shape
            if channel_idx >= num_channels:
                print(
                    f"[补写 spectral_centroid_mhz] 源文件 {source_file_str} 通道数不足，"
                    f"channel_idx={channel_idx}, num_channels={num_channels}，跳过。"
                )
                continue
            waveforms = ch_data[:, channel_idx, :].astype(np.float32)

        # 依据现有参数文件中某个数据集的长度来确定事件数（以防与 CH0-3 文件不一致）
        with h5py.File(param_path, "r") as f_param:
            if "tanh_p0" not in f_param:
                print(f"[补写 spectral_centroid_mhz] 参数文件 {param_path} 中缺少 tanh_p0 数据集，跳过。")
                continue
            n_param_events = f_param["tanh_p0"].shape[0]

        n_events = min(waveforms.shape[1], n_param_events)
        if n_events == 0:
            print(f"[补写 spectral_centroid_mhz] 文件 {param_path} 无事件可处理，跳过。")
            continue

        sc_array = np.zeros(n_events, dtype=np.float32)
        max_workers = os.cpu_count() or 1
        print(
            f"[补写 spectral_centroid_mhz] 使用 {max_workers} 个 CPU 核，"
            f"为 {os.path.basename(param_path)} 计算 {n_events} 个事件的频谱质心。"
        )

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for ev in range(n_events):
                wf_ev = waveforms[:, ev]
                fut = executor.submit(_compute_spectral_centroid_mhz, wf_ev, 4.0)
                futures[fut] = ev

            finished = 0
            for fut in as_completed(futures):
                ev = futures[fut]
                sc_array[ev] = np.float32(fut.result())
                finished += 1

        # 将结果追加写入参数文件
        with h5py.File(param_path, "a") as f_param:
            if "spectral_centroid_mhz" in f_param:
                print(f"[补写 spectral_centroid_mhz] 数据集已存在，跳过写入: {param_path}")
            else:
                f_param.create_dataset("spectral_centroid_mhz", data=sc_array)
                print(f"[补写 spectral_centroid_mhz] 写入完成: {param_path}")

def _backfill_n_fit_points_for_param_dir(param_dir: str, channel_idx: int) -> None:
    """
    为已有的 CH2/CH3 参数文件补写 n_fit_points：
        - 扫描 param_dir 下的所有 h5 文件；
        - 若已存在 "n_fit_points" 数据集，则跳过该文件；
        - 若不存在，则：
            * 通过 attrs["source_file"] 找到对应的 CH0-3 源文件；
            * 读取该通道的所有波形；
            * 使用全部 CPU 内核并行调用 _compute_fast_fit_params_with_npoints，
              重新计算每个事件“实际用于拟合的点数”；
            * 将结果追加写入参数 h5 文件中。

    注意：这是对旧参数文件的“重算式”补写，开销与重新拟合一遍相当，但只做一次。
    """
    if not os.path.isdir(param_dir):
        print(f"[补写 n_fit_points] 目录不存在，跳过: {param_dir}")
        return

    files = [
        os.path.join(param_dir, name)
        for name in sorted(os.listdir(param_dir))
        if name.lower().endswith(".h5")
    ]
    if not files:
        print(f"[补写 n_fit_points] 目录 {param_dir} 下未找到任何 h5 文件。")
        return

    print(f"[补写 n_fit_points] 在 {param_dir} 中找到 {len(files)} 个参数文件，开始检查。")

    for param_path in files:
        with h5py.File(param_path, "r") as f_param:
            if "n_fit_points" in f_param:
                continue
            source_file = f_param.attrs.get("source_file", None)
            ch_idx_attr = f_param.attrs.get("channel_index", None)
            if source_file is None or ch_idx_attr is None:
                print(f"[补写 n_fit_points] 文件缺少 source_file/channel_index 属性，跳过: {param_path}")
                continue
            ch_idx_attr = int(ch_idx_attr)
            if ch_idx_attr != channel_idx:
                print(
                    f"[补写 n_fit_points] 文件 {param_path} 中 channel_index={ch_idx_attr} "
                    f"与期望通道 {channel_idx} 不一致，跳过。"
                )
                continue
            try:
                source_file_str = source_file.decode("utf-8") if isinstance(source_file, bytes) else str(source_file)
            except Exception:
                source_file_str = str(source_file)

        if not os.path.exists(source_file_str):
            print(f"[补写 n_fit_points] 找不到源 CH0-3 文件，跳过: {source_file_str}")
            continue

        # 从源 CH0-3 文件读取该通道的所有事件波形
        with h5py.File(source_file_str, "r") as f_ch:
            ch_data = f_ch["channel_data"]
            time_samples, num_channels, num_events = ch_data.shape
            if channel_idx >= num_channels:
                print(
                    f"[补写 n_fit_points] 源文件 {source_file_str} 通道数不足，"
                    f"channel_idx={channel_idx}, num_channels={num_channels}，跳过。"
                )
                continue

        # 依据现有参数文件中某个数据集的长度来确定事件数
        with h5py.File(param_path, "r") as f_param:
            if "tanh_p0" not in f_param:
                print(f"[补写 n_fit_points] 参数文件 {param_path} 中缺少 tanh_p0 数据集，跳过。")
                continue
            n_param_events = f_param["tanh_p0"].shape[0]

        n_events = min(num_events, n_param_events)
        if n_events == 0:
            print(f"[补写 n_fit_points] 文件 {param_path} 无事件可处理，跳过。")
            continue

        n_fit_points = np.zeros(n_events, dtype=np.int32)
        max_workers = os.cpu_count() or 1
        print(
            f"[补写 n_fit_points] 使用 {max_workers} 个 CPU 核，"
            f"为 {os.path.basename(param_path)} 重新计算 {n_events} 个事件的拟合点数。"
        )

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for ev in range(n_events):
                wf_ev = ch_data[:, channel_idx, ev].astype(np.float32)
                fut = executor.submit(_compute_fast_fit_params_with_npoints, wf_ev)
                futures[fut] = ev

            for fut in as_completed(futures):
                ev = futures[fut]
                _, n_pts = fut.result()
                n_fit_points[ev] = int(n_pts)

        with h5py.File(param_path, "a") as f_param:
            if "n_fit_points" in f_param:
                print(f"[补写 n_fit_points] 数据集已存在，跳过写入: {param_path}")
            else:
                # 若参数文件事件数多于源 CH0-3 事件数，剩余的事件点数保持 0
                if n_param_events > n_events:
                    pad = np.zeros(n_param_events - n_events, dtype=np.int32)
                    data_to_write = np.concatenate([n_fit_points, pad])
                else:
                    data_to_write = n_fit_points
                f_param.create_dataset("n_fit_points", data=data_to_write)
                print(f"[补写 n_fit_points] 写入完成: {param_path}")



def main() -> None:
    """
    命令行入口：
        1) 批量处理 CH0-3 目录下所有 h5：
           python fit_ch2_ch3_parallel.py
        2) 仅处理指定的某一个 h5：
           python fit_ch2_ch3_parallel.py /path/to/CH0-3/xxx_processed.h5
    """
    if len(sys.argv) == 1:
        # 批处理模式：遍历 CH0-3 目录下所有 h5 文件
        if not os.path.isdir(ch0_3_dir):
            raise FileNotFoundError(f"未找到 CH0-3 目录: {ch0_3_dir}")

        files = [
            os.path.join(ch0_3_dir, name)
            for name in sorted(os.listdir(ch0_3_dir))
            if name.lower().endswith(".h5")
        ]
        if not files:
            print(f"目录 {ch0_3_dir} 下未找到任何 h5 文件。")
            return

        total = len(files)
        print(f"在 {ch0_3_dir} 中找到 {total} 个 CH0-3 h5 文件，将依次进行 CH2/CH3 拟合。")

        for idx, ch0_3_file in enumerate(files, 1):
            print("=" * 60)
            print(f"[{idx}/{total}] 开始处理文件: {os.path.basename(ch0_3_file)}")
            fit_ch2_ch3_for_file(ch0_3_file)

        print("=" * 60)
        print("所有 CH0-3 文件的 CH2/CH3 拟合处理完成。")

        # 额外一步：为已有的 CH2/CH3 参数文件补写 spectral_centroid_mhz（若缺失）
        print("=" * 60)
        print("开始为 CH2_parameters 补写 spectral_centroid_mhz（如有缺失）。")
        _backfill_spectral_centroid_for_param_dir(ch2parameters_save_path, channel_idx=2)
        print("开始为 CH3_parameters 补写 spectral_centroid_mhz（如有缺失）。")
        _backfill_spectral_centroid_for_param_dir(ch3parameters_save_path, channel_idx=3)
        print("开始为 CH2_parameters 补写 n_fit_points（如有缺失）。")
        _backfill_n_fit_points_for_param_dir(ch2parameters_save_path, channel_idx=2)
        print("开始为 CH3_parameters 补写 n_fit_points（如有缺失）。")
        _backfill_n_fit_points_for_param_dir(ch3parameters_save_path, channel_idx=3)

        print("=" * 60)
        print("所有补写 spectral_centroid_mhz 的任务完成。")
        return

    if len(sys.argv) == 2:
        # 单文件模式
        ch0_3_file = sys.argv[1]
        fit_ch2_ch3_for_file(ch0_3_file)
        return

    print(
        "用法:\n"
        "  1) 批处理 CH0-3 目录下所有 h5：\n"
        "     python fit_ch2_ch3_parallel.py\n"
        "  2) 仅处理指定的某一个 h5：\n"
        "     python fit_ch2_ch3_parallel.py /path/to/CH0-3/xxx_processed.h5"
    )
    sys.exit(1)

if __name__ == "__main__":
    main()

