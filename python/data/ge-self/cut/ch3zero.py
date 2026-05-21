#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 CH3_parameters 中筛选 min_ch3 == 0 的事件，并绘制对应事件的 CH0/CH3 波形。

图像规范：
- 同一张 figure 中叠加 CH0 与 CH3；
- 左侧 y 轴为 CH0，右侧 y 轴为 CH3；
- x 轴为采样点索引（可切换为 us）。
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np


# 路径推断：
# __file__ = .../python/data/ge-self/cut/ch3zero.py
# current_dir = .../python/data/ge-self/cut
# data_dir = .../python/data
# python_dir = .../python
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.dirname(os.path.dirname(current_dir))
python_dir = os.path.dirname(data_dir)
project_root = os.path.dirname(python_dir)

ch3_param_dir_default = os.path.join(project_root, "data", "hdf5", "raw_pulse", "CH3_parameters")
ch0_3_dir_default = os.path.join(project_root, "data", "hdf5", "raw_pulse", "CH0-3")
ch0_param_dir_default = os.path.join(project_root, "data", "hdf5", "raw_pulse", "CH0_parameters")
ch1_param_dir_default = os.path.join(project_root, "data", "hdf5", "raw_pulse", "CH1_parameters")
ch4_param_dir_default = os.path.join(project_root, "data", "hdf5", "raw_pulse", "CH4_parameters")
ch5_param_dir_default = os.path.join(project_root, "data", "hdf5", "raw_pulse", "CH5_parameters")


def cut_ch0_min_positive(ch0_min: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    return ch0_min > threshold


def cut_ch0_max_saturation(max_ch0: np.ndarray, max_val: float = 16382.0) -> np.ndarray:
    return max_ch0 <= max_val


def cut_ch5_self_trigger(max_ch5: np.ndarray, rt_threshold: float = 6000.0) -> np.ndarray:
    return max_ch5 <= rt_threshold


def cut_pedestal_3sigma(
    ch0_ped_mean: np.ndarray,
    ch1_ped_mean: np.ndarray,
    max_ch5: np.ndarray,
    rt_threshold: float = 6000.0,
    n_sigma: float = 3.0,
    min_rt_events: int = 10,
) -> np.ndarray:
    n = ch0_ped_mean.shape[0]
    mask = np.ones(n, dtype=bool)
    rt_mask = max_ch5 > rt_threshold

    ch0_ped_rt = ch0_ped_mean[rt_mask]
    if ch0_ped_rt.size >= min_rt_events:
        ped_mu0 = float(ch0_ped_rt.mean())
        ped_sigma0 = float(ch0_ped_rt.std(ddof=1))
        if ped_sigma0 > 0.0:
            mask = mask & (np.abs(ch0_ped_mean - ped_mu0) <= n_sigma * ped_sigma0)

    ch1_ped_rt = ch1_ped_mean[rt_mask]
    if ch1_ped_rt.size >= min_rt_events:
        ped_mu1 = float(ch1_ped_rt.mean())
        ped_sigma1 = float(ch1_ped_rt.std(ddof=1))
        if ped_sigma1 > 0.0:
            mask = mask & (np.abs(ch1_ped_mean - ped_mu1) <= n_sigma * ped_sigma1)

    return mask


def cut_act(
    max_ch4: np.ndarray,
    tmax_ch4: np.ndarray,
    trigger_threshold: float = 7060.0,
    t_ge_us: float = 40.0,
    sampling_interval_ns: float = 4.0,
    dt_min_us: float = 1.0,
    dt_max_us: float = 16.0,
) -> np.ndarray:
    n = max_ch4.shape[0]
    tmax_ch4 = np.asarray(tmax_ch4, dtype=np.float64)[:n]
    max_ch4 = np.asarray(max_ch4, dtype=np.float64)[:n]
    nai_ok = max_ch4 >= trigger_threshold
    t_ch4_us = tmax_ch4 * sampling_interval_ns * 1e-3
    delta_t_us = t_ge_us - t_ch4_us
    act_mask = (delta_t_us < dt_min_us) | (delta_t_us > dt_max_us)
    return (~nai_ok) | (nai_ok & act_mask)


def cut_mincut(
    ch0_min: np.ndarray,
    ch1_min: np.ndarray,
    max_ch4: np.ndarray,
    tmax_ch4: np.ndarray,
    n_sigma: float = 3.0,
    min_fit_events: int = 10,
    trigger_threshold: float = 7060.0,
    t_ge_us: float = 40.0,
    sampling_interval_ns: float = 4.0,
    dt_min_us: float = 1.0,
    dt_max_us: float = 16.0,
) -> np.ndarray:
    n = ch0_min.shape[0]
    mask = np.ones(n, dtype=bool)
    fit_mask = cut_act(max_ch4, tmax_ch4, trigger_threshold, t_ge_us, sampling_interval_ns, dt_min_us, dt_max_us)

    ch0_min_fit = ch0_min[fit_mask]
    if ch0_min_fit.size >= min_fit_events:
        mu0 = float(ch0_min_fit.mean())
        sigma0 = float(ch0_min_fit.std(ddof=1))
        if sigma0 > 0.0:
            mask = mask & (np.abs(ch0_min - mu0) <= n_sigma * sigma0)

    ch1_min_fit = ch1_min[fit_mask]
    if ch1_min_fit.size >= min_fit_events:
        mu1 = float(ch1_min_fit.mean())
        sigma1 = float(ch1_min_fit.std(ddof=1))
        if sigma1 > 0.0:
            mask = mask & (np.abs(ch1_min - mu1) <= n_sigma * sigma1)

    return mask


def _resolve_source_file(param_path: str, source_file_attr: object, ch0_3_dir: str) -> str:
    """
    优先使用参数文件中的 source_file 属性；
    若缺失或无效，则回退为 CH0-3/同名文件。
    """
    source_file = ""
    if source_file_attr is not None:
        try:
            source_file = (
                source_file_attr.decode("utf-8")
                if isinstance(source_file_attr, bytes)
                else str(source_file_attr)
            )
        except Exception:
            source_file = str(source_file_attr)
    if source_file and os.path.exists(source_file):
        return source_file

    fallback = os.path.join(ch0_3_dir, os.path.basename(param_path))
    return fallback


def _find_zero_events_for_one_param_file(
    param_path: str,
    ch0_3_dir: str,
    ch0_param_dir: str,
    ch1_param_dir: str,
    ch4_param_dir: str,
    ch5_param_dir: str,
) -> List[Tuple[str, int]]:
    """
    返回该参数文件中所有 min_ch3 == 0 的事件索引：
    [(source_file, event_idx), ...]
    """
    out: List[Tuple[str, int]] = []
    try:
        file_name = os.path.basename(param_path)
        ch0_path = os.path.join(ch0_param_dir, file_name)
        ch1_path = os.path.join(ch1_param_dir, file_name)
        ch4_path = os.path.join(ch4_param_dir, file_name)
        ch5_path = os.path.join(ch5_param_dir, file_name)
        if not (os.path.exists(ch0_path) and os.path.exists(ch1_path) and os.path.exists(ch4_path) and os.path.exists(ch5_path)):
            return out

        with h5py.File(param_path, "r") as f_param:
            if "min_ch3" not in f_param:
                return out
            min_ch3 = np.asarray(f_param["min_ch3"][...], dtype=np.float64)
            source_file = _resolve_source_file(param_path, f_param.attrs.get("source_file", None), ch0_3_dir)

        with h5py.File(ch0_path, "r") as f0:
            if "max_ch0" not in f0 or "ch0_min" not in f0 or "ch0ped_mean" not in f0:
                return out
            max_ch0 = np.asarray(f0["max_ch0"][...], dtype=np.float64)
            ch0_min = np.asarray(f0["ch0_min"][...], dtype=np.float64)
            ch0_ped_mean = np.asarray(f0["ch0ped_mean"][...], dtype=np.float64)

        with h5py.File(ch1_path, "r") as f1:
            if "ch1ped_mean" not in f1 or "ch1_min" not in f1:
                return out
            ch1_ped_mean = np.asarray(f1["ch1ped_mean"][...], dtype=np.float64)
            ch1_min = np.asarray(f1["ch1_min"][...], dtype=np.float64)

        with h5py.File(ch4_path, "r") as f4:
            if "max_ch4" not in f4 or "tmax_ch4" not in f4:
                return out
            max_ch4 = np.asarray(f4["max_ch4"][...], dtype=np.float64)
            tmax_ch4 = np.asarray(f4["tmax_ch4"][...], dtype=np.float64)

        with h5py.File(ch5_path, "r") as f5:
            if "max_ch5" not in f5:
                return out
            max_ch5 = np.asarray(f5["max_ch5"][...], dtype=np.float64)

        n = min(
            min_ch3.shape[0],
            max_ch0.shape[0],
            ch0_min.shape[0],
            ch0_ped_mean.shape[0],
            ch1_ped_mean.shape[0],
            ch1_min.shape[0],
            max_ch4.shape[0],
            tmax_ch4.shape[0],
            max_ch5.shape[0],
        )
        if n == 0:
            return out

        min_ch3 = min_ch3[:n]
        max_ch0 = max_ch0[:n]
        ch0_min = ch0_min[:n]
        ch0_ped_mean = ch0_ped_mean[:n]
        ch1_ped_mean = ch1_ped_mean[:n]
        ch1_min = ch1_min[:n]
        max_ch4 = max_ch4[:n]
        tmax_ch4 = tmax_ch4[:n]
        max_ch5 = max_ch5[:n]

        m1 = cut_ch0_min_positive(ch0_min)
        m2 = cut_ch0_max_saturation(max_ch0)
        m3 = cut_ch5_self_trigger(max_ch5)
        m4 = cut_pedestal_3sigma(ch0_ped_mean, ch1_ped_mean, max_ch5)
        m5 = cut_act(max_ch4, tmax_ch4)
        m6 = cut_mincut(ch0_min, ch1_min, max_ch4, tmax_ch4)
        six_mask = m1 & m2 & m3 & m4 & m5 & m6

        zero_indices = np.where(six_mask & (min_ch3 == 0))[0]
        for idx in zero_indices:
            out.append((source_file, int(idx)))
    except Exception as e:
        print(f"[警告] 读取参数文件失败，跳过: {param_path}, 错误: {e}")
    return out


def _collect_zero_events(
    ch3_param_dir: str,
    ch0_3_dir: str,
    ch0_param_dir: str,
    ch1_param_dir: str,
    ch4_param_dir: str,
    ch5_param_dir: str,
    max_workers: Optional[int] = None,
) -> List[Tuple[str, int]]:
    """
    并行扫描 CH3_parameters 下所有 h5，收集 min_ch3 == 0 的事件。
    """
    if not os.path.isdir(ch3_param_dir):
        raise FileNotFoundError(f"CH3_parameters 目录不存在: {ch3_param_dir}")

    files = [
        os.path.join(ch3_param_dir, name)
        for name in sorted(os.listdir(ch3_param_dir))
        if name.lower().endswith(".h5")
    ]
    if not files:
        print(f"目录 {ch3_param_dir} 下未找到 h5 文件。")
        return []

    workers = max_workers if (max_workers is not None and max_workers > 0) else (os.cpu_count() or 1)
    workers = max(1, workers)
    print(f"扫描 {len(files)} 个参数文件，使用 {workers} 个 CPU 核并行筛选“六个mask后 min_ch3 == 0”事件。")

    matched: List[Tuple[str, int]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _find_zero_events_for_one_param_file,
                p,
                ch0_3_dir,
                ch0_param_dir,
                ch1_param_dir,
                ch4_param_dir,
                ch5_param_dir,
            ): p
            for p in files
        }
        for fut in as_completed(futures):
            matched.extend(fut.result())

    print(f"共找到 {len(matched)} 个“六个mask后 min_ch3 == 0”的事件。")
    return matched


def _plot_one_event(
    source_file: str,
    event_idx: int,
    use_time_us: bool = False,
    sampling_interval_ns: float = 4.0,
) -> None:
    """
    绘制单个事件的 CH0/CH3 波形（双 y 轴）。
    """
    if not os.path.exists(source_file):
        print(f"[警告] 源文件不存在，跳过: {source_file}")
        return

    with h5py.File(source_file, "r") as f_src:
        if "channel_data" not in f_src:
            print(f"[警告] 源文件缺少 channel_data，跳过: {source_file}")
            return
        ch_data = f_src["channel_data"]
        time_samples, num_channels, num_events = ch_data.shape
        if num_channels <= 3:
            print(f"[警告] 源文件通道数不足（需要 CH0 和 CH3），跳过: {source_file}")
            return
        if event_idx < 0 or event_idx >= num_events:
            print(f"[警告] event_idx 越界，跳过: {source_file}, event={event_idx}")
            return

        wf_ch0 = ch_data[:, 0, event_idx].astype(np.float64)
        wf_ch3 = ch_data[:, 3, event_idx].astype(np.float64)

    if use_time_us:
        x = np.arange(time_samples, dtype=np.float64) * sampling_interval_ns / 1000.0
        xlabel = "Time [us]"
    else:
        x = np.arange(time_samples, dtype=np.int32)
        xlabel = "Sample index"

    fig, ax_left = plt.subplots(figsize=(10, 5.5))
    ax_right = ax_left.twinx()

    line0 = ax_left.plot(x, wf_ch0, color="C0", linewidth=1.0, label="CH0")[0]
    line3 = ax_right.plot(x, wf_ch3, color="C3", linewidth=1.0, label="CH3")[0]

    ax_left.set_xlabel(xlabel)
    ax_left.set_ylabel("CH0 amplitude", color="C0")
    ax_right.set_ylabel("CH3 amplitude", color="C3")
    ax_left.tick_params(axis="y", labelcolor="C0")
    ax_right.tick_params(axis="y", labelcolor="C3")

    title = f"{os.path.basename(source_file)} | event={event_idx} | min_ch3=0"
    ax_left.set_title(title)
    ax_left.grid(True, alpha=0.25, linestyle="--")
    ax_left.legend([line0, line3], ["CH0", "CH3"], loc="upper right")

    fig.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 CH3_parameters 中筛选 min_ch3==0 的事件并绘制 CH0/CH3 双 y 轴波形。"
    )
    parser.add_argument(
        "--ch3-param-dir",
        type=str,
        default=ch3_param_dir_default,
        help=f"CH3 参数目录，默认: {ch3_param_dir_default}",
    )
    parser.add_argument(
        "--ch0-3-dir",
        type=str,
        default=ch0_3_dir_default,
        help=f"CH0-3 原始波形目录，默认: {ch0_3_dir_default}",
    )
    parser.add_argument("--ch0-param-dir", type=str, default=ch0_param_dir_default, help=f"CH0 参数目录，默认: {ch0_param_dir_default}")
    parser.add_argument("--ch1-param-dir", type=str, default=ch1_param_dir_default, help=f"CH1 参数目录，默认: {ch1_param_dir_default}")
    parser.add_argument("--ch4-param-dir", type=str, default=ch4_param_dir_default, help=f"CH4 参数目录，默认: {ch4_param_dir_default}")
    parser.add_argument("--ch5-param-dir", type=str, default=ch5_param_dir_default, help=f"CH5 参数目录，默认: {ch5_param_dir_default}")
    parser.add_argument(
        "--max-plots",
        type=int,
        default=50,
        help="最多绘制多少个事件（默认 50，<=0 表示全部）。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="并行扫描参数文件使用的进程数；0 表示自动使用 os.cpu_count()。",
    )
    parser.add_argument(
        "--time-us",
        action="store_true",
        help="x 轴使用时间（us），否则使用采样点索引。",
    )
    parser.add_argument(
        "--sampling-interval-ns",
        type=float,
        default=4.0,
        help="采样间隔（ns），仅在 --time-us 时生效，默认 4.0。",
    )
    args = parser.parse_args()

    workers = None if args.workers <= 0 else args.workers
    matched_events = _collect_zero_events(
        ch3_param_dir=args.ch3_param_dir,
        ch0_3_dir=args.ch0_3_dir,
        ch0_param_dir=args.ch0_param_dir,
        ch1_param_dir=args.ch1_param_dir,
        ch4_param_dir=args.ch4_param_dir,
        ch5_param_dir=args.ch5_param_dir,
        max_workers=workers,
    )
    if not matched_events:
        print("未找到经过六个 mask 后且 min_ch3 == 0 的事件。")
        return

    if args.max_plots > 0:
        matched_events = matched_events[: args.max_plots]
    print(f"开始绘图，共 {len(matched_events)} 个事件。关闭当前图窗口后显示下一个。")

    for source_file, event_idx in matched_events:
        _plot_one_event(
            source_file=source_file,
            event_idx=event_idx,
            use_time_us=args.time_us,
            sampling_interval_ns=args.sampling_interval_ns,
        )

    print("绘图完成。")


if __name__ == "__main__":
    main()
