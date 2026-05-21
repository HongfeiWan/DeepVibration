#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
校验并修正 CH2/CH3 参数文件中的幅值极值。

需求来源：
- 读取 data/hdf5/raw_pulse/CH2_parameters 与 CH3_parameters 中的参数（结构参考 python/data/fit_ch2_ch3_parallel.py）
- 通过参数文件 attrs["source_file"] 定位到 data/hdf5/raw_pulse/CH0-3 中对应事件波形
- 校验 min_ch2/max_ch2/min_ch3/max_ch3 是否等于真实波形的最小/最大值；若不是则写回修正

用法示例：
- 校验并修正全部 CH2/CH3 参数文件（默认 root=项目根目录）：
    python python/data/debug_ch2_param

- 指定数据根目录（root 下应包含 data/hdf5/raw_pulse/CH0-3 等目录结构）：
    python python/data/debug_ch2_param --root /path/to/DeepVibration

- 只处理某一个参数文件（可以给 basename 或完整路径）：
    python python/data/debug_ch2_param --only 20250520_xxx_processed.h5
    python python/data/debug_ch2_param --only /abs/path/to/CH2_parameters/xxx.h5
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable, Optional, Tuple

import h5py
import numpy as np


@dataclass(frozen=True)
class _Dirs:
    project_root: str
    ch0_3_dir: str
    ch2_param_dir: str
    ch3_param_dir: str


def _infer_dirs(root: Optional[str]) -> _Dirs:
    # 本脚本位于 .../python/data/ 下
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)  # .../python
    project_root = os.path.dirname(parent_dir)
    if root is not None:
        project_root = os.path.abspath(root)

    raw_pulse = os.path.join(project_root, "data", "hdf5", "raw_pulse")
    return _Dirs(
        project_root=project_root,
        ch0_3_dir=os.path.join(raw_pulse, "CH0-3"),
        ch2_param_dir=os.path.join(raw_pulse, "CH2_parameters"),
        ch3_param_dir=os.path.join(raw_pulse, "CH3_parameters"),
    )


def _decode_attr(x) -> str:
    if x is None:
        return ""
    if isinstance(x, bytes):
        try:
            return x.decode("utf-8")
        except Exception:
            return x.decode(errors="ignore")
    return str(x)


def _load_channel_waveforms(source_file: str, channel_idx: int, n_events: int) -> Tuple[np.ndarray, bool]:
    """
    从 CH0-3 源文件读取指定通道波形，返回：
      - wf: shape 为 (time, n_events) 或 (n_events, time)
      - is_time_ch_event: 是否为 (time, ch, event) 布局
    """
    with h5py.File(source_file, "r") as f:
        ch_data = f["channel_data"]
        if ch_data.ndim != 3:
            raise ValueError(f"channel_data 维度异常: {ch_data.shape}, file={source_file}")

        d0, d1, d2 = ch_data.shape
        if channel_idx >= d1:
            raise ValueError(f"通道数不足: channel_idx={channel_idx}, num_channels={d1}, file={source_file}")

        # 兼容两种常见布局：
        # 1) (time, ch, event)  -> d0 通常 >= d2（本项目常见 time_samples=30000, events~10000）
        # 2) (event, ch, time)
        is_time_ch_event = d0 >= d2
        if is_time_ch_event:
            # (time, ch, event)
            ne = min(int(d2), int(n_events))
            wf = np.asarray(ch_data[:, channel_idx, :ne], dtype=np.float32)
            return wf, True  # (time, ne)
        else:
            # (event, ch, time)
            ne = min(int(d0), int(n_events))
            wf = np.asarray(ch_data[:ne, channel_idx, :], dtype=np.float32)
            return wf, False  # (ne, time)


def _compute_extrema(wf: np.ndarray, is_time_ch_event: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    输入 wf 为：
      - is_time_ch_event=True : (time, n_events)
      - is_time_ch_event=False: (n_events, time)
    返回 (max, min, tmax, tmin) 均为按事件维度的一维数组
    """
    if wf.size == 0:
        zf = np.zeros(0, dtype=np.float32)
        zi = np.zeros(0, dtype=np.int32)
        return zf, zf, zi, zi

    if is_time_ch_event:
        max_arr = np.max(wf, axis=0).astype(np.float32, copy=False)
        min_arr = np.min(wf, axis=0).astype(np.float32, copy=False)
        tmax_arr = np.argmax(wf, axis=0).astype(np.int32, copy=False)
        tmin_arr = np.argmin(wf, axis=0).astype(np.int32, copy=False)
    else:
        max_arr = np.max(wf, axis=1).astype(np.float32, copy=False)
        min_arr = np.min(wf, axis=1).astype(np.float32, copy=False)
        tmax_arr = np.argmax(wf, axis=1).astype(np.int32, copy=False)
        tmin_arr = np.argmin(wf, axis=1).astype(np.int32, copy=False)
    return max_arr, min_arr, tmax_arr, tmin_arr


def _load_1d_float(f: h5py.File, key: str) -> Optional[np.ndarray]:
    if key not in f:
        return None
    arr = np.asarray(f[key][...], dtype=np.float32)
    return arr.reshape(-1)


def _load_1d_int(f: h5py.File, key: str) -> Optional[np.ndarray]:
    if key not in f:
        return None
    arr = np.asarray(f[key][...], dtype=np.int32)
    return arr.reshape(-1)


def _needs_update_float(old: Optional[np.ndarray], new: np.ndarray, atol: float = 0.0) -> bool:
    if old is None:
        return True
    n = min(old.shape[0], new.shape[0])
    if n == 0:
        return old.shape[0] != new.shape[0]
    return not np.allclose(old[:n], new[:n], atol=atol, rtol=0.0, equal_nan=True)


def _needs_update_int(old: Optional[np.ndarray], new: np.ndarray) -> bool:
    if old is None:
        return True
    n = min(old.shape[0], new.shape[0])
    if n == 0:
        return old.shape[0] != new.shape[0]
    return not np.array_equal(old[:n], new[:n])


def _update_dataset(f: h5py.File, key: str, data: np.ndarray) -> None:
    if key in f:
        del f[key]
    f.create_dataset(key, data=data)


def _validate_and_fix_one_param_file(param_path: str, default_channel_idx: int, atol: float) -> str:
    """
    校验并修正某个 CH2/CH3 参数文件中的：
      - max_ch{idx}, min_ch{idx}, tmax_ch{idx}, tmin_ch{idx}
    至少保证 min/max 正确（你要求的），tmin/tmax 同步保持一致（避免后续用到索引时报错）。
    """
    if not os.path.exists(param_path):
        return f"[跳过] 参数文件不存在: {param_path}"

    with h5py.File(param_path, "r") as f_param:
        source_file = _decode_attr(f_param.attrs.get("source_file", None))
        ch_idx_attr = f_param.attrs.get("channel_index", None)
        ch_idx = int(ch_idx_attr) if ch_idx_attr is not None else int(default_channel_idx)

        if not source_file:
            return f"[跳过] 缺少 source_file 属性: {param_path}"
        if "tanh_p0" not in f_param:
            return f"[跳过] 缺少 tanh_p0 数据集，无法确定事件数: {param_path}"
        n_param_events = int(f_param["tanh_p0"].shape[0])

        max_key = f"max_ch{ch_idx}"
        min_key = f"min_ch{ch_idx}"
        tmax_key = f"tmax_ch{ch_idx}"
        tmin_key = f"tmin_ch{ch_idx}"

        old_max = _load_1d_float(f_param, max_key)
        old_min = _load_1d_float(f_param, min_key)
        old_tmax = _load_1d_int(f_param, tmax_key)
        old_tmin = _load_1d_int(f_param, tmin_key)

    if not os.path.exists(source_file):
        return f"[跳过] 找不到 source_file: {source_file} (from {param_path})"

    wf, is_time_ch_event = _load_channel_waveforms(source_file, ch_idx, n_param_events)
    max_arr, min_arr, tmax_arr, tmin_arr = _compute_extrema(wf, is_time_ch_event)

    # 参数文件事件数可能大于源文件事件数：保持与参数文件长度一致，尾部补零
    if n_param_events > max_arr.shape[0]:
        pad_f = np.zeros(n_param_events - max_arr.shape[0], dtype=np.float32)
        pad_i = np.zeros(n_param_events - tmax_arr.shape[0], dtype=np.int32)
        max_to_write = np.concatenate([max_arr, pad_f])
        min_to_write = np.concatenate([min_arr, pad_f])
        tmax_to_write = np.concatenate([tmax_arr, pad_i])
        tmin_to_write = np.concatenate([tmin_arr, pad_i])
    else:
        max_to_write = max_arr[:n_param_events]
        min_to_write = min_arr[:n_param_events]
        tmax_to_write = tmax_arr[:n_param_events]
        tmin_to_write = tmin_arr[:n_param_events]

    need_max = _needs_update_float(old_max, max_to_write, atol=atol)
    need_min = _needs_update_float(old_min, min_to_write, atol=atol)
    need_tmax = _needs_update_int(old_tmax, tmax_to_write)
    need_tmin = _needs_update_int(old_tmin, tmin_to_write)

    if not (need_max or need_min or need_tmax or need_tmin):
        return f"[OK] 极值一致，无需修改: {os.path.basename(param_path)}"

    with h5py.File(param_path, "a") as f_param:
        if need_max:
            _update_dataset(f_param, f"max_ch{ch_idx}", max_to_write)
        if need_min:
            _update_dataset(f_param, f"min_ch{ch_idx}", min_to_write)
        if need_tmax:
            _update_dataset(f_param, f"tmax_ch{ch_idx}", tmax_to_write)
        if need_tmin:
            _update_dataset(f_param, f"tmin_ch{ch_idx}", tmin_to_write)

    parts = []
    if need_max:
        parts.append("max")
    if need_min:
        parts.append("min")
    if need_tmax:
        parts.append("tmax")
    if need_tmin:
        parts.append("tmin")
    return f"[FIX] 已修正 {','.join(parts)}: {os.path.basename(param_path)}"


def _iter_param_files(param_dir: str) -> Iterable[str]:
    if not os.path.isdir(param_dir):
        return []
    for name in sorted(os.listdir(param_dir)):
        if name.lower().endswith(".h5"):
            yield os.path.join(param_dir, name)


def _resolve_only_path(only: str, param_dir: str) -> str:
    # 允许传 basename 或全路径
    if os.path.isabs(only) or os.path.exists(only):
        return os.path.abspath(only)
    return os.path.join(param_dir, only)


def main() -> None:
    ap = argparse.ArgumentParser(description="校验并修正 CH2/CH3 参数文件的 min/max 是否与源波形一致。")
    ap.add_argument("--root", default=None, help="项目根目录（默认为脚本自动推断）")
    ap.add_argument("--only", default=None, help="只处理一个参数文件（basename 或绝对路径）")
    ap.add_argument("--channel", choices=["ch2", "ch3", "both"], default="both", help="处理 CH2/CH3 或两者")
    ap.add_argument("--atol", type=float, default=0.0, help="浮点比较容差（默认 0，严格一致）")
    ap.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="并行 worker 数（按文件粒度并行；默认 1）",
    )
    args = ap.parse_args()

    dirs = _infer_dirs(args.root)
    print(f"project_root = {dirs.project_root}")
    print(f"CH0-3         = {dirs.ch0_3_dir}")
    print(f"CH2_params    = {dirs.ch2_param_dir}")
    print(f"CH3_params    = {dirs.ch3_param_dir}")

    targets: list[tuple[str, int]] = []
    if args.channel in ("ch2", "both"):
        targets.append((dirs.ch2_param_dir, 2))
    if args.channel in ("ch3", "both"):
        targets.append((dirs.ch3_param_dir, 3))

    any_found = False
    for param_dir, default_idx in targets:
        if args.only:
            param_path = _resolve_only_path(args.only, param_dir)
            print(_validate_and_fix_one_param_file(param_path, default_channel_idx=default_idx, atol=args.atol))
            any_found = True
            continue

        files = list(_iter_param_files(param_dir))
        if not files:
            print(f"[提示] 目录下未找到 .h5 参数文件: {param_dir}")
            continue
        any_found = True
        jobs = int(args.jobs)
        if jobs <= 1 or len(files) <= 1:
            for p in files:
                print(_validate_and_fix_one_param_file(p, default_channel_idx=default_idx, atol=args.atol))
            continue

        worker_count = min(jobs, len(files), (os.cpu_count() or 1))
        print(f"[并行] {os.path.basename(param_dir)}: files={len(files)}, worker={worker_count}")
        with ProcessPoolExecutor(max_workers=worker_count) as ex:
            futs = [
                ex.submit(_validate_and_fix_one_param_file, p, default_idx, args.atol)
                for p in files
            ]
            for fut in as_completed(futs):
                print(fut.result())

    if not any_found:
        print("[提示] 未找到任何可处理的参数文件。请确认数据目录是否存在，或用 --root 指定正确的项目根目录。")


if __name__ == "__main__":
    main()

