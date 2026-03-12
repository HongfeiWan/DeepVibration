import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np


def _print_attrs(prefix: str, obj: h5py.AttributeManager) -> None:
    """打印对象的 attributes（仅用于结构检查）。"""
    if not obj.keys():
        return
    print(f"{prefix}  attrs:")
    for k, v in obj.items():
        print(f"{prefix}    - {k}: {v!r}")


def _visit_item(name: str, node) -> None:
    """h5py.File.visititems 的回调函数，用于结构检查。"""
    indent_level = name.count("/")
    indent = "  " * indent_level

    if isinstance(node, h5py.Group):
        print(f"{indent}Group: /{name if name else ''}")
        _print_attrs(indent, node.attrs)
    elif isinstance(node, h5py.Dataset):
        print(
            f"{indent}Dataset: /{name}  "
            f"shape={node.shape}, dtype={node.dtype}"
        )
        _print_attrs(indent, node.attrs)


def inspect_hdf5(path: Path) -> None:
    """打印 hdf5 文件的大概结构和关键信息。"""
    if not path.exists():
        raise FileNotFoundError(f"hdf5 文件不存在: {path}")

    with h5py.File(path, "r") as f:
        print(f"文件: {path}")
        print("根 attributes:")
        for k, v in f.attrs.items():
            print(f"  - {k}: {v!r}")
        print("\n层次结构:\n")
        print("Group: /")
        _print_attrs("", f.attrs)
        f.visititems(_visit_item)


def _load_event_mapping(path: Path):
    """
    读取 15parameter_spectrum.py 写出的事件映射 HDF5。

    结构参考：
        - file_paths               : (n_files,)  字节串/字符串
        - event_file_indices       : (n_events,)
        - event_event_indices      : (n_events,)
        - event_cluster_labels     : (n_events,)
    """
    if not path.exists():
        raise FileNotFoundError(f"事件映射 HDF5 不存在: {path}")

    with h5py.File(path, "r") as f:
        file_paths_raw = f["file_paths"][...]
        event_file_indices = f["event_file_indices"][...]
        event_event_indices = f["event_event_indices"][...]
        labels = f["event_cluster_labels"][...]

    file_paths: List[str] = []
    for p in file_paths_raw:
        if isinstance(p, bytes):
            file_paths.append(p.decode("utf-8"))
        else:
            file_paths.append(str(p))

    return file_paths, event_file_indices, event_event_indices, labels


def _group_indices_by_cluster(
    labels: np.ndarray,
) -> Dict[int, np.ndarray]:
    """把全体事件索引按 cluster label 分组。"""
    cluster_to_indices: Dict[int, List[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        lab_int = int(lab)
        cluster_to_indices[lab_int].append(i)
    return {lab: np.asarray(idxs, dtype=np.int64) for lab, idxs in cluster_to_indices.items()}


def _plot_cluster_ch0_ch3_waveforms(
    file_paths: Sequence[str],
    event_file_indices: np.ndarray,
    event_event_indices: np.ndarray,
    cluster_indices: np.ndarray,
    cluster_label: int,
    max_events_per_cluster: int = 9,
    ch0_index: int = 0,
    ch3_index: int = 3,
) -> None:
    """
    对某一个 cluster，把对应事件的 ch0 波形叠加画在一张图上。
    """
    if cluster_indices.size == 0:
        return

    # 每个 cluster 最多只展示 9 个事件
    limit = min(max_events_per_cluster, 9)
    if cluster_indices.size > limit:
        cluster_indices = np.random.choice(
            cluster_indices, size=limit, replace=False
        )

    # 准备 (file_idx -> [event_indices...]) 映射，避免重复打开文件
    file_to_events: Dict[int, List[int]] = defaultdict(list)
    for i in cluster_indices:
        fi = int(event_file_indices[i])
        ev = int(event_event_indices[i])
        file_to_events[fi].append(ev)

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"]})

    # 先把需要画的 ch0 / ch3 波形收集出来，最多 9 条（相同事件）
    waveforms_ch0: List[np.ndarray] = []
    waveforms_ch3: List[np.ndarray] = []
    meta: List[str] = []  # 保存 (file_idx, event_idx) 文字，用于 subplot 标题

    for fi, ev_list in file_to_events.items():
        if fi < 0 or fi >= len(file_paths):
            continue
        fpath = Path(file_paths[fi])
        if not fpath.exists():
            print(f"警告: 源数据文件不存在，跳过: {fpath}")
            continue

        with h5py.File(fpath, "r") as f_ch:
            if "channel_data" not in f_ch:
                print(f"警告: 文件中缺少 'channel_data' 数据集，跳过: {fpath}")
                continue
            ch_data = f_ch["channel_data"]
            # ch_data: [n_samples, n_channels, n_events]
            for ev in ev_list:
                if ev < 0 or ev >= ch_data.shape[2]:
                    continue
                wf_ch0 = np.asarray(ch_data[:, ch0_index, ev], dtype=np.float64)
                wf_ch3 = np.asarray(ch_data[:, ch3_index, ev], dtype=np.float64)
                waveforms_ch0.append(wf_ch0)
                waveforms_ch3.append(wf_ch3)
                meta.append(f"file_idx={fi}, ev={ev}")
                if len(waveforms_ch0) >= limit:
                    break
            if len(waveforms_ch0) >= limit:
                break

    if not waveforms_ch0:
        print(f"Cluster {cluster_label}: 未找到可用的 ch{ch0_index}/ch{ch3_index} 波形。")
        return

    # 按最多 3x3 子图方式展示最多 9 个 ch0/ch3 图像：
    #   第一行：ch0
    #   第二行：ch3
    n = len(waveforms_ch0)
    n_cols = 3
    n_rows = 2

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3 * n_rows),
        squeeze=False,
    )

    # 第一行：ch0
    for idx in range(min(n, n_cols * 1)):
        c = idx % n_cols
        ax = axes[0][c]
        ax.plot(waveforms_ch0[idx], color="C0")
        ax.set_title(f"ch{ch0_index} | {meta[idx]}", fontsize=8)
        ax.grid(True, alpha=0.3)

    # 第二行：ch3（或指定的另一通道）
    for idx in range(min(n, n_cols * 1)):
        c = idx % n_cols
        ax = axes[1][c]
        ax.plot(waveforms_ch3[idx], color="C1")
        ax.set_title(f"ch{ch3_index} | {meta[idx]}", fontsize=8)
        ax.grid(True, alpha=0.3)

    # 多余的子图关掉坐标轴
    # 多余的子图关掉坐标轴
    for r in range(n_rows):
        for c in range(n_cols):
            idx = c  # 这里只是简单限制列数，行数固定
            if (r == 0 and idx >= n) or (r == 1 and idx >= n):
                axes[r][c].axis("off")

    fig.suptitle(
        f"Cluster {cluster_label} | ch{ch0_index} & ch{ch3_index} waveforms (n={n})",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def visualize_all_clusters_ch0(
    mapping_path: Path,
    include_noise: bool = False,
    max_events_per_cluster: int = 200,
    ch_index: int = 0,
) -> None:
    """
    对事件映射 HDF5 中的每个 cluster label，把对应事件的 ch0 波形可视化。

    - 每个 cluster 一张图，叠加若干条 ch0 波形；
    - 默认跳过 label = -1（噪声），可通过 include_noise=True 开启。
    """
    file_paths, event_file_indices, event_event_indices, labels = _load_event_mapping(
        mapping_path
    )
    cluster_to_indices = _group_indices_by_cluster(labels)

    unique_labels = sorted(cluster_to_indices.keys())
    print(f"在映射文件中找到 {len(unique_labels)} 个 cluster label: {unique_labels}")

    for lab in unique_labels:
        if not include_noise and lab == -1:
            print("跳过噪声点 cluster (-1)")
            continue
        idxs = cluster_to_indices[lab]
        print(f"\n可视化 Cluster {lab}，事件数 = {idxs.size}")
        _plot_cluster_ch0_ch3_waveforms(
            file_paths=file_paths,
            event_file_indices=event_file_indices,
            event_event_indices=event_event_indices,
            cluster_indices=idxs,
            cluster_label=lab,
            max_events_per_cluster=max_events_per_cluster,
            ch0_index=ch_index,
            ch3_index=3,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "1) 打印 HDF5 结构；"
            "2) 对 15 参数 UMAP+HDBSCAN 事件映射中的每个 cluster 可视化 ch0 波形。"
        )
    )
    parser.add_argument(
        "hdf5_path",
        nargs="?",
        help="事件映射 HDF5 路径；若不指定，则使用项目中的示例文件。",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="只打印 HDF5 结构，不做波形可视化。",
    )
    parser.add_argument(
        "--include-noise",
        action="store_true",
        help="是否包含 cluster label = -1 的噪声点。",
    )
    parser.add_argument(
        "--max-events-per-cluster",
        type=int,
        default=200,
        help="每个 cluster 最多可视化的事件数（随机子采样）。",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="要可视化的通道索引（默认 ch0=0）。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.hdf5_path:
        hdf5_path = Path(args.hdf5_path)
    else:
        # 默认使用 15parameter_spectrum.py 生成的示例文件（相对于当前脚本所在目录）
        script_dir = Path(__file__).resolve().parent
        hdf5_path = (
            script_dir
            / ".."
            / ".."
            / ".."
            / "data"
            / "hdf5"
            / "ch0_15param_umap_hdbscan_eventmap_1863169events_20260312_174533.h5"
        ).resolve()

    if args.inspect_only:
        inspect_hdf5(hdf5_path)
    else:
        visualize_all_clusters_ch0(
            mapping_path=hdf5_path,
            include_noise=args.include_noise,
            max_events_per_cluster=args.max_events_per_cluster,
            ch_index=args.channel,
        )


if __name__ == "__main__":
    main()

