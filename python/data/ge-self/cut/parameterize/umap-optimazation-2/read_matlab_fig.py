#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
读取 MATLAB 保存的 .fig 文件中的图形数据。

说明
----
- 多数 .fig 实际是 MATLAB v5/v6 MAT-file（本仓库中的 `DZL_vetospec_12kev_0615.fig` 即如此），
  可用 `scipy.io.loadmat` 直接打开。
- R2014b 及以后保存的 .fig 也可能是 HDF5 格式；若 `loadmat` 失败，本脚本会尝试用 `h5py`
  打开并做浅层遍历（复杂嵌套结构需自行扩展）。

主要入口：`load_matlab_fig`、`extract_xy_series`、`extract_axes_info`、`plot_xy_series`。

命令行默认会绘图（适合在编辑器中按 F5 /「运行当前文件」）；若只要文本输出请加 `--no-plot`。
默认图窗仅绘制 DisplayName 为 raw data / basic cut / basic cut + ACV 的三条曲线（不含 errorbar 内部 line）。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

try:
    import scipy.io as sio
except ImportError as e:  # pragma: no cover
    raise ImportError("需要 scipy：pip install scipy") from e


MatStruct = Any

# 默认绘图只保留这些 DisplayName（不画 line[1] 等内部线段，也不画未列出的图例项）
DEFAULT_PLOT_DISPLAY_NAMES: Tuple[str, ...] = (
    "raw data",
    "basic cut",
    "basic cut + ACV",
)


def _series_for_plot(
    series: List[Dict[str, Any]],
    *,
    plot_all: bool,
) -> List[Dict[str, Any]]:
    if plot_all:
        return list(series)
    allowed = {n.strip() for n in DEFAULT_PLOT_DISPLAY_NAMES}
    by_name: Dict[str, Dict[str, Any]] = {}
    for s in series:
        name = str(s.get("display_name", "")).strip()
        if name in allowed:
            by_name[name] = s
    return [by_name[n] for n in DEFAULT_PLOT_DISPLAY_NAMES if n in by_name]


def _is_mat_struct(obj: Any) -> bool:
    return hasattr(obj, "_fieldnames") and hasattr(obj, "type")


def _get_figure_root(raw: Dict[str, Any]) -> Tuple[str, MatStruct]:
    """从 loadmat 结果中找到 handle graphics 根节点（变量名常为 hgS_xxxxx）。"""
    candidates = []
    for k, v in raw.items():
        if k.startswith("__"):
            continue
        if _is_mat_struct(v) and getattr(v, "type", None) == "figure":
            candidates.append((k, v))
    if len(candidates) == 1:
        return candidates[0]
    for k, v in raw.items():
        if k.startswith("__"):
            continue
        if _is_mat_struct(v) and hasattr(v, "children"):
            return k, v
    raise ValueError("未在 .fig 中找到有效的 figure 根结构（hgS_*）")


def load_matlab_fig(path: Union[str, Path]) -> Dict[str, Any]:
    """
    读取 .fig 为 Python dict（与 scipy.loadmat 一致）。
    使用 struct_as_record=False, squeeze_me=True 以便用属性访问 mat_struct。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    try:
        return sio.loadmat(str(path), struct_as_record=False, squeeze_me=True)
    except Exception as e_mat:
        # 新版 MATLAB 可能将 .fig 存为 HDF5，需 h5py
        try:
            import h5py

            with h5py.File(str(path), "r") as f:
                return _h5py_to_dict(f)
        except Exception as e_h5:
            raise OSError(
                f"无法用 scipy.loadmat 打开 {path}：{e_mat!r}；"
                f"h5py 亦失败：{e_h5!r}"
            ) from e_h5


def _h5py_to_dict(f: Any) -> Dict[str, Any]:
    """极简 HDF5 遍历，仅用于提示性导出；复杂 .fig 建议仍在 MATLAB 中另存为 .mat。"""
    out: Dict[str, Any] = {}

    def visit(name: str, obj: Any) -> None:
        if hasattr(obj, "shape") and hasattr(obj, "dtype"):
            try:
                out[name.replace("/", ".")] = obj[()]
            except Exception:
                out[name.replace("/", ".")] = "<dataset>"

    f.visititems(visit)
    return {"__hdf5_flat__": out}


def _walk_hg(
    node: Any, prefix: str = ""
) -> Iterator[Tuple[str, Any]]:
    """深度优先遍历 handle graphics 树，产出 (路径, 节点)。"""
    yield prefix or "root", node
    if not _is_mat_struct(node):
        return
    ch = getattr(node, "children", None)
    if ch is None:
        return
    arr = np.atleast_1d(ch)
    for i, child in enumerate(arr.flat):
        sub = f"{prefix}/children[{i}]" if prefix else f"children[{i}]"
        yield from _walk_hg(child, sub)


def _props_xy(props: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not hasattr(props, "XData") or not hasattr(props, "YData"):
        return None
    xd = np.asarray(props.XData).astype(np.float64).ravel()
    yd = np.asarray(props.YData).astype(np.float64).ravel()
    if xd.size == 0 or yd.size == 0:
        return None
    n = min(xd.size, yd.size)
    return xd[:n], yd[:n]


def extract_xy_series(
    raw: Dict[str, Any],
    include_child_lines: bool = True,
) -> List[Dict[str, Any]]:
    """
    从 load_matlab_fig 的结果中提取所有带 XData/YData 的序列。

    Parameters
    ----------
    raw : loadmat 返回的 dict
    include_child_lines : 若为 False，仅保留含 DisplayName 的父级对象（如 errorbar 主序列），
        跳过纯 'line' 子对象（errorbar 内部的线段往往与父级重复或更密）。

    每个序列 dict 键：
        path, type, display_name, x, y, l_data, u_data（后两者来自 errorbar 时存在）
    """
    _, root = _get_figure_root(raw)
    series: List[Dict[str, Any]] = []

    for path, node in _walk_hg(root):
        if not _is_mat_struct(node):
            continue
        typ = getattr(node, "type", "")
        props = getattr(node, "properties", None)
        if props is None:
            continue
        xy = _props_xy(props)
        if xy is None:
            continue
        x, y = xy
        if not include_child_lines and typ == "line":
            continue

        display_name = ""
        if hasattr(props, "DisplayName"):
            dn = props.DisplayName
            display_name = str(dn) if dn is not None else ""

        item: Dict[str, Any] = {
            "path": path,
            "type": str(typ),
            "display_name": display_name,
            "x": x,
            "y": y,
        }
        if hasattr(props, "LData") and hasattr(props, "UData"):
            try:
                lo = np.asarray(props.LData).astype(np.float64).ravel()
                hi = np.asarray(props.UData).astype(np.float64).ravel()
                if lo.size and hi.size:
                    m = min(lo.size, hi.size, x.size)
                    item["l_data"] = lo[:m]
                    item["u_data"] = hi[:m]
            except Exception:
                pass
        series.append(item)

    return series


def extract_axes_info(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    从 .fig 中第一个 axes 读取坐标范围、Y 轴刻度类型、轴标题文字（若有）。
    返回键：xlim, ylim, yscale, xlabel, ylabel, font_name（可能为空字符串）。
    """
    _, root = _get_figure_root(raw)
    ax_node = None
    for c in np.atleast_1d(root.children).flat:
        if _is_mat_struct(c) and getattr(c, "type", "") == "axes":
            ax_node = c
            break
    if ax_node is None:
        return {}

    pr = ax_node.properties
    out: Dict[str, Any] = {
        "xlim": None,
        "ylim": None,
        "yscale": "linear",
        "xlabel": "",
        "ylabel": "",
        "font_name": "",
    }
    if hasattr(pr, "XLim"):
        out["xlim"] = np.asarray(pr.XLim, dtype=np.float64).ravel()
    if hasattr(pr, "YLim"):
        out["ylim"] = np.asarray(pr.YLim, dtype=np.float64).ravel()
    if hasattr(pr, "YScale"):
        out["yscale"] = str(pr.YScale)
    if hasattr(pr, "FontName"):
        out["font_name"] = str(pr.FontName)

    for c in np.atleast_1d(ax_node.children).flat:
        if not _is_mat_struct(c) or getattr(c, "type", "") != "text":
            continue
        props = c.properties
        if not hasattr(props, "String"):
            continue
        s = str(props.String)
        if "Energy" in s:
            out["xlabel"] = s
        elif "Event" in s or "Rate" in s or "cpkkd" in s:
            out["ylabel"] = s

    return out


def plot_xy_series(
    series: List[Dict[str, Any]],
    *,
    axes_info: Optional[Dict[str, Any]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    dpi: int = 150,
) -> None:
    """
    将 `extract_xy_series` 得到的序列绘制成图（误差棒与 MATLAB errorbar 一致时使用 LData/UData）。

    默认使用 axes_info 中的对数坐标与坐标范围；对数轴下仅绘制 y>0 的点。
    """
    if not show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if axes_info is None:
        axes_info = {}

    font = axes_info.get("font_name") or ""
    if "times" in font.lower():
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Times"]
    elif font:
        try:
            plt.rcParams["font.family"] = font
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)

    y_is_log = axes_info.get("yscale", "linear") == "log"

    for i, s in enumerate(series):
        x = np.asarray(s["x"], dtype=np.float64).ravel()
        y = np.asarray(s["y"], dtype=np.float64).ravel()
        n = min(x.size, y.size)
        x, y = x[:n], y[:n]
        label = str(s.get("display_name", "")).strip()
        if not label:
            label = f"{s.get('type', 'series')}[{i}]"
        fmt = "o"
        markersize = 1.8
        if "l_data" in s and "u_data" in s:
            lo = np.asarray(s["l_data"], dtype=np.float64).ravel()
            hi = np.asarray(s["u_data"], dtype=np.float64).ravel()
            m = min(lo.size, hi.size, n)
            lo, hi = lo[:m], hi[:m]
            x, y = x[:m], y[:m]
            if y_is_log:
                ok = y > 0
                x, y, lo, hi = x[ok], y[ok], lo[ok], hi[ok]
            # yerr: (2, n) 第一行=下误差, 第二行=上误差，与 MATLAB LData/UData 一致
            yerr = np.vstack([lo, hi])
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt=fmt,
                label=label,
                capsize=2,
                capthick=0.8,
                elinewidth=1.2,
                markersize=markersize,
            )
        else:
            if y_is_log:
                ok = y > 0
                x, y = x[ok], y[ok]
            ax.plot(x, y, fmt, label=label, markersize=markersize)

    yscale = axes_info.get("yscale", "linear")
    if yscale == "log":
        ax.set_yscale("log")

    xl = axes_info.get("xlim")
    if xl is not None and len(xl) >= 2:
        ax.set_xlim(float(xl[0]), float(xl[1]))
    yl = axes_info.get("ylim")
    if yl is not None and len(yl) >= 2:
        ax.set_ylim(float(yl[0]), float(yl[1]))

    if axes_info.get("xlabel"):
        ax.set_xlabel(axes_info["xlabel"])
    if axes_info.get("ylabel"):
        ax.set_ylabel(axes_info["ylabel"])

    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(Path(save_path), dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _main() -> None:
    here = Path(__file__).resolve().parent
    default_fig = here / "DZL_vetospec_12kev_0615.fig"

    p = argparse.ArgumentParser(description="读取 MATLAB .fig 中的曲线数据")
    p.add_argument(
        "fig",
        nargs="?",
        type=Path,
        default=default_fig,
        help=f".fig 路径（默认: {default_fig.name}）",
    )
    p.add_argument(
        "--only-named",
        action="store_true",
        help="仅保留 DisplayName 非空的序列（通常对应图例中的曲线）",
    )
    p.add_argument(
        "--no-child-lines",
        action="store_true",
        help="跳过 type=='line' 的子对象（errorbar 内部的线段）",
    )
    p.add_argument(
        "--npz",
        type=Path,
        default=None,
        help="将序列保存为 .npz（键 series_000, series_001, ...）",
    )
    p.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
        help="仅打印/导出，不绘图（默认：直接运行或 F5 时会弹出图窗）",
    )
    p.set_defaults(plot=True)
    p.add_argument(
        "--plot-all",
        action="store_true",
        help="绘图时包含全部已提取序列；默认只画 "
        + " / ".join(DEFAULT_PLOT_DISPLAY_NAMES),
    )
    p.add_argument(
        "--plot-save",
        type=Path,
        default=None,
        metavar="PATH",
        help="将图保存为该路径（默认可与 --no-show 联用）",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="不弹出窗口；常与 --plot-save 或默认保存到与 .fig 同名的 .png",
    )
    args = p.parse_args()

    raw = load_matlab_fig(args.fig)
    keys = [k for k in raw if not k.startswith("__")]
    print(f"顶层变量: {keys}")

    series = extract_xy_series(
        raw,
        include_child_lines=not args.no_child_lines,
    )
    if args.only_named:
        series = [s for s in series if str(s.get("display_name", "")).strip()]
    named_count = sum(1 for s in series if str(s.get("display_name", "")).strip())
    print(f"共输出 {len(series)} 条 XY 序列；其中带 DisplayName 的 {named_count} 条")

    for i, s in enumerate(series[:20]):
        extra = ""
        if "l_data" in s:
            extra = " [±error]"
        print(
            f"  [{i}] {s['type']!r} {s['display_name']!r} "
            f"n={len(s['x'])}{extra}"
        )
    if len(series) > 20:
        print(f"  ... 另有 {len(series) - 20} 条未列出")

    if args.npz:
        pack = {}
        for i, s in enumerate(series):
            prefix = f"series_{i:03d}"
            pack[f"{prefix}_x"] = s["x"]
            pack[f"{prefix}_y"] = s["y"]
            pack[f"{prefix}_meta_type"] = np.array(s["type"])
            pack[f"{prefix}_meta_name"] = np.array(s["display_name"])
            if "l_data" in s:
                pack[f"{prefix}_l_data"] = s["l_data"]
                pack[f"{prefix}_u_data"] = s["u_data"]
        np.savez_compressed(args.npz, **pack)
        print(f"已写入 {args.npz}")

    if args.plot:
        to_plot = _series_for_plot(series, plot_all=args.plot_all)
        if not to_plot:
            print(
                "绘图：无匹配曲线（检查 .fig 中 DisplayName 是否与 "
                f"{DEFAULT_PLOT_DISPLAY_NAMES} 一致）；可用 --plot-all 绘制全部序列。"
            )
        axinfo = extract_axes_info(raw)
        save = args.plot_save
        if args.no_show and save is None:
            save = args.fig.with_suffix(".png")
        if to_plot:
            plot_xy_series(
                to_plot,
                axes_info=axinfo,
                save_path=save,
                show=not args.no_show,
            )
            if save is not None:
                print(f"图已保存: {save}")


if __name__ == "__main__":
    _main()
