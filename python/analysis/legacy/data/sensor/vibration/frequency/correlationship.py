#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
振动传感器频率时间相关性分析脚本

功能：
- 调用 frequency/select.py 中的 select_by_date_range_vibration 读入 5 个探测器的频率数据
- 对每个分量 Frequency_x / Frequency_y / Frequency_z，分别计算 5×5 的相关性矩阵
- 可选画出相关性矩阵的热力图
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Dict, Optional, Tuple, List
import importlib.util


# 动态导入同目录下的 select.py 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
select_path = os.path.join(current_dir, 'select.py')
spec = importlib.util.spec_from_file_location("freq_select", select_path)
select_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(select_module)
select_by_date_range_vibration = select_module.select_by_date_range_vibration


def compute_frequency_correlation_matrices(
    data_dict: Dict[str, np.ndarray],
    detectors: Optional[List[int]] = None
) -> Dict[str, pd.DataFrame]:
    """
    计算 5 个探测器在 x/y/z 三个方向上的频率时间相关性矩阵

    参数:
        data_dict: 来自 select_by_date_range_vibration 的数据字典，
                   至少包含 'datetime', 'detector_num', 'Frequency_x/y/z'
        detectors: 需要参与相关性计算的探测器编号列表（默认自动按数据中的唯一值排序）

    返回:
        一个字典，键为 'x'/'y'/'z'，值为对应的 pandas DataFrame 相关性矩阵（行列均为探测器编号）
    """
    required_keys = ['datetime', 'detector_num']
    for k in required_keys:
        if k not in data_dict:
            raise ValueError(f"data_dict 中缺少必要列 '{k}'，无法计算相关性矩阵")

    # 构建 DataFrame 方便后续透视和相关性计算
    df_dict = {
        'datetime': data_dict['datetime'],
        'detector_num': data_dict['detector_num'].astype(int),
    }

    for col in ['Frequency_x', 'Frequency_y', 'Frequency_z']:
        if col in data_dict:
            df_dict[col] = data_dict[col].astype(float)

    df = pd.DataFrame(df_dict)

    if detectors is None:
        detectors = sorted(np.unique(df['detector_num']).astype(int))

    corr_mats: Dict[str, pd.DataFrame] = {}
    axis_map = {'x': 'Frequency_x', 'y': 'Frequency_y', 'z': 'Frequency_z'}

    for axis, col_name in axis_map.items():
        if col_name not in df.columns:
            print(f"警告：数据中没有 {col_name}，跳过 {axis} 方向的相关性计算")
            continue

        # 透视表：索引为时间，列为探测器编号，值为对应频率
        pivot = df.pivot(index='datetime', columns='detector_num', values=col_name)

        # 只保留指定的探测器列，并按编号排序
        existing_cols = [d for d in detectors if d in pivot.columns]
        if len(existing_cols) == 0:
            print(f"警告：在 {axis} 方向上没有任何指定探测器的数据，跳过")
            continue

        pivot = pivot[existing_cols]

        # 计算皮尔逊相关系数矩阵（自动按列两两相关）
        corr = pivot.corr(method='pearson')
        corr_mats[axis] = corr

    return corr_mats


def plot_frequency_correlation_heatmaps(
    corr_mats: Dict[str, pd.DataFrame],
    figsize: Tuple[int, int] = (14, 4),
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    将 x/y/z 三个方向的 5×5 相关性矩阵画成热力图

    参数:
        corr_mats: compute_frequency_correlation_matrices 的返回值
        figsize: 单个子图大小 (宽, 高)，总图大小会按子图数量自动扩展
        save_path: 保存图片路径（可选），为 None 时不保存
        show_plot: 是否在屏幕上显示
    """
    if not corr_mats:
        print("警告：corr_mats 为空，无法绘制热力图")
        return

    # 设置统一的科研风格，与温度脚本保持一致
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'axes.linewidth': 1.2,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.minor.width': 0.8,
        'ytick.minor.width': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'figure.dpi': 100
    })

    axes_labels = {'x': 'Frequency X', 'y': 'Frequency Y', 'z': 'Frequency Z'}
    available_axes = [ax for ax in ['x', 'y', 'z'] if ax in corr_mats]

    n_axes = len(available_axes)
    if n_axes == 0:
        print("警告：corr_mats 中没有 x/y/z 任何一个方向的矩阵，无法绘图")
        return

    fig, axs = plt.subplots(
        1, n_axes,
        figsize=(figsize[0] * n_axes, figsize[1]),
        squeeze=False
    )
    axs = axs[0]

    im = None
    for i, axis in enumerate(available_axes):
        ax = axs[i]
        corr = corr_mats[axis]

        im = ax.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1, origin='lower')

        # 坐标刻度与标签
        det_labels = [str(int(d)) for d in corr.index]
        ax.set_xticks(range(len(det_labels)))
        ax.set_yticks(range(len(det_labels)))
        ax.set_xticklabels(det_labels)
        ax.set_yticklabels(det_labels)

        ax.set_xlabel('Detector')
        ax.set_ylabel('Detector')
        ax.set_title(f'{axes_labels.get(axis, axis)} Correlation')

        # 网格线
        ax.set_xticks(np.arange(-0.5, len(det_labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(det_labels), 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

        # 在每个格子里写上数值
        for r in range(corr.shape[0]):
            for c in range(corr.shape[1]):
                val = corr.values[r, c]
                ax.text(
                    c, r, f'{val:.2f}',
                    ha='center', va='center',
                    color='black' if abs(val) < 0.7 else 'white',
                    fontsize=9
                )

        # 坐标轴边框
        for spine in ax.spines.values():
            spine.set_color('black')

    # 加一个统一的 colorbar
    if im is not None:
        cbar = fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Correlation Coefficient', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'相关性矩阵热力图已保存至: {save_path}')

    if show_plot:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':
    # 获取项目根目录
    # 从 python/data/sensor/vibration/frequency/correlationship.py 向上5层到达项目根目录
    # correlationship -> frequency -> vibration -> sensor -> data -> python -> DeepVibration
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(current_dir)
                )
            )
        )
    )
    data_dir = os.path.join(project_root, 'data', 'vibration', 'hdf5')

    print('=' * 70)
    print('振动传感器频率时间相关性分析')
    print('=' * 70)

    # 默认分析 5 个探测器，时间范围与其他示例保持一致
    detectors = [1, 2, 3, 4, 5]
    print(f'\n按日期范围筛选数据，探测器: {detectors}')
    print('-' * 70)

    try:
        data_multi = select_by_date_range_vibration(
            data_dir,
            detector_num=detectors,
            start_date='2025-05-28',
            end_date='2025-06-10',
            downsample_factor=100  # 可根据需要调整
        )

        if not data_multi:
            print('警告：未能读取到任何数据，无法进行相关性分析')
        else:
            print('\n开始计算频率相关性矩阵 (5×5)...')
            corr_mats = compute_frequency_correlation_matrices(data_multi, detectors=detectors)

            for axis_key, corr_df in corr_mats.items():
                print('\n' + '=' * 30)
                print(f'{axis_key.upper()} 方向频率相关性矩阵 (5×5):')
                print('=' * 30)
                print(corr_df)

            # 绘制热力图
            print('\n绘制相关性矩阵热力图...')
            save_png = os.path.join(project_root, 'imgaes', 'frequency_correlation_matrices.png')
            os.makedirs(os.path.dirname(save_png), exist_ok=True)
            plot_frequency_correlation_heatmaps(
                corr_mats,
                save_path=save_png,
                show_plot=True
            )

    except Exception as e:
        print(f'错误: {e}')
        import traceback
        traceback.print_exc()
