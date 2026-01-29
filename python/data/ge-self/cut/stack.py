#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
堆叠显示未过阈值的锗自触发事件的 CH0 波形。

逻辑：
1. 复用 overthreshold.py 中的 select_physical_events_no_overthreshold，先筛选：
   - 既非 RT 也非 Inhibit 的 Physical 事件
   - 且 CH0 最大值 <= 阈值（不过阈值）
2. 只读取这些事件的 CH0 波形。
3. 使用 matplotlib 交互：
   - 初始显示第 1 个事件的 CH0 波形（不透明，alpha=1.0）
   - 每按一次空格：
       * 之前所有波形的透明度调低（例如 alpha=0.2）
       * 在同一张图上叠加下一个事件的 CH0 波形（alpha=1.0）
   - 所有波形始终叠加在一幅图上，最新事件最清晰。
"""

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import importlib.util


# -----------------------------------------------------------------------------
# 导入 overthreshold.py 中的筛选函数
# -----------------------------------------------------------------------------

current_dir = os.path.dirname(os.path.abspath(__file__))

overthreshold_path = os.path.join(current_dir, "overthreshold.py")
spec_over = importlib.util.spec_from_file_location("overthreshold_module", overthreshold_path)
overthreshold_module = importlib.util.module_from_spec(spec_over)
spec_over.loader.exec_module(overthreshold_module)

select_physical_events_no_overthreshold = overthreshold_module.select_physical_events_no_overthreshold


class CH0StackPlotter:
    """
    使用空格键交互堆叠显示 CH0 波形：
    - space: 显示下一个事件，并把之前的波形变得更透明
    """

    def __init__(
        self,
        ch0_3_file: str,
        ch0_indices: np.ndarray,
        ch0_idx: int = 0,
        max_events: Optional[int] = None,
    ) -> None:
        self.ch0_3_file = ch0_3_file
        self.ch0_indices = np.asarray(ch0_indices, dtype=np.int64)
        self.ch0_idx = int(ch0_idx)

        if self.ch0_indices.size == 0:
            raise ValueError("没有任何可用于堆叠显示的事件索引（selected_indices 为空）")

        # 限制最多显示的事件数量
        if max_events is not None and max_events > 0:
            self.ch0_indices = self.ch0_indices[:max_events]

        # 读取 CH0 波形
        with h5py.File(self.ch0_3_file, "r") as f_ch0:
            ch0_channel_data = f_ch0["channel_data"]
            time_samples, num_channels, num_events = ch0_channel_data.shape

            if self.ch0_idx >= num_channels:
                raise ValueError(
                    f"ch0_idx={self.ch0_idx} 超过通道数 {num_channels}，无法读取 CH0 波形"
                )

            # 只取需要的事件
            waveforms = ch0_channel_data[:, self.ch0_idx, self.ch0_indices].astype(
                np.float64
            )

        # waveforms 形状: (time_samples, n_events)
        self.waveforms = waveforms
        self.n_events = waveforms.shape[1]
        self.time_samples = waveforms.shape[0]

        # 时间轴（与 overthreshold.py 保持一致：采样间隔 4 ns，单位 μs）
        sampling_interval_ns = 4.0
        self.time_axis_us = (
            np.arange(self.time_samples) * sampling_interval_ns / 1000.0
        )

        # matplotlib 对象
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.lines = []  # 已绘制的 Line2D 对象
        self.current_idx = 0

        # 初始化第一条波形
        self._plot_first_waveform()

        # 注册键盘事件
        self.cid_key = self.fig.canvas.mpl_connect(
            "key_press_event", self._on_key_press
        )

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _plot_first_waveform(self) -> None:
        """绘制第一条事件的 CH0 波形。"""
        wf = self.waveforms[:, 0]
        line, = self.ax.plot(
            self.time_axis_us,
            wf,
            color="C0",
            linewidth=1.0,
            alpha=1.0,
            label=f"Event #{self.ch0_indices[0]}",
        )
        self.lines.append(line)

        self.ax.set_xlabel("Time (μs)", fontsize=12)
        self.ax.set_ylabel("Amplitude (ADC)", fontsize=12)
        self.ax.set_title(
            f"CH0 Waveform Stack (Non-overthreshold Physical Events)\n"
            f"Total events: {self.n_events}, Current: 1 / {self.n_events}",
            fontsize=13,
        )
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc="upper right", fontsize=9)
        self.fig.tight_layout()

        # 设置合理 y 轴范围（基于第一条波形）
        y_min = np.min(wf)
        y_max = np.max(wf)
        data_range = y_max - y_min
        if data_range > 0:
            margin = data_range * 0.1
            y_min -= margin
            y_max += margin
        else:
            center = (y_min + y_max) / 2.0
            margin = max(abs(center) * 0.1, 100.0)
            y_min = center - margin
            y_max = center + margin
        self.ax.set_ylim(y_min, y_max)

    def _on_key_press(self, event) -> None:
        """键盘事件：空格键切换到下一个事件，并堆叠显示。"""
        if event.key != " ":
            return

        if self.current_idx >= self.n_events - 1:
            print("已经到达最后一个事件，不再增加新的波形。")
            return

        self.current_idx += 1
        idx = self.current_idx

        # 将已有波形变得更透明
        for line in self.lines:
            line.set_alpha(0.2)

        # 绘制新的波形，alpha=1.0
        wf = self.waveforms[:, idx]
        line, = self.ax.plot(
            self.time_axis_us,
            wf,
            color="C0",
            linewidth=1.0,
            alpha=1.0,
            label=f"Event #{self.ch0_indices[idx]}",
        )
        self.lines.append(line)

        # 更新标题
        self.ax.set_title(
            f"CH0 Waveform Stack (Non-overthreshold Physical Events)\n"
            f"Total events: {self.n_events}, Current: {idx + 1} / {self.n_events}",
            fontsize=13,
        )

        # 只保留一个图例（使用最后一条线的 label）
        self.ax.legend([line], [line.get_label()], loc="upper right", fontsize=9)

        self.fig.canvas.draw_idle()


def stack_ch0_waveforms_no_overthreshold(
    ch0_3_file: Optional[str] = None,
    ch5_file: Optional[str] = None,
    rt_cut: float = 6000.0,
    ch0_threshold: float = 16382.0,
    ch0_idx: int = 0,
    ch5_idx: int = 0,
    max_events_to_use: Optional[int] = None,
) -> None:
    """
    入口函数：
    1. 调用 select_physical_events_no_overthreshold 筛选不过阈值的 Physical 事件。
    2. 只取这些事件的 CH0 波形。
    3. 交互式堆叠显示（按空格切换下一个事件）。
    """
    print("=" * 70)
    print("堆叠显示未过阈值的锗自触发事件的 CH0 波形")
    print("=" * 70)

    # 先筛选不过阈值的 Physical 事件
    selection_result = select_physical_events_no_overthreshold(
        ch0_3_file=ch0_3_file,
        ch5_file=ch5_file,
        rt_cut=rt_cut,
        ch0_threshold=ch0_threshold,
        ch0_idx=ch0_idx,
        ch5_idx=ch5_idx,
    )

    ch0_3_file = selection_result["ch0_3_file"]
    selected_indices = selection_result["selected_indices"]
    final_physical_count = selection_result["final_physical_count"]

    if final_physical_count == 0 or selected_indices.size == 0:
        print("未发现不过阈值的 Physical 信号，无法进行堆叠显示。")
        return

    print(
        f"\n共找到 {final_physical_count} 个不过阈值的 Physical 事件，"
        f"将用于 CH0 波形堆叠显示。"
    )

    # 创建交互式堆叠显示对象
    plotter = CH0StackPlotter(
        ch0_3_file=ch0_3_file,
        ch0_indices=selected_indices,
        ch0_idx=ch0_idx,
        max_events=max_events_to_use,
    )

    print("\n操作说明：")
    print("  - 空格键（Space）：叠加显示下一个事件的 CH0 波形，并让之前的波形变得透明。")
    print("  - 关闭窗口或 Ctrl+C：结束程序。")

    plt.show()


if __name__ == "__main__":
    try:
        # 示例：自动选择文件对，使用默认阈值，最多堆叠显示前 50 个事件
        stack_ch0_waveforms_no_overthreshold(
            ch0_3_file=None,
            ch5_file=None,
            rt_cut=6000.0,
            ch0_threshold=16382.0,
            ch0_idx=0,
            ch5_idx=0,
            max_events_to_use=10000,
        )
    except Exception as e:
        print(f"堆叠显示失败: {e}")
        import traceback

        traceback.print_exc()
