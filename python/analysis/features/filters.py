import numpy as np
from typing import Sequence, Union

ArrayLike = Union[Sequence[float], np.ndarray]


def median_filter(x: ArrayLike, kernel_size: int = 3) -> np.ndarray:
    """
    对一维时序信号进行中值滤波。

    Parameters
    ----------
    x : Sequence[float] or np.ndarray
        输入的一维时序信号。
    kernel_size : int, optional
        滤波窗口长度（必须为正奇数），默认 3。

    Returns
    -------
    np.ndarray
        中值滤波后的信号（长度与输入相同）。
    """
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size 必须为正奇数，例如 3、5、7 ...")

    x_arr = np.asarray(x, dtype=float)
    n = x_arr.size
    if n == 0:
        return x_arr

    pad = kernel_size // 2
    # 端点采用边界延拓（repeat）方式填充
    x_padded = np.pad(x_arr, pad_width=pad, mode="edge")

    result = np.empty_like(x_arr)
    for i in range(n):
        window = x_padded[i : i + kernel_size]
        result[i] = np.median(window)

    return result

