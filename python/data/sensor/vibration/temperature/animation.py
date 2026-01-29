#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
振动传感器温度动态变化动画生成脚本
读取5个传感器的温度数据，在2D正方形中布置，使用自然邻域插值生成温度分布图，并创建GIF动画
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合多进程
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import Voronoi
from scipy.interpolate import griddata
import imageio
from typing import Dict, Optional, Tuple, List
import importlib.util
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# 尝试导入PyTorch用于GPU加速（可选）
try:
    import torch
    TORCH_AVAILABLE = True
    # 检测GPU可用性
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        USE_GPU = True
    else:
        DEVICE = torch.device('cpu')
        USE_GPU = False
except ImportError:
    TORCH_AVAILABLE = False
    USE_GPU = False
    DEVICE = None

# 导入select.py模块
current_dir = os.path.dirname(os.path.abspath(__file__))
select_path = os.path.join(current_dir, 'select.py')
spec = importlib.util.spec_from_file_location("select", select_path)
select_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(select_module)
select_by_date_range_vibration = select_module.select_by_date_range_vibration


def get_sensor_positions(square_size: float = 1100.0) -> Dict[int, Tuple[float, float]]:
    """
    定义5个传感器在正方形中的位置
    
    参数:
        square_size: 正方形边长，默认1100
    
    返回:
        字典，键为传感器编号，值为(x, y)坐标元组
    """
    positions = {
        1: (0.0, 0.0),                    # 左下角
        2: (square_size / 2, 0.0),        # 下边中点
        3: (square_size, 0.0),            # 右下角
        4: (0.0, square_size),            # 左上角
        5: (square_size, square_size)    # 右上角
    }
    return positions

def _generate_single_frame(args: Tuple) -> Tuple[int, np.ndarray]:
    """
    生成单帧的辅助函数（用于多进程并行处理）
    
    参数:
        args: 包含所有必要参数的元组
    
    返回:
        (frame_idx, frame_array): 帧索引和帧图像数组
    """
    try:
        (frame_idx, idx, datetime_arr, temp_arr, detector_num_arr, 
         sensor_positions, grid_x, grid_y, square_size_val, temp_min, temp_max,
         grid_resolution) = args
        
        # 确保matplotlib使用非交互式后端（在多进程环境中很重要）
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # 创建新的图形（每个进程独立）
        fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
        ax.set_xlim(0, square_size_val)
        ax.set_ylim(0, square_size_val)
        ax.set_aspect('equal')
        ax.set_xlabel('X (mm)', fontsize=12)
        ax.set_ylabel('Y (mm)', fontsize=12)
        ax.set_title('Temperature Distribution Animation', fontsize=14, fontweight='bold')
        
        # 绘制传感器位置
        for det_num, (x_pos, y_pos) in sensor_positions.items():
            ax.plot(x_pos, y_pos, 'ko', markersize=10, markeredgewidth=2, markeredgecolor='white')
            ax.text(x_pos, y_pos - square_size_val*0.02, f'Det {det_num}', 
                    ha='center', va='top', fontsize=10, fontweight='bold', color='black',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.8))
        
        # 从时间戳恢复datetime对象
        current_timestamp = datetime_arr[idx]
        current_time = pd.Timestamp.fromtimestamp(current_timestamp)
        
        # 获取当前时间点的传感器温度
        # 使用时间戳进行比较（更可靠）
        time_mask = np.abs(datetime_arr - current_timestamp) < 1.0  # 1秒容差
        if not np.any(time_mask):
            time_diff = np.abs(datetime_arr - current_timestamp)
            closest_idx = np.argmin(time_diff)
            time_mask = np.zeros(len(datetime_arr), dtype=bool)
            time_mask[closest_idx] = True
        
        sensor_temps = {}
        for det_num in [1, 2, 3, 4, 5]:
            det_mask = (detector_num_arr == det_num) & time_mask
            if np.any(det_mask):
                sensor_temps[det_num] = temp_arr[det_mask][0]
            else:
                det_only_mask = detector_num_arr == det_num
                if np.any(det_only_mask):
                    det_indices = np.where(det_only_mask)[0]
                    time_diffs = np.abs(datetime_arr[det_indices] - current_timestamp)
                    closest_det_idx = det_indices[np.argmin(time_diffs)]
                    sensor_temps[det_num] = temp_arr[closest_det_idx]
                else:
                    sensor_temps[det_num] = np.nanmean(temp_arr[detector_num_arr == det_num]) if np.any(detector_num_arr == det_num) else temp_min
        
        # 进行插值
        interpolated_temp = natural_neighbor_interpolation(
            sensor_positions, sensor_temps, grid_x, grid_y, square_size_val
        )
        
        # 绘制温度分布
        cmap = plt.cm.jet
        im = ax.imshow(interpolated_temp, 
                       extent=[0, square_size_val, 0, square_size_val],
                       origin='lower', cmap=cmap, 
                       vmin=temp_min, vmax=temp_max,
                       interpolation='bilinear')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, label='Temperature (°C)', shrink=0.8)
        cbar.ax.tick_params(labelsize=10)
        
        # 添加时间文本
        time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
        ax.text(0.02, 0.98, f'Time: {time_str}', transform=ax.transAxes,
               fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                        edgecolor='black', alpha=0.8))
        
        # 转换为图像数组
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)  # 关闭图形以释放内存
        
        return (frame_idx, frame)
    except Exception as e:
        import traceback
        frame_idx_val = args[0] if 'args' in locals() else 0
        error_msg = f"生成第 {frame_idx_val} 帧时出错: {e}\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        # 返回一个黑色帧作为占位符
        placeholder_frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
        return (frame_idx_val, placeholder_frame)

def _gpu_interpolation(points: np.ndarray, values: np.ndarray, 
                       grid_points: np.ndarray, power: float = 2.0) -> np.ndarray:
    """
    使用GPU加速的逆距离加权（IDW）插值
    
    参数:
        points: 传感器位置数组 (N, 2)
        values: 传感器温度值数组 (N,)
        grid_points: 网格点数组 (M, 2)
        power: 距离权重幂次，默认2.0
    
    返回:
        插值后的温度数组 (M,)
    """
    if not TORCH_AVAILABLE or not USE_GPU:
        raise RuntimeError("GPU不可用，应使用CPU插值")
    
    # 转换为PyTorch张量
    points_t = torch.from_numpy(points).float().to(DEVICE)  # (N, 2)
    values_t = torch.from_numpy(values).float().to(DEVICE)  # (N,)
    grid_t = torch.from_numpy(grid_points).float().to(DEVICE)  # (M, 2)
    
    # 计算每个网格点到所有传感器点的距离
    # grid_t: (M, 2), points_t: (N, 2)
    # 使用广播计算所有距离: (M, N)
    diff = grid_t.unsqueeze(1) - points_t.unsqueeze(0)  # (M, N, 2)
    distances = torch.norm(diff, dim=2)  # (M, N)
    
    # 避免除零，添加小值
    epsilon = 1e-10
    distances = torch.clamp(distances, min=epsilon)
    
    # 逆距离加权
    weights = 1.0 / (distances ** power)  # (M, N)
    weights_sum = weights.sum(dim=1, keepdim=True)  # (M, 1)
    
    # 加权平均
    weighted_sum = (weights * values_t.unsqueeze(0)).sum(dim=1)  # (M,)
    interpolated = weighted_sum / weights_sum.squeeze(1)  # (M,)
    
    # 转换回numpy
    return interpolated.cpu().numpy()


def natural_neighbor_interpolation(sensor_positions: Dict[int, Tuple[float, float]],
                                   sensor_temps: Dict[int, float],
                                   grid_x: np.ndarray,
                                   grid_y: np.ndarray,
                                   square_size: float = 1100.0,
                                   use_gpu: Optional[bool] = None) -> np.ndarray:
    """
    使用自然邻域插值（Natural Neighbor Interpolation）生成温度分布
    
    参数:
        sensor_positions: 传感器位置字典
        sensor_temps: 传感器温度字典
        grid_x: 网格x坐标数组
        grid_y: 网格y坐标数组
        square_size: 正方形边长
        use_gpu: 是否使用GPU加速（None表示自动检测）
    
    返回:
        插值后的温度分布数组
    """
    # 准备插值点
    points = np.array([sensor_positions[i] for i in sorted(sensor_positions.keys())])
    values = np.array([sensor_temps[i] for i in sorted(sensor_positions.keys())])
    
    # 创建网格点
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    
    # 决定是否使用GPU
    if use_gpu is None:
        use_gpu = USE_GPU  # 使用全局设置
    
    # 如果GPU可用且网格分辨率较大，使用GPU加速
    if use_gpu and TORCH_AVAILABLE and USE_GPU and len(grid_points) > 10000:
        try:
            # 使用GPU加速的IDW插值
            interpolated = _gpu_interpolation(points, values, grid_points, power=2.0)
            return interpolated.reshape(grid_x.shape)
        except Exception as e:
            # GPU失败时回退到CPU
            print(f"GPU插值失败，回退到CPU: {e}", file=sys.stderr)
            use_gpu = False
    
    # CPU插值（原有方法）
    # 使用自然邻域插值（使用'cubic'或'linear'作为近似，因为scipy没有直接的自然邻域插值）
    # 这里使用'cubic'插值方法，它会产生平滑的结果
    interpolated = griddata(points, values, grid_points, method='cubic', fill_value=np.nan)
    
    # 如果cubic方法产生NaN，使用linear方法填充
    nan_mask = np.isnan(interpolated)
    if np.any(nan_mask):
        linear_interp = griddata(points, values, grid_points[nan_mask], method='linear', fill_value=np.nan)
        interpolated[nan_mask] = linear_interp
    
    # 如果还有NaN，使用最近邻方法填充
    nan_mask = np.isnan(interpolated)
    if np.any(nan_mask):
        nearest_interp = griddata(points, values, grid_points[nan_mask], method='nearest')
        interpolated[nan_mask] = nearest_interp
    
    return interpolated.reshape(grid_x.shape)

def create_temperature_animation(data_dict: Dict[str, np.ndarray],
                                  square_size: float = 1100.0,
                                  grid_resolution: int = 100,
                                  frame_interval: int = 10,
                                  fps: int = 10,
                                  save_path: Optional[str] = None,
                                  show_animation: bool = False,
                                  downsample_frames: int = 1) -> None:
    """
    创建温度动态变化动画
    
    参数:
        data_dict: 包含数据的字典，必须包含 'datetime', 'Temperature', 'detector_num' 列
        square_size: 正方形边长，默认1100
        grid_resolution: 网格分辨率（网格点数），默认100
        frame_interval: 帧间隔（每隔N个数据点取一帧），默认10
        fps: 动画帧率，默认10
        save_path: 保存GIF的路径，如果为None则不保存
        show_animation: 是否显示动画窗口
        downsample_frames: 降采样帧数（每隔N帧取一帧），默认1（不降采样）
    """
    if 'datetime' not in data_dict:
        raise ValueError("数据字典中必须包含 'datetime' 列")
    if 'Temperature' not in data_dict:
        raise ValueError("数据字典中必须包含 'Temperature' 列")
    if 'detector_num' not in data_dict:
        raise ValueError("数据字典中必须包含 'detector_num' 列")
    
    # 获取数据
    datetime_arr = data_dict['datetime']
    temp_arr = data_dict['Temperature']
    detector_num_arr = data_dict['detector_num']
    
    # 过滤NaN值
    valid_mask = ~np.isnan(temp_arr)
    datetime_arr = datetime_arr[valid_mask]
    temp_arr = temp_arr[valid_mask]
    detector_num_arr = detector_num_arr[valid_mask]
    
    if len(datetime_arr) == 0:
        raise ValueError("没有有效数据")
    
    # 获取传感器位置
    sensor_positions = get_sensor_positions(square_size)
    
    # 创建网格
    x = np.linspace(0, square_size, grid_resolution)
    y = np.linspace(0, square_size, grid_resolution)
    grid_x, grid_y = np.meshgrid(x, y)
    
    # 按时间排序并选择帧
    time_indices = np.arange(0, len(datetime_arr), frame_interval)
    if downsample_frames > 1:
        time_indices = time_indices[::downsample_frames]
    
    print(f'总数据点: {len(datetime_arr)}, 将生成 {len(time_indices)} 帧动画')
    
    # 计算温度范围（用于设置颜色映射）
    temp_min = np.nanmin(temp_arr)
    temp_max = np.nanmax(temp_arr)
    temp_range = temp_max - temp_min
    if temp_range == 0:
        temp_range = 1.0
    
    # 创建颜色映射
    cmap = plt.cm.jet  # 可以使用其他colormap，如 'coolwarm', 'viridis', 'plasma' 等
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, square_size)
    ax.set_ylim(0, square_size)
    ax.set_aspect('equal')
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.set_title('Temperature Distribution Animation', fontsize=14, fontweight='bold')
    
    # 绘制传感器位置
    for det_num, (x_pos, y_pos) in sensor_positions.items():
        ax.plot(x_pos, y_pos, 'ko', markersize=10, markeredgewidth=2, markeredgecolor='white')
        ax.text(x_pos, y_pos - square_size*0.02, f'Det {det_num}', 
                ha='center', va='top', fontsize=10, fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.8))
    
    # 初始化图像对象
    im = ax.imshow(np.zeros((grid_resolution, grid_resolution)), 
                   extent=[0, square_size, 0, square_size],
                   origin='lower', cmap=cmap, 
                   vmin=temp_min, vmax=temp_max,
                   interpolation='bilinear')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, label='Temperature (°C)', shrink=0.8)
    cbar.ax.tick_params(labelsize=10)
    
    # 添加时间文本
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                                edgecolor='black', alpha=0.8))
    
    # 如果只需要保存GIF，使用多进程并行生成帧
    if save_path:
        print(f'\n使用多进程并行生成 {len(time_indices)} 帧...')
        
        # 确定并行进程数（使用所有可用CPU核心以最大化计算利用率）
        max_workers = os.cpu_count() or 1  # 使用所有CPU核心
        print(f'使用 {max_workers} 个CPU核心进行并行处理')
        
        # 将datetime数组转换为可序列化的格式（numpy datetime64在多进程间可能有问题）
        # 转换为时间戳（秒）以便在多进程间传递
        datetime_timestamps = np.array([pd.Timestamp(dt).timestamp() for dt in datetime_arr])
        
        # 准备参数列表
        frame_args = []
        for frame_idx, idx in enumerate(time_indices):
            args = (frame_idx, idx, datetime_timestamps, temp_arr, detector_num_arr,
                   sensor_positions, grid_x, grid_y, square_size, temp_min, temp_max,
                   grid_resolution)
            frame_args.append(args)
        
        # 使用多进程并行生成帧
        frames_dict = {}
        completed = 0
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_generate_single_frame, args): args[0] for args in frame_args}
            
            for future in as_completed(futures):
                try:
                    frame_idx, frame = future.result()
                    frames_dict[frame_idx] = frame
                    completed += 1
                    if completed % 10 == 0 or completed == len(time_indices):
                        print(f'  已生成 {completed}/{len(time_indices)} 帧')
                except Exception as e:
                    original_idx = futures[future]
                    print(f'  错误：生成第 {original_idx} 帧时出错: {e}')
        
        # 按帧索引排序
        frames = [frames_dict[i] for i in sorted(frames_dict.keys())]
        
        print(f'\n正在保存GIF动画到: {save_path}')
        print(f'共 {len(frames)} 帧，帧率: {fps} fps')
        
        # 使用imageio保存GIF
        imageio.mimsave(save_path, frames, fps=fps, loop=0)
        print(f'GIF动画已保存')
        
        plt.close(fig)
        return None
    
    # 如果需要显示动画，使用传统方式
    if show_animation:
        def animate(frame_idx):
            """动画更新函数"""
            if frame_idx >= len(time_indices):
                return
            
            idx = time_indices[frame_idx]
            current_time = datetime_arr[idx]
            
            time_mask = datetime_arr == current_time
            if not np.any(time_mask):
                time_diff = np.abs((datetime_arr - current_time).astype('timedelta64[s]').astype(float))
                closest_idx = np.argmin(time_diff)
                time_mask = np.zeros(len(datetime_arr), dtype=bool)
                time_mask[closest_idx] = True
            
            sensor_temps = {}
            for det_num in [1, 2, 3, 4, 5]:
                det_mask = (detector_num_arr == det_num) & time_mask
                if np.any(det_mask):
                    sensor_temps[det_num] = temp_arr[det_mask][0]
                else:
                    det_only_mask = detector_num_arr == det_num
                    if np.any(det_only_mask):
                        det_indices = np.where(det_only_mask)[0]
                        time_diffs = np.abs((datetime_arr[det_indices] - current_time).astype('timedelta64[s]').astype(float))
                        closest_det_idx = det_indices[np.argmin(time_diffs)]
                        sensor_temps[det_num] = temp_arr[closest_det_idx]
                    else:
                        sensor_temps[det_num] = np.nanmean(temp_arr[detector_num_arr == det_num]) if np.any(detector_num_arr == det_num) else temp_min
            
            interpolated_temp = natural_neighbor_interpolation(
                sensor_positions, sensor_temps, grid_x, grid_y, square_size
            )
            
            im.set_array(interpolated_temp)
            time_str = pd.to_datetime(current_time).strftime('%Y-%m-%d %H:%M:%S')
            time_text.set_text(f'Time: {time_str}')
        
        anim = animation.FuncAnimation(fig, animate, frames=len(time_indices), 
                                       interval=1000/fps, repeat=True, blit=False)
        plt.show()
        return anim
    else:
        plt.close(fig)
        return None

# 示例使用
if __name__ == '__main__':
    # 获取项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))))
    data_dir = os.path.join(project_root, 'data', 'vibration', 'hdf5')
    
    print('=' * 70)
    print('振动传感器温度动态变化动画生成')
    print('=' * 70)
    
    try:
        # 读取5个传感器的数据
        print('\n读取传感器数据...')
        data = select_by_date_range_vibration(
            data_dir,
            detector_num=[1, 2, 3, 4, 5],
            start_date='2025-05-28',
            end_date='2025-06-10',
            downsample_factor=100  # 降采样以减少数据量
        )
        
        if not data:
            print('错误：未能读取到数据')
            sys.exit(1)
        
        print(f'读取完成：共 {len(data["datetime"])} 个数据点')
        
        # 创建动画
        print('\n创建温度分布动画...')
        output_path = os.path.join(project_root, 'data', 'vibration', 'temperature_animation.gif')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        create_temperature_animation(
            data,
            square_size=1100.0,
            grid_resolution=10,
            frame_interval=5000,  # 每隔50个数据点取一帧
            fps=50,  # 帧率5fps
            save_path=output_path,
            show_animation=False,
            downsample_frames=1
        )
        
        print(f'\n动画已保存到: {output_path}')
        
    except Exception as e:
        print(f'错误: {e}')
        import traceback
        traceback.print_exc()
