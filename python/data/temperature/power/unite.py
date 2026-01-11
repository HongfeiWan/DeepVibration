#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
振动传感器和制冷机温度数据联合绘制脚本
在同一幅图中同时绘制振动传感器温度和制冷机温度曲线
"""
import os
import sys
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from typing import Optional, Tuple

# 添加父目录到路径，以便导入其他模块
current_dir = os.path.dirname(os.path.abspath(__file__))
# current_dir = python/data/temperature/power
parent_dir = os.path.dirname(current_dir)  # python/data/temperature
grandparent_dir = os.path.dirname(parent_dir)  # python/data
great_grandparent_dir = os.path.dirname(grandparent_dir)  # python
project_root = os.path.dirname(great_grandparent_dir)  # 项目根目录

# 导入振动传感器温度模块
# 路径: python/data/sensor/vibration/temperature/select.py
vibration_temp_select_path = os.path.join(great_grandparent_dir, 'data', 'sensor', 'vibration', 'temperature', 'select.py')
spec_vibration = importlib.util.spec_from_file_location("vibration_temp_select", vibration_temp_select_path)
vibration_temp_select = importlib.util.module_from_spec(spec_vibration)
spec_vibration.loader.exec_module(vibration_temp_select)
select_by_date_range_vibration = vibration_temp_select.select_by_date_range_vibration

# 导入制冷机模块
# 路径: python/data/compressor/select.py
compressor_select_path = os.path.join(great_grandparent_dir, 'data', 'compressor', 'select.py')
spec_compressor = importlib.util.spec_from_file_location("compressor_select", compressor_select_path)
compressor_select = importlib.util.module_from_spec(spec_compressor)
spec_compressor.loader.exec_module(compressor_select)
select_compressor_by_date_range = compressor_select.select_by_date_range


def plot_united_temperature(vibration_data_dir: str,
                           compressor_file_path: str,
                           detector_num: int = 2,
                           start_date: str = None,
                           end_date: str = None,
                           start_time: Optional[str] = None,
                           end_time: Optional[str] = None,
                           compressor_temp_column = 'Compressor temp',
                           save_path: Optional[str] = None,
                           show_plot: bool = True,
                           figsize: Tuple[int, int] = (14, 7)) -> None:
    """
    在同一幅图中绘制振动传感器温度和制冷机温度
    
    参数:
        vibration_data_dir: 振动传感器HDF5数据文件夹路径
        compressor_file_path: 制冷机数据文件路径
        detector_num: 探测器编号，默认为2
        start_date: 起始日期，格式 'YYYY-MM-DD'
        end_date: 终止日期，格式 'YYYY-MM-DD'
        start_time: 起始时间（可选），格式 'HH:MM:SS'
        end_time: 终止时间（可选），格式 'HH:MM:SS'
        compressor_temp_column: 要绘制的制冷机温度列名，可以是字符串或列表
            例如: 'Compressor temp' 或 ['Compressor temp', 'Controller temp', 'Coldhead temp']
        save_path: 保存图片的路径，如果为None则不保存
        show_plot: 是否显示图片
        figsize: 图片大小 (宽度, 高度)
    """
    # 将单个字符串转换为列表
    if isinstance(compressor_temp_column, str):
        compressor_temp_columns = [compressor_temp_column]
    elif isinstance(compressor_temp_column, (list, tuple)):
        compressor_temp_columns = list(compressor_temp_column)
    else:
        raise ValueError(f"compressor_temp_column 必须是字符串或列表，当前类型: {type(compressor_temp_column)}")
    # 读取振动传感器数据
    print('=' * 70)
    print('读取振动传感器数据...')
    print('-' * 70)
    vibration_data = select_by_date_range_vibration(
        vibration_data_dir,
        detector_num=detector_num,
        start_date=start_date,
        end_date=end_date,
        start_time=start_time,
        end_time=end_time
    )
    
    if not vibration_data:
        raise ValueError('未能读取到振动传感器数据')
    
    # 读取制冷机数据
    print('\n' + '=' * 70)
    print('读取制冷机数据...')
    print('-' * 70)
    compressor_data = select_compressor_by_date_range(
        compressor_file_path,
        start_date=start_date,
        end_date=end_date,
        start_time=start_time,
        end_time=end_time
    )
    
    if not compressor_data:
        raise ValueError('未能读取到制冷机数据')
    
    # 获取振动传感器温度数据
    vibration_datetime = vibration_data['datetime']
    vibration_temp = vibration_data['Temperature']
    
    # 过滤NaN值和温度小于20的点（与select.py保持一致）
    vibration_valid_mask = (~np.isnan(vibration_temp)) & (vibration_temp >= 20.0)
    vibration_datetime = vibration_datetime[vibration_valid_mask]
    vibration_temp = vibration_temp[vibration_valid_mask]
    
    if len(vibration_datetime) == 0:
        raise ValueError("振动传感器温度数据中没有有效数据（过滤后）")
    
    # 获取制冷机温度数据（支持多个温度列）
    compressor_data_list = []
    compressor_datetime_all = compressor_data['datetime']
    
    # 定义marker列表，用于区分不同的compressor温度部分
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'X', '<', '>']
    
    # 温度列名到简短标签的映射
    label_map = {
        'Compressor temp': 'Compressor',
        'Controller temp': 'Controller',
        'Coldhead temp': 'Coldhead',
        'Coldtip temp': 'Coldtip',
        'Coldtip setp': 'Coldtip Setpoint'
    }
    
    for i, col_name in enumerate(compressor_temp_columns):
        if col_name not in compressor_data:
            raise ValueError(f"制冷机数据中不包含 '{col_name}' 列")
        
        compressor_temp = compressor_data[col_name]
        
        # 过滤NaN值
        compressor_valid_mask = ~np.isnan(compressor_temp)
        compressor_datetime = compressor_datetime_all[compressor_valid_mask]
        compressor_temp_filtered = compressor_temp[compressor_valid_mask]
        
        if len(compressor_datetime) == 0:
            print(f"警告：制冷机 '{col_name}' 列中没有有效数据，跳过")
            continue
        
        compressor_data_list.append({
            'column': col_name,
            'label': label_map.get(col_name, col_name),
            'datetime': compressor_datetime,
            'temperature': compressor_temp_filtered,
            'marker': markers[i % len(markers)]
        })
    
    if len(compressor_data_list) == 0:
        raise ValueError("没有有效的制冷机温度数据")
    
    # 获取制冷机功率数据（Cooler power）
    compressor_power = None
    compressor_power_datetime = None
    if 'Cooler power' in compressor_data:
        compressor_power = compressor_data['Cooler power']
        compressor_power_datetime = compressor_datetime_all
        
        # 过滤NaN值
        power_valid_mask = ~np.isnan(compressor_power)
        compressor_power_datetime = compressor_power_datetime[power_valid_mask]
        compressor_power = compressor_power[power_valid_mask]
        
        if len(compressor_power_datetime) == 0:
            print("警告：制冷机 'Cooler power' 列中没有有效数据，将不显示功率曲线")
            compressor_power = None
    else:
        print("警告：制冷机数据中不包含 'Cooler power' 列，将不显示功率曲线")
    
    # 设置matplotlib参数以获得更好的科研图表样式
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
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'figure.dpi': 100
    })
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制振动传感器温度（蓝色）
    ax.plot(vibration_datetime, vibration_temp,
            linewidth=1.5,
            alpha=0.85,
            color='#2E86AB',
            label='Vibration Sensor Temperature')
    
    # 绘制制冷机温度（使用相同颜色，不同marker区分）
    compressor_color = '#C73E1D'  # 统一的红色
    
    for compressor_data_item in compressor_data_list:
        ax.plot(compressor_data_item['datetime'], compressor_data_item['temperature'],
                linewidth=1.5,
                alpha=0.85,
                color=compressor_color,
                marker=compressor_data_item['marker'],
                markersize=4,
                markevery=max(1, len(compressor_data_item['datetime']) // 200),  # 每200个点显示一个marker
                label=f"Compressor {compressor_data_item['label']}")
    
    # 创建右侧y轴用于显示功率
    ax2 = None
    if compressor_power is not None:
        ax2 = ax.twinx()  # 创建共享x轴的第二个y轴
        
        # 在右侧y轴绘制功率曲线（使用点显示）
        ax2.plot(compressor_power_datetime, compressor_power,
                linewidth=0,  # 不显示线
                alpha=0.85,
                color='#6A994E',  # 绿色表示功率
                linestyle='None',  # 不显示线
                marker='.',  # 使用点显示
                markersize=3,  # 点的大小
                label='Cooler Power')
        
        # 设置右侧y轴标签和格式
        ax2.set_ylabel('Cooler Power (W)', fontsize=13, fontweight='normal', color='#6A994E')
        ax2.tick_params(axis='y', labelcolor='#6A994E', labelsize=10)
        ax2.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
        ax2.yaxis.get_major_formatter().set_scientific(False)
        
        # 设置右侧y轴的次要刻度
        power_range = np.max(compressor_power) - np.min(compressor_power)
        if power_range > 0:
            minor_interval = power_range / 20
            ax2.yaxis.set_minor_locator(ticker.MultipleLocator(minor_interval))
    
    # 设置标签和标题（使用英文）
    ax.set_xlabel('Time', fontsize=13, fontweight='normal')
    ax.set_ylabel('Temperature (°C)', fontsize=13, fontweight='normal')
    
    # 格式化x轴日期 - 根据时间范围自动选择格式
    # 使用所有数据集的共同时间范围
    all_datetime_list = [vibration_datetime]
    for compressor_data_item in compressor_data_list:
        all_datetime_list.append(compressor_data_item['datetime'])
    all_datetime = np.concatenate(all_datetime_list)
    all_datetime = np.sort(all_datetime)
    
    if len(all_datetime) > 1:
        time_span = all_datetime[-1] - all_datetime[0]
        
        # 转换为timedelta（处理numpy datetime类型）
        if hasattr(time_span, 'days'):
            days = time_span.days
            total_hours = time_span.total_seconds() / 3600
        else:
            # 如果是numpy timedelta64
            days = time_span.astype('timedelta64[D]').astype(int)
            total_hours = time_span.astype('timedelta64[h]').astype(int)
        
        if days > 30:
            # 超过30天，显示日期和每天的6:00、12:00、18:00
            date_format = '%m-%d %H:%M'
            # 使用HourLocator指定每天的小时数：6:00, 12:00, 18:00
            locator = mdates.HourLocator(byhour=[6, 12, 18])
            minor_locator = mdates.HourLocator(interval=3)  # 每3小时一个次要刻度
        elif days > 1:
            # 1-30天，显示日期和时间，每天的6:00、12:00、18:00
            date_format = '%m-%d %H:%M'
            # 使用HourLocator指定每天的小时数：6:00, 12:00, 18:00
            locator = mdates.HourLocator(byhour=[6, 12, 18])
            minor_locator = mdates.HourLocator(interval=3)  # 每3小时一个次要刻度
        else:
            # 小于1天，显示时间，每6小时一个刻度（6:00, 12:00, 18:00, 0:00）
            date_format = '%H:%M'
            locator = mdates.HourLocator(byhour=[0, 6, 12, 18])  # 包括0:00
            minor_locator = mdates.HourLocator(interval=3)  # 每3小时一个次要刻度
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_minor_locator(minor_locator)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 设置y轴格式，确保显示真实值而不是偏移量
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
    ax.yaxis.get_major_formatter().set_scientific(False)
    
    # 添加y轴的次要刻度（基于所有温度值的范围）
    all_temp_values_list = [vibration_temp]
    for compressor_data_item in compressor_data_list:
        all_temp_values_list.append(compressor_data_item['temperature'])
    all_temp_values = np.concatenate(all_temp_values_list)
    temp_range = np.max(all_temp_values) - np.min(all_temp_values)
    if temp_range > 0:
        # 根据温度范围自动设置次要刻度间隔
        minor_interval = temp_range / 20
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(minor_interval))
    
    # 改进网格线样式
    ax.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.3, color='gray')
    ax.grid(True, which='minor', linestyle='--', linewidth=0.5, alpha=0.2, color='gray')
    
    # 设置坐标轴边框样式
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_color('gray')
    ax.spines['right'].set_color('gray')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    
    # 计算统计信息
    vibration_mean = np.mean(vibration_temp)
    vibration_std = np.std(vibration_temp)
    vibration_min = np.min(vibration_temp)
    vibration_max = np.max(vibration_temp)
    
    # 计算所有compressor温度列的统计信息
    compressor_stats_list = []
    for compressor_data_item in compressor_data_list:
        temp_arr = compressor_data_item['temperature']
        compressor_stats_list.append({
            'label': compressor_data_item['label'],
            'n': len(temp_arr),
            'min': np.min(temp_arr),
            'max': np.max(temp_arr),
            'mean': np.mean(temp_arr),
            'std': np.std(temp_arr)
        })
    
    # 添加图例（合并左侧和右侧y轴的图例）
    lines1, labels1 = ax.get_legend_handles_labels()
    if ax2 is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, 
                 loc='upper right', framealpha=0.9, edgecolor='gray',
                 frameon=True, fancybox=False, shadow=False)
    else:
        ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray',
                 frameon=True, fancybox=False, shadow=False)
    
    # 在图的角落添加统计信息框
    stats_text = (f'Vibration Sensor:\n'
                 f'  N = {len(vibration_temp)}\n'
                 f'  Min = {vibration_min:.2f} °C\n'
                 f'  Max = {vibration_max:.2f} °C\n'
                 f'  Mean = {vibration_mean:.2f} °C\n'
                 f'  Std = {vibration_std:.2f} °C\n'
                 f'\nCompressor:')
    
    for stats in compressor_stats_list:
        stats_text += (f'\n  {stats["label"]}:\n'
                      f'    N = {stats["n"]}\n'
                      f'    Mean = {stats["mean"]:.2f} °C')
    
    # 添加功率统计信息
    if compressor_power is not None:
        power_mean = np.mean(compressor_power)
        power_std = np.std(compressor_power)
        power_min = np.min(compressor_power)
        power_max = np.max(compressor_power)
        stats_text += (f'\n\nCooler Power:\n'
                      f'  N = {len(compressor_power)}\n'
                      f'  Min = {power_min:.2f} W\n'
                      f'  Max = {power_max:.2f} W\n'
                      f'  Mean = {power_mean:.2f} W\n'
                      f'  Std = {power_std:.2f} W')
    
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                     edgecolor='gray', alpha=0.8, linewidth=0.8),
            fontsize=9,
            family='monospace')
    
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'\n图片已保存至: {save_path}')
    
    # 显示图片
    if show_plot:
        plt.show()
    else:
        plt.close()


# 示例使用
if __name__ == '__main__':
    # 获取项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # current_dir = python/data/temperature/power
    parent_dir = os.path.dirname(current_dir)  # python/data/temperature
    grandparent_dir = os.path.dirname(parent_dir)  # python/data
    great_grandparent_dir = os.path.dirname(grandparent_dir)  # python
    project_root = os.path.dirname(great_grandparent_dir)  # 项目根目录
    
    # 设置数据路径
    vibration_data_dir = os.path.join(project_root, 'data', 'vibration', 'hdf5')
    compressor_file_path = os.path.join(project_root, 'data', 'compressor', 'txt', 'EC1CP5.txt')
    
    print('=' * 70)
    print('振动传感器和制冷机温度联合绘制')
    print('=' * 70)
    
    try:
        # 绘制联合温度图
        plot_united_temperature(
            vibration_data_dir=vibration_data_dir,
            compressor_file_path=compressor_file_path,
            detector_num=2,
            start_date='2025-05-28',
            end_date='2025-06-10',
            compressor_temp_column=['Compressor temp', 'Controller temp', 'Coldhead temp'],  # 可以传入单个字符串或列表
            show_plot=True
        )
        
    except Exception as e:
        print(f'错误: {e}')
        import traceback
        traceback.print_exc()
