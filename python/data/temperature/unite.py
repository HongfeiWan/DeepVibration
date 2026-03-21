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
# current_dir = python/data/temperature
parent_dir = os.path.dirname(current_dir)  # python/data
grandparent_dir = os.path.dirname(parent_dir)  # python
project_root = os.path.dirname(grandparent_dir)  # 项目根目录

# 导入振动传感器温度模块
# 路径: python/data/sensor/vibration/temperature/select.py
vibration_temp_select_path = os.path.join(grandparent_dir, 'data', 'sensor', 'vibration', 'temperature', 'select.py')
spec_vibration = importlib.util.spec_from_file_location("vibration_temp_select", vibration_temp_select_path)
vibration_temp_select = importlib.util.module_from_spec(spec_vibration)
spec_vibration.loader.exec_module(vibration_temp_select)
select_by_date_range_vibration = vibration_temp_select.select_by_date_range_vibration

# 导入制冷机模块
# 路径: python/data/compressor/select.py
compressor_select_path = os.path.join(grandparent_dir, 'data', 'compressor', 'select.py')
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
    
    # 设置matplotlib参数以获得更好的科研图表样式
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'axes.linewidth': 1.2,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.minor.width': 0.8,
        'ytick.minor.width': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': False,
        'ytick.right': False,
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'figure.dpi': 100
    })
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制振动传感器温度（蓝色）
    ax.plot(vibration_datetime, vibration_temp,
            linewidth=1.0,
            alpha=0.85,
            color='#2E86AB',
            label='Vibration Sensor Temperature')
    
    # 绘制制冷机温度（使用相同颜色，不同marker区分）
    compressor_color = '#C73E1D'  # 统一的红色
    for compressor_data_item in compressor_data_list:
        ax.plot(compressor_data_item['datetime'], compressor_data_item['temperature'],
                linewidth=1.0,
                alpha=0.85,
                color=compressor_color,
                marker=compressor_data_item['marker'],
                markersize=6,
                markevery=max(1, len(compressor_data_item['datetime']) // 200),  # 每200个点显示一个marker
                label=f"Compressor {compressor_data_item['label']}")
    
    # 设置标签和标题（使用英文）
    ax.set_xlabel('Time', fontsize=16, fontweight='normal')
    ax.set_ylabel('Temperature (°C)', fontsize=16, fontweight='normal')
    ax.tick_params(axis="both", which="major", labelsize=12)
    
    # 使用 AutoDateLocator + ConciseDateFormatter 自动生成简洁清晰的时间轴
    locator = mdates.AutoDateLocator(minticks=5, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')
    
    # 设置y轴格式，确保显示真实值而不是偏移量
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
    ax.yaxis.get_major_formatter().set_scientific(False)
    
    # 顶部和右边不显示刻度线及边框
    ax.tick_params(top=False, right=False)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    
    # 添加图例
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray',
              frameon=True, fancybox=False, shadow=False, fontsize=12)
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
    # current_dir = python/data/temperature
    parent_dir = os.path.dirname(current_dir)  # python/data
    grandparent_dir = os.path.dirname(parent_dir)  # python
    project_root = os.path.dirname(grandparent_dir)  # 项目根目录
    
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
