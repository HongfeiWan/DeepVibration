#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
压缩机数据读取和筛选脚本
从EC1CP5.txt文件中读取数据，并按日期范围筛选
"""
import os
import re
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker


def read_compressor_data(file_path: str) -> pd.DataFrame:
    """
    读取压缩机数据文件
    
    参数:
        file_path: 数据文件路径
    
    返回:
        DataFrame，包含所有列数据，以及合并的datetime列
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'文件不存在: {file_path}')
    
    # 读取文件，第一行是表头
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if len(lines) < 2:
        raise ValueError('文件数据不足，至少需要表头和数据行')
    
    # 解析表头
    header = lines[0].strip()
    # 使用正则表达式分割表头，处理多个空格
    header_parts = re.split(r'\s{2,}', header.strip())
    
    # 数据列表
    data_rows = []
    
    # 从第二行开始解析数据
    for line in lines[1:]:
        line = line.strip()
        if not line:  # 跳过空行
            continue
        
        # 使用正则表达式分割数据行，处理多个空格
        # 注意：Fault status可能包含空格，需要特殊处理
        parts = re.split(r'\s{2,}', line)
        
        if len(parts) < 8:
            continue  # 跳过格式不正确的行
        
        # 解析日期和时间
        date_str = parts[0]
        time_str = parts[1]
        
        # 将日期从 DD.MM.YYYY 格式转换为 YYYY-MM-DD
        date_parts = date_str.split('.')
        if len(date_parts) == 3:
            day, month, year = date_parts
            date_formatted = f'{year}-{month}-{day}'
        else:
            continue  # 跳过格式错误的日期
        
        # 合并日期和时间
        datetime_str = f'{date_formatted} {time_str}'
        try:
            dt = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            continue  # 跳过无法解析的日期时间
        
        # 解析数值数据
        try:
            cooler_power = float(parts[2])
            coldtip_temp = float(parts[3])
            coldtip_setp = float(parts[4])
            coldhead_temp = float(parts[5])
            compressor_temp = float(parts[6])
            controller_temp = float(parts[7])
        except (ValueError, IndexError):
            continue  # 跳过无法解析的数值
        
        # 故障状态（可能包含空格，所以取剩余部分）
        fault_status = ' '.join(parts[8:]) if len(parts) > 8 else ''
        
        data_rows.append({
            'datetime': dt,
            'Date': date_str,
            'Time': time_str,
            'Cooler power': cooler_power,
            'Coldtip temp': coldtip_temp,
            'Coldtip setp': coldtip_setp,
            'Coldhead temp': coldhead_temp,
            'Compressor temp': compressor_temp,
            'Controller temp': controller_temp,
            'Fault status': fault_status
        })
    
    # 转换为DataFrame
    df = pd.DataFrame(data_rows)
    
    # 按datetime排序
    if not df.empty:
        df = df.sort_values('datetime').reset_index(drop=True)
    
    return df

def select_by_date_range(file_path: str, 
                         start_date: str, 
                         end_date: str,
                         start_time: Optional[str] = None,
                         end_time: Optional[str] = None) -> Dict[str, np.ndarray]:
    """
    按日期范围筛选数据
    
    参数:
        file_path: 数据文件路径
        start_date: 起始日期，格式 'YYYY-MM-DD' 或 'DD.MM.YYYY'
        end_date: 终止日期，格式 'YYYY-MM-DD' 或 'DD.MM.YYYY'
        start_time: 起始时间（可选），格式 'HH:MM:SS'
        end_time: 终止时间（可选），格式 'HH:MM:SS'
    
    返回:
        字典，包含所有列的numpy数组，键名为列名
    """
    # 读取数据
    df = read_compressor_data(file_path)
    
    if df.empty:
        raise ValueError('文件中没有有效数据')
    
    # 解析起始和终止日期
    # 支持两种格式：'YYYY-MM-DD' 或 'DD.MM.YYYY'
    def parse_date(date_str: str) -> datetime:
        if '.' in date_str:
            # DD.MM.YYYY 格式
            parts = date_str.split('.')
            if len(parts) == 3:
                day, month, year = parts
                return datetime(int(year), int(month), int(day))
        else:
            # YYYY-MM-DD 格式
            return datetime.strptime(date_str, '%Y-%m-%d')
        raise ValueError(f'无法解析日期格式: {date_str}')
    
    start_dt = parse_date(start_date)
    end_dt = parse_date(end_date)
    
    # 如果提供了时间，则添加到日期中
    if start_time:
        time_parts = start_time.split(':')
        if len(time_parts) == 3:
            start_dt = start_dt.replace(hour=int(time_parts[0]), 
                                       minute=int(time_parts[1]), 
                                       second=int(time_parts[2]))
    else:
        start_dt = start_dt.replace(hour=0, minute=0, second=0)
    
    if end_time:
        time_parts = end_time.split(':')
        if len(time_parts) == 3:
            end_dt = end_dt.replace(hour=int(time_parts[0]), 
                                   minute=int(time_parts[1]), 
                                   second=int(time_parts[2]))
    else:
        end_dt = end_dt.replace(hour=23, minute=59, second=59)
    
    # 筛选数据
    mask = (df['datetime'] >= start_dt) & (df['datetime'] <= end_dt)
    filtered_df = df[mask].copy()
    
    if filtered_df.empty:
        print(f'警告：在 {start_date} 到 {end_date} 之间没有找到数据')
        return {}
    
    # 转换为字典，每个列对应一个numpy数组
    result = {}
    for col in filtered_df.columns:
        if col == 'datetime':
            result['datetime'] = filtered_df[col].values
        elif col == 'Date' or col == 'Time':
            result[col] = filtered_df[col].values
        elif col == 'Fault status':
            result[col] = filtered_df[col].values
        else:
            # 数值列转换为float数组
            result[col] = filtered_df[col].values.astype(float)
    
    print(f'筛选完成：从 {len(df)} 条记录中筛选出 {len(filtered_df)} 条记录')
    print(f'日期范围：{filtered_df["datetime"].min()} 到 {filtered_df["datetime"].max()}')
    
    return result

def plot_temp_vs_datetime(data_dict: Dict[str, np.ndarray], 
                          temp_column,
                          save_path: Optional[str] = None,
                          show_plot: bool = True,
                          figsize: Tuple[int, int] = (12, 6)) -> None:
    """
    绘制温度-时间图
    
    参数:
        data_dict: 包含数据的字典，必须包含 'datetime' 和目标温度列
        temp_column: 要绘制的温度列名（字符串）或温度列名列表，可选值：
            - 'Coldtip temp'
            - 'Coldtip setp'
            - 'Coldhead temp'
            - 'Compressor temp'
            - 'Controller temp'
            - 或上述值的列表，如 ['Compressor temp', 'Controller temp']
        save_path: 保存图片的路径，如果为None则不保存
        show_plot: 是否显示图片
        figsize: 图片大小 (宽度, 高度)
    """
    if 'datetime' not in data_dict:
        raise ValueError("数据字典中必须包含 'datetime' 列")
    
    # 将单个字符串转换为列表
    if isinstance(temp_column, str):
        temp_columns = [temp_column]
    elif isinstance(temp_column, (list, tuple)):
        temp_columns = list(temp_column)
    else:
        raise ValueError(f"temp_column 必须是字符串或列表，当前类型: {type(temp_column)}")
    
    # 检查所有列是否存在
    for col in temp_columns:
        if col not in data_dict:
            raise ValueError(f"数据字典中不包含 '{col}' 列")
    
    # 温度列名到英文标签的映射
    label_map = {
        'Coldtip temp': 'Coldtip Temperature',
        'Coldtip setp': 'Coldtip Setpoint',
        'Coldhead temp': 'Coldhead Temperature',
        'Compressor temp': 'Compressor Temperature',
        'Controller temp': 'Controller Temperature',
        'Cooler power': 'Cooler Power'
    }
    
    # 定义颜色列表（科研图表常用的颜色）
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4749']
    
    # 获取datetime数据
    datetime_arr = data_dict['datetime']
    
    # 预处理所有温度列的数据
    temp_data_list = []
    valid_datetime_list = []
    
    for col in temp_columns:
        temp_arr = data_dict[col]
        # 过滤NaN值
        valid_mask = ~np.isnan(temp_arr)
        valid_datetime = datetime_arr[valid_mask]
        valid_temp = temp_arr[valid_mask]
        
        if len(valid_datetime) == 0:
            raise ValueError(f"'{col}' 列中没有有效数据")
        
        temp_data_list.append({
            'column': col,
            'label': label_map.get(col, col),
            'datetime': valid_datetime,
            'temperature': valid_temp
        })
        valid_datetime_list.append(valid_datetime)
    
    # 找到所有数据的共同时间范围
    all_datetime = datetime_arr
    
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
    
    # 绘制所有温度数据线
    all_temp_values = []  # 用于计算y轴范围
    for i, temp_data in enumerate(temp_data_list):
        color = colors[i % len(colors)]
        ax.plot(temp_data['datetime'], temp_data['temperature'],
                linewidth=1.5,
                alpha=0.85,
                color=color,
                label=temp_data['label'])
        all_temp_values.extend(temp_data['temperature'])
    
    # 设置标签和标题（使用英文）
    ax.set_xlabel('Time', fontsize=13, fontweight='normal')
    ax.set_ylabel('Temperature (°C)', fontsize=13, fontweight='normal')
    
    # 格式化x轴日期 - 根据时间范围自动选择格式
    time_span = datetime_arr[-1] - datetime_arr[0]
    
    # 转换为timedelta（处理numpy datetime类型）
    if hasattr(time_span, 'days'):
        days = time_span.days
        total_hours = time_span.total_seconds() / 3600
    else:
        # 如果是numpy timedelta64
        days = time_span.astype('timedelta64[D]').astype(int)
        total_hours = time_span.astype('timedelta64[h]').astype(int)
    
    if days > 30:
        # 超过30天，只显示日期
        date_format = '%Y-%m-%d'
        locator = mdates.DayLocator(interval=max(1, days // 10))
        minor_locator = mdates.HourLocator(interval=6)  # 每6小时一个次要刻度
    elif days > 1:
        # 1-30天，显示日期和时间
        date_format = '%m-%d %H:%M'
        major_interval = max(1, int(total_hours / 10))
        locator = mdates.HourLocator(interval=major_interval)
        minor_locator = mdates.MinuteLocator(interval=30)  # 每30分钟一个次要刻度
    else:
        # 小于1天，只显示时间
        date_format = '%H:%M'
        major_interval = max(1, int(total_hours / 8))
        locator = mdates.HourLocator(interval=major_interval)
        minor_locator = mdates.MinuteLocator(interval=15)  # 每15分钟一个次要刻度
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_minor_locator(minor_locator)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 设置y轴格式，确保显示真实值而不是偏移量
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
    ax.yaxis.get_major_formatter().set_scientific(False)
    
    # 添加y轴的次要刻度（基于所有温度值的范围）
    if len(all_temp_values) > 0:
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
    
    # 添加图例（显示所有数据系列）
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray',
              frameon=True, fancybox=False, shadow=False)
    
    # 在图的角落添加统计信息框（显示每条线的统计信息）
    if len(temp_data_list) == 1:
        # 单条线：显示详细统计信息
        temp_data = temp_data_list[0]
        mean_val = np.mean(temp_data['temperature'])
        std_val = np.std(temp_data['temperature'])
        min_val = np.min(temp_data['temperature'])
        max_val = np.max(temp_data['temperature'])
        stats_text = (f'N = {len(temp_data["temperature"])}\n'
                     f'Min = {min_val:.2f} °C\n'
                     f'Max = {max_val:.2f} °C\n'
                     f'Mean = {mean_val:.2f} °C\n'
                     f'Std = {std_val:.2f} °C')
    else:
        # 多条线：显示汇总信息
        stats_lines = []
        for temp_data in temp_data_list:
            mean_val = np.mean(temp_data['temperature'])
            # 使用简短标签
            short_label = temp_data['label'].replace(' Temperature', '').replace('Temperature', 'Temp')
            stats_lines.append(f'{short_label}: {mean_val:.2f} °C')
        stats_text = 'Mean Values:\n' + '\n'.join(stats_lines)
    
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
        print(f'图片已保存至: {save_path}')
    
    # 显示图片
    if show_plot:
        plt.show()
    else:
        plt.close()

# 示例使用
if __name__ == '__main__':
    # 获取项目根目录
    # 从 python/data/compressor/select.py 向上3层到达项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    file_path = os.path.join(project_root, 'data', 'compressor', 'txt', 'EC1CP5.txt')
    
    print('=' * 70)
    print('压缩机数据读取和筛选示例')
    print('=' * 70)
    
    # 示例1: 读取并筛选数据
    print('\n示例1: 按日期范围筛选数据')
    print('-' * 70)
    try:
        # 筛选 2024年12月6日 到 2024年12月7日 的数据
        data = select_by_date_range(file_path, 
                                    start_date='28.05.2025',
                                    end_date='15.06.2025')
        
        if data:
            print(f'\n筛选结果包含以下列: {list(data.keys())}')
            print(f'数据点数量: {len(data["datetime"])}')
            
            # 示例2: 绘制温度图
            print('\n示例2: 绘制温度-时间图')
            print('-' * 70)
            
            # 绘制单个温度（Coldtip temp）
            # plot_temp_vs_datetime(data, 
            #                      temp_column='Coldtip temp',
            #                      show_plot=True)
            
            # 绘制多个温度（Compressor temp 和 Controller temp）
            plot_temp_vs_datetime(data, 
                                 temp_column=['Compressor temp', 'Controller temp','Coldhead temp'],
                                 show_plot=True)
            
            # 也可以绘制其他温度组合
            # plot_temp_vs_datetime(data, 
            #                      temp_column=['Coldtip temp', 'Coldhead temp'],
            #                      show_plot=True)
            
    except Exception as e:
        print(f'错误: {e}')
        import traceback
        traceback.print_exc()

