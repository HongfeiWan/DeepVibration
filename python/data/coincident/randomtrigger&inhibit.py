#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Random Trigger 和 Inhibit 信号的 Coincident 分析
分析CH0-3和CH5目录中对应文件的coincident事件
"""
import os
import sys
import h5py
import numpy as np
from typing import Optional, Dict, Tuple, List

# 添加路径以便导入 utils 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
python_dir = os.path.dirname(os.path.dirname(current_dir))
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)
from utils.visualize import get_h5_files

def find_matched_file_pairs(ch0_3_file: str = None, ch5_file: str = None) -> List[Tuple[str, str]]:
    """
    查找CH0-3和CH5目录中文件名匹配的文件对
    参数:
        ch0_3_file: CH0-3文件的路径，如果为None则查找所有匹配的文件对
        ch5_file: CH5文件的路径，如果为None则查找所有匹配的文件对
    
    返回:
        文件对列表 [(ch0_3_path, ch5_path), ...]
    """
    h5_files = get_h5_files()
    
    if ch0_3_file is not None and ch5_file is not None:
        # 直接使用指定的文件对
        return [(ch0_3_file, ch5_file)]
    
    ch0_3_files = h5_files.get('CH0-3', [])
    ch5_files = h5_files.get('CH5', [])
    
    if not ch0_3_files or not ch5_files:
        raise FileNotFoundError('CH0-3或CH5目录中未找到文件')
    
    # 匹配文件名（去除路径，只比较文件名）
    matched_pairs = []
    
    # 创建文件名到路径的映射
    ch0_3_dict = {os.path.basename(f): f for f in ch0_3_files}
    ch5_dict = {os.path.basename(f): f for f in ch5_files}
    
    # 查找匹配的文件名
    for filename in ch0_3_dict.keys():
        if filename in ch5_dict:
            matched_pairs.append((ch0_3_dict[filename], ch5_dict[filename]))
    
    if not matched_pairs:
        raise ValueError('未找到匹配的文件对')
    
    return matched_pairs

def analyze_coincident_events(ch0_3_file: str,
                              ch5_file: str,
                              rt_cut: float = 6000.0,
                              ch0_idx: int = 0,
                              ch5_idx: int = 0,
                              verbose: bool = True) -> Dict:
    """
    分析一对文件的coincident事件
    参数:
        ch0_3_file: CH0-3文件路径
        ch5_file: CH5文件路径
        rt_cut: RT信号的截断阈值（CH5最大值 > rt_cut）
        ch0_idx: CH0-3文件中的CH0通道索引（默认0）
        ch5_idx: CH5文件中的通道索引（默认0）
        verbose: 是否打印详细进度信息
    返回:
        包含统计信息的字典
    """
    if not os.path.exists(ch0_3_file):
        raise FileNotFoundError(f'文件不存在: {ch0_3_file}')
    if not os.path.exists(ch5_file):
        raise FileNotFoundError(f'文件不存在: {ch5_file}')
    
    if verbose:
        print('=' * 70)
        print(f'分析文件对:')
        print(f'  CH0-3: {os.path.basename(ch0_3_file)}')
        print(f'  CH5:   {os.path.basename(ch5_file)}')
        print('=' * 70)
    
    try:
        # 读取CH0-3文件（分析CH0的最小值）
        with h5py.File(ch0_3_file, 'r') as f_ch0:
            ch0_channel_data = f_ch0['channel_data']
            ch0_time_samples, ch0_num_channels, ch0_num_events = ch0_channel_data.shape
            
            if verbose:
                print(f'\nCH0-3文件维度: (时间点={ch0_time_samples}, 通道数={ch0_num_channels}, 事件数={ch0_num_events})')
            
            if ch0_idx < 0 or ch0_idx >= ch0_num_channels:
                raise IndexError(f'CH0通道索引 {ch0_idx} 超出范围 [0, {ch0_num_channels-1}]')
            
            # 提取所有事件的CH0最小值
            if verbose:
                print(f'\n正在计算CH0信号最小值...')
            ch0_min_values = np.zeros(ch0_num_events, dtype=np.float64)
            batch_size = 1000
            
            for i in range(0, ch0_num_events, batch_size):
                end_idx = min(i + batch_size, ch0_num_events)
                batch_data = ch0_channel_data[:, ch0_idx, i:end_idx]
                batch_min = np.min(batch_data, axis=0)
                ch0_min_values[i:end_idx] = batch_min
                
                if verbose and ((i // batch_size + 1) % 10 == 0 or end_idx == ch0_num_events):
                    print(f'  已处理 {end_idx}/{ch0_num_events} 个事件 ({end_idx/ch0_num_events*100:.1f}%)')
        
        # 读取CH5文件（分析CH5的最大值）
        with h5py.File(ch5_file, 'r') as f_ch5:
            ch5_channel_data = f_ch5['channel_data']
            ch5_time_samples, ch5_num_channels, ch5_num_events = ch5_channel_data.shape
            
            if verbose:
                print(f'\nCH5文件维度: (时间点={ch5_time_samples}, 通道数={ch5_num_channels}, 事件数={ch5_num_events})')
            
            if ch5_idx < 0 or ch5_idx >= ch5_num_channels:
                raise IndexError(f'CH5通道索引 {ch5_idx} 超出范围 [0, {ch5_num_channels-1}]')
            
            # 检查事件数是否匹配
            if ch0_num_events != ch5_num_events:
                raise ValueError(f'事件数不匹配！CH0-3: {ch0_num_events}, CH5: {ch5_num_events}')
            
            # 提取所有事件的CH5最大值
            if verbose:
                print(f'\n正在计算CH5信号最大值...')
            ch5_max_values = np.zeros(ch5_num_events, dtype=np.float64)
            
            for i in range(0, ch5_num_events, batch_size):
                end_idx = min(i + batch_size, ch5_num_events)
                batch_data = ch5_channel_data[:, ch5_idx, i:end_idx]
                batch_max = np.max(batch_data, axis=0)
                ch5_max_values[i:end_idx] = batch_max
                
                if verbose and ((i // batch_size + 1) % 10 == 0 or end_idx == ch5_num_events):
                    print(f'  已处理 {end_idx}/{ch5_num_events} 个事件 ({end_idx/ch5_num_events*100:.1f}%)')
        
        # 判断信号类型
        # RT信号：CH5最大值 > rt_cut
        rt_mask = ch5_max_values > rt_cut
        rt_count = np.sum(rt_mask)
        
        # Inhibit信号：CH0最小值 <= 0
        inhibit_mask = ch0_min_values <= 0
        inhibit_count = np.sum(inhibit_mask)
        
        # Coincident信号：同时满足RT和Inhibit条件
        coincident_mask = rt_mask & inhibit_mask
        coincident_count = np.sum(coincident_mask)
        
        total_events = ch0_num_events
        
        # 打印统计信息
        if verbose:
            print(f'\n' + '=' * 70)
            print(f'Coincident 分析结果:')
            print('=' * 70)
            print(f'总事件数: {total_events}')
            print(f'\nRandom Trigger 信号数量 (CH5 max > {rt_cut:.2f}): {rt_count}')
            print(f'Random Trigger 信号比例: {rt_count/total_events*100:.2f}%')
            print(f'\nInhibit 信号数量 (CH0 min <= 0): {inhibit_count}')
            print(f'Inhibit 信号比例: {inhibit_count/total_events*100:.2f}%')
            print(f'\nCoincident 信号数量 (同时满足RT和Inhibit): {coincident_count}')
            print(f'Coincident 信号比例: {coincident_count/total_events*100:.2f}%')
            print(f'\n信号分类统计:')
            print(f'  仅RT信号 (非Inhibit): {rt_count - coincident_count}')
            print(f'  仅Inhibit信号 (非RT): {inhibit_count - coincident_count}')
            print(f'  既非RT也非Inhibit: {total_events - rt_count - inhibit_count + coincident_count}')
            print('=' * 70)
        
        # 返回统计结果
        stats = {
            'total_events': int(total_events),
            'rt_count': int(rt_count),
            'rt_rate': float(rt_count / total_events * 100),
            'inhibit_count': int(inhibit_count),
            'inhibit_rate': float(inhibit_count / total_events * 100),
            'coincident_count': int(coincident_count),
            'coincident_rate': float(coincident_count / total_events * 100),
            'rt_only_count': int(rt_count - coincident_count),
            'inhibit_only_count': int(inhibit_count - coincident_count),
            'neither_count': int(total_events - rt_count - inhibit_count + coincident_count),
            'rt_mask': rt_mask,
            'inhibit_mask': inhibit_mask,
            'coincident_mask': coincident_mask,
            'rt_cut': float(rt_cut),
            'ch0_min_values': ch0_min_values,
            'ch5_max_values': ch5_max_values
        }
        
        return stats
    
    except Exception as e:
        print(f'分析过程中出错: {e}')
        raise


# 示例使用
if __name__ == '__main__':
    # 方法1: 分析单个文件对
    print('=' * 70)
    print('方法1: 分析单个文件对')
    print('=' * 70)
    
    try:
        # 自动查找匹配的文件对，分析第一对
        matched_pairs = find_matched_file_pairs()
        if matched_pairs:
            ch0_3_file, ch5_file = matched_pairs[0]
            stats = analyze_coincident_events(
                ch0_3_file, ch5_file,
                rt_cut=6000.0,  # RT信号截断阈值
                ch0_idx=0,      # CH0通道索引
                ch5_idx=0,      # CH5通道索引
                verbose=True
            )
            
            print(f'\n关键统计结果:')
            print(f'  Random Trigger 信号数量: {stats["rt_count"]}')
            print(f'  Inhibit 信号数量: {stats["inhibit_count"]}')
            print(f'  Coincident 信号数量: {stats["coincident_count"]}')
    except Exception as e:
        print(f'分析失败: {e}')
    

