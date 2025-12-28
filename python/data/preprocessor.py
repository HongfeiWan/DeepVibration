#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
二进制文件预处理脚本
将V1725数据采集卡的.bin文件转换为.mat格式的原始脉冲数据
"""
import os
import sys
import struct
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# 添加父目录到路径，以便导入utils模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.save import save_hdf5

# 定义文件路径和文件名（相对于项目根目录）
# 从 python/data/ 目录到项目根目录的 data/ 目录
project_root = os.path.dirname(os.path.dirname(current_dir))
read_path = os.path.join(project_root, 'data', 'bin')
amp_save_path = os.path.join(project_root, 'data', 'hdf5', 'raw_pulse', 'CH0-3')
trigger_save_path = os.path.join(project_root, 'data', 'hdf5', 'raw_pulse', 'CH5')
filename_input = '20250520_CEvNS_DZL_sm_pre10000_tri10mV_SA6us0.8x50_SA12us0.8x50_TAout10us1.2x100_TAout10us0.5x3_RT50mHz_NaISA1us1.0x20_plasticsci1-10_bkg'

RUN_Start_NUMBER = 281  # 起始运行编号
RUN_End_NUMBER = 281    # 结束运行编号

# 定义通道数和事件数
AMP_CHANNEL_LIST = [0, 1, 2, 3]     # 指定要保存的通道索引（0=CH0, 1=CH1, 2=CH2, 3=CH3）
TRIGGER_CHANNEL_LIST = [5]          # 指定要保存的随机触发通道索引（5=CH5）
EVENT_NUMBER = 10000    # 每个bin文件中的理论上事件数
MAX_WINDOWS = 30000     # 时间窗 120μs （30000个时间点 x 4ns）


def bin2rawpulse(run_filename, channel_list, event_number, save_path):
    """
    处理bin文件中对应channel_list通道的原始波形并且保存
    参数:
        run_filename: 输入文件路径
        channel_list: 要保存的通道索引列表（例如 [0, 1, 2, 3] 表示CH0-CH3）
        event_number: 预期事件数
        save_path: 保存路径
    """
    print('=' * 45)
    print(f'Opening {run_filename}')

    # 打开文件
    fid = None
    try:
        fid = open(run_filename, 'rb')
    except (IOError, OSError, FileNotFoundError) as e:
        raise OSError(f'Failed to open file: {run_filename}') from e
    
    try:
        # 读取 Run Header
        # 使用小端字节序（'<'），因为MATLAB默认使用小端
        pstt = struct.unpack('<d', fid.read(8))[0]  # Program Start Time (double, 8 bytes)
        print(f'Program Start Time: {pstt:.6f} s.')
        
        # 读取 V1725-1 Channel DAC (读取16个通道的DAC值)
        # numpy.frombuffer默认使用系统字节序，通常是小端，与MATLAB一致
        V1725_1_DAC = np.frombuffer(fid.read(16 * 4), dtype='<u4')  # 16个uint32，小端
        print('V1725-1 Channel DAC:', ' '.join(str(x) for x in V1725_1_DAC))
        
        # 读取其他 Run Header 信息
        V1725_1_twd = struct.unpack('<I', fid.read(4))[0]  # Time Window (uint32, 4 bytes)
        print(f'V1725-1 Time Window: {V1725_1_twd}')
        
        V1725_1_pretg = struct.unpack('<I', fid.read(4))[0]  # Pre Trigger (uint32, 4 bytes)
        print(f'V1725-1 Pre Trigger: {V1725_1_pretg}')
        
        V1725_1_opch = struct.unpack('<I', fid.read(4))[0]  # Opened Channel (uint32, 4 bytes)
        print(f'V1725-1 Opened Channel: {V1725_1_opch}')
        
        # 读取 Run Start Time
        rstt = struct.unpack('<d', fid.read(8))[0]  # Run Start Time (double, 8 bytes)
        print(f'Run Start Time: {rstt:.6f} s.')
        
        # 初始化事件数据结构
        idevt = np.zeros(event_number, dtype=np.float64)
        trig = np.zeros(event_number, dtype=np.float64)
        time_array = np.zeros(event_number, dtype=np.float64)
        deadtime = np.zeros(event_number, dtype=np.float64)
        
        # 预分配事件数据数组
        hit_pat_array = np.zeros(event_number, dtype=np.uint32)
        v1729_tg_rec_array = np.zeros(event_number, dtype=np.uint32)
        evt_endtime_array = np.zeros(event_number, dtype=np.uint32)
        v1725_1_tgno_array = np.zeros(event_number, dtype=np.uint32)
        v1725_1_tag_array = np.zeros(event_number, dtype=np.uint32)
        
        # 初始化数据结构 - 保存原始脉冲数据
        # 只保存指定通道的数据
        num_channels_to_save = len(channel_list)
        channel_data = np.zeros((V1725_1_twd, num_channels_to_save, event_number), dtype=np.uint16)
        time_data = np.zeros(event_number, dtype=np.float64)
        
        # 验证通道索引有效性
        max_channel_index = max(channel_list) if channel_list else -1
        if max_channel_index >= 16:
            raise ValueError(f'通道索引超出范围！最大支持通道索引为15，但指定了通道 {max_channel_index}')
        if min(channel_list) < 0:
            raise ValueError(f'通道索引不能为负数！')
        
        # 首先读取所有事件的数据
        actual_events = 0
        for j in range(event_number):
            # 检查是否到达文件末尾
            current_pos = fid.tell()
            fid.seek(0, 2)  # 移动到文件末尾
            file_size = fid.tell()
            fid.seek(current_pos)  # 恢复位置
            
            if current_pos >= file_size:
                print(f'警告：文件在事件 {j+1} 处结束，实际读取了 {actual_events} 个事件')
                break
            
            # 读取 Event Header
            hit_pat_bytes = fid.read(4)  # Hit_pat (uint32, 4 bytes)
            if len(hit_pat_bytes) < 4:
                print(f'警告：文件在事件 {j+1} 处结束，实际读取了 {actual_events} 个事件')
                break
            hit_pat_data = struct.unpack('<I', hit_pat_bytes)[0]
            hit_pat_array[j] = hit_pat_data
            
            v1729_tg_rec_bytes = fid.read(4)  # V1729_tg_rec (uint32, 4 bytes)
            if len(v1729_tg_rec_bytes) < 4:
                print(f'警告：文件在事件 {j+1} 处结束，实际读取了 {actual_events} 个事件')
                break
            v1729_tg_rec_data = struct.unpack('<I', v1729_tg_rec_bytes)[0]
            v1729_tg_rec_array[j] = v1729_tg_rec_data
            
            evt_endtime_bytes = fid.read(4)  # Evt_endtime (uint32, 4 bytes)
            if len(evt_endtime_bytes) < 4:
                print(f'警告：文件在事件 {j+1} 处结束，实际读取了 {actual_events} 个事件')
                break
            evt_endtime_data = struct.unpack('<I', evt_endtime_bytes)[0]
            evt_endtime_array[j] = evt_endtime_data
            
            v1725_1_tgno_bytes = fid.read(4)  # V1725_1_tgno (uint32, 4 bytes)
            if len(v1725_1_tgno_bytes) < 4:
                print(f'警告：文件在事件 {j+1} 处结束，实际读取了 {actual_events} 个事件')
                break
            v1725_1_tgno_data = struct.unpack('<I', v1725_1_tgno_bytes)[0]
            v1725_1_tgno_array[j] = v1725_1_tgno_data
            
            v1725_1_tag_bytes = fid.read(4)  # V1725_1_tag (uint32, 4 bytes)
            if len(v1725_1_tag_bytes) < 4:
                print(f'警告：文件在事件 {j+1} 处结束，实际读取了 {actual_events} 个事件')
                break
            v1725_1_tag_data = struct.unpack('<I', v1725_1_tag_bytes)[0]
            v1725_1_tag_array[j] = v1725_1_tag_data
            
            # 读取每个通道的数据 (只处理前4个通道，但需要跳过其他通道)
            for k in range(16):  # 仍然需要读取所有16个通道的数据
                channel_data_bytes = fid.read(V1725_1_twd * 2)  # uint16 = 2 bytes
                if len(channel_data_bytes) < V1725_1_twd * 2:
                    print(f'警告：文件在事件 {j+1} 通道 {k+1} 处结束，实际读取了 {actual_events} 个事件')
                    break
                channel_data_temp = np.frombuffer(channel_data_bytes, dtype='<u2')  # uint16，小端
                
                # 只保存指定通道的数据
                if k in channel_list:
                    # 找到k在channel_list中的位置索引
                    channel_idx = channel_list.index(k)
                    channel_data[:, channel_idx, j] = channel_data_temp
            
            # 计算时间信息（将在后续处理中加上pstt）
            time_with_pstt = evt_endtime_data / 1000.0  # 先转换为秒，后续会加上pstt
            time_data[j] = time_with_pstt
            
            actual_events += 1
        
        # 更新实际事件数并调整数组大小
        if actual_events < event_number:
            event_number = actual_events
            print(f'实际读取的事件数：{event_number}')
            
            # 调整数组大小以匹配实际事件数
            hit_pat_array = hit_pat_array[:actual_events]
            v1729_tg_rec_array = v1729_tg_rec_array[:actual_events]
            evt_endtime_array = evt_endtime_array[:actual_events]
            v1725_1_tgno_array = v1725_1_tgno_array[:actual_events]
            v1725_1_tag_array = v1725_1_tag_array[:actual_events]
            channel_data = channel_data[:, :, :actual_events]
            time_data = time_data[:actual_events]
            
            # 重新初始化数组以匹配实际事件数
            idevt = idevt[:event_number]
            trig = trig[:event_number]
            time_array = time_array[:event_number]
            deadtime = deadtime[:event_number]
        
        # 关闭文件
        fid.close()
        
        # 处理所有事件（串行处理）
        for j in range(event_number):
            # 获取预读取的事件头信息
            Hit_pat = hit_pat_array[j]
            V1729_tg_rec = v1729_tg_rec_array[j]
            Evt_endtime = evt_endtime_array[j]
            V1725_1_tgno = v1725_1_tgno_array[j]
            V1725_1_tag = v1725_1_tag_array[j]
            
            # 计算时间
            TTTV1725 = V1725_1_tag & 0x7FFFFFFF
            TimeV1725 = 10.0 * float(TTTV1725)
            
            # 存储事件信息
            idevt[j] = j + 1  # 简化ID计算
            trig[j] = V1725_1_tgno
            time_array[j] = Evt_endtime / 1000.0  # 转换为秒
            deadtime[j] = 0  # 假设 deadtime 为 0，根据实际情况修改
            
            # 处理每个通道的数据（只处理保存的通道）
            for channel_idx, original_channel in enumerate(channel_list):
                V1725_1_pulse = channel_data[:, channel_idx, j]
                # 注释掉所有计算，只保留原始波形读取和时间计算
                # 如果需要计算，可以在这里添加相应的代码
                # original_channel 是原始通道索引（0-15）
                # channel_idx 是在保存数组中的索引（0到len(channel_list)-1）
        
        # 更新时间数据（加上pstt）
        time_data = time_data + pstt
        
        # 保存文件 - 使用优化的保存方法
        filename = os.path.basename(run_filename)
        base_filename = os.path.join(save_path, f'{os.path.splitext(filename)[0]}_processed')
        
        # 确保保存目录存在
        try:
            os.makedirs(save_path, exist_ok=True)
            print(f'保存目录: {save_path}')
        except Exception as e:
            print(f'创建保存目录失败: {save_path}')
            print(f'错误信息: {e}')
            raise
        
        start_time = time.time()
        # 保存所有数据 - 使用v7.3格式，因为channel_data可能超过2GB
        mdict = {
            'channel_data': channel_data,
            'time_data': time_data
        }
        
        # 使用HDF5原生格式（.h5）- 更快，MATLAB可以直接读取
        output_file = f'{base_filename}.h5'
        save_hdf5(output_file, mdict)
        
        print(f'保存完成，耗时: {time.time() - start_time:.2f}秒')
        #print(f'输出文件路径: {os.path.abspath(output_file)}')
        #print(f'MATLAB读取方式: channel_data = h5read("{os.path.basename(output_file)}", "/channel_data");')
        #print(f'                  time_data = h5read("{os.path.basename(output_file)}", "/time_data");')
        
        # 显示文件大小信息
        if os.path.exists(output_file):
            file_size_GB = os.path.getsize(output_file) / (1024**3)
            print(f'文件大小: {file_size_GB:.2f} GB')
            print(f'文件已确认存在: {os.path.abspath(output_file)}')
        else:
            print(f'警告：文件不存在: {output_file}')
            print(f'完整路径: {os.path.abspath(output_file)}')
        print('数据保存完成！')
        
    except Exception as e:
        if fid is not None:
            fid.close()
        raise

def main():
    """主函数 - 使用多进程并行处理"""
    # 生成所有任务列表
    tasks = []
    for i in range(RUN_Start_NUMBER, RUN_End_NUMBER + 1):
        run_filename = os.path.join(read_path, f'{filename_input}FADC_RAW_Data_{i}.bin')
        # 为每个文件添加两个任务：AMP通道和TRIGGER通道
        tasks.append((run_filename, AMP_CHANNEL_LIST, EVENT_NUMBER, amp_save_path))
        tasks.append((run_filename, TRIGGER_CHANNEL_LIST, EVENT_NUMBER, trigger_save_path))
    # 获取可用CPU核心数，最多使用所有核心
    max_workers = os.cpu_count()
    print(f'使用 {max_workers} 个CPU核心进行并行处理')
    print(f'共 {len(tasks)} 个任务')

    # 使用进程池并行执行
    start_time = time.time()
    success_count = 0
    fail_count = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务，直接使用 bin2rawpulse 函数
        future_to_task = {executor.submit(bin2rawpulse, *task): task for task in tasks}
        # 收集结果
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            run_filename, channel_list, _, _ = task
            try:
                future.result()  # 如果成功，没有返回值也没有异常
                success_count += 1
                print(f'[成功 {success_count}] 成功处理: {run_filename} (通道: {channel_list})')
            except Exception as e:
                fail_count += 1
                print(f'[失败 {fail_count}] 处理文件 {run_filename} (通道: {channel_list}) 时出错: {e}')
    
    elapsed_time = time.time() - start_time
    print('=' * 60)
    print(f'所有任务处理完成！')
    print(f'总耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)')
    print(f'成功: {success_count} 个任务')
    print(f'失败: {fail_count} 个任务')

if __name__ == '__main__':
    main()

