#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据保存工具模块
提供HDF5和MATLAB v7.3格式的保存功能
"""
import numpy as np
import h5py

def save_hdf5(filename, data_dict):
    """
    使用h5py保存为HDF5原生格式（.h5）
    优势：
    1. 不需要Fortran顺序转换（保持C顺序，避免2GB+数据复制）
    2. 不需要MATLAB元数据开销
    3. MATLAB可以直接用h5read读取
    4. 性能比MAT v7.3快2-3倍
    
    MATLAB读取示例：
    channel_data = h5read('filename.h5', '/channel_data');
    time_data = h5read('filename.h5', '/time_data');
    
    参数:
        filename: 保存的文件名（建议使用.h5扩展名）
        data_dict: 要保存的数据字典
    """
    try:
        #print(f'正在保存HDF5文件: {filename}')
        with h5py.File(filename, 'w') as f:
            for key, value in data_dict.items():
                if value is not None:
                    if isinstance(value, np.ndarray):
                        # HDF5原生格式：保持C顺序即可，不需要转换！
                        # 这避免了2GB+数据的复制，大幅提升性能
                        if not (value.flags['C_CONTIGUOUS'] or value.flags['F_CONTIGUOUS']):
                            # 只有不连续时才转换
                            value = np.ascontiguousarray(value)
                        
                        # 处理bool类型
                        if value.dtype == np.bool_:
                            value = value.astype(np.uint8)
                        
                        # 直接写入，保持C顺序（最快）
                        f.create_dataset(key, data=value)
                    else:
                        # 标量值
                        if not isinstance(value, (int, float, complex)):
                            value = np.array(value)
                        f.create_dataset(key, data=value)
        #print(f'HDF5文件保存成功')
    except Exception as e:
        print(f'保存HDF5文件失败 {filename}: {e}')
        raise

def save_mat_v73(filename, data_dict):
    """
    使用h5py保存为MATLAB v7.3格式（HDF5格式）
    支持大于2GB的文件，优化版本：避免不必要的数组复制
    注意：需要Fortran顺序转换，性能较慢
    
    参数:
        filename: 保存的文件名
        data_dict: 要保存的数据字典
    """
    try:
        print(f'正在保存文件: {filename}')
        with h5py.File(filename, 'w') as f:
            # 设置MATLAB版本标识
            f.attrs['MATLAB_version'] = '7.3'
            
            # MATLAB数据类型映射（预定义以提高效率）
            dtype_map = {
                np.uint8: ('uint8', np.uint8),
                np.uint16: ('uint16', np.uint16),
                np.uint32: ('uint32', np.uint32),
                np.uint64: ('uint64', np.uint64),
                np.int8: ('int8', np.int8),
                np.int16: ('int16', np.int16),
                np.int32: ('int32', np.int32),
                np.int64: ('int64', np.int64),
                np.float32: ('single', np.float32),
                np.float64: ('double', np.float64),
                np.complex64: ('single', np.complex64),
                np.complex128: ('double', np.complex128),
                bool: ('logical', np.uint8)
            }
            
            for key, value in data_dict.items():
                if value is not None:
                    if isinstance(value, np.ndarray):
                        # 优化：避免不必要的数组复制
                        # 如果数组已经是连续的，直接使用；否则才转换
                        if value.ndim > 1:
                            # 多维数组：优先使用Fortran顺序（MATLAB列优先）
                            if value.flags['F_CONTIGUOUS']:
                                # 已经是Fortran连续，直接使用
                                data_to_save = value
                            elif value.flags['C_CONTIGUOUS']:
                                # C连续，需要转换为Fortran顺序（但这是必要的）
                                data_to_save = np.asfortranarray(value)
                            else:
                                # 不连续，先转为连续再转Fortran
                                data_to_save = np.asfortranarray(np.ascontiguousarray(value))
                        else:
                            # 一维数组：保持C连续即可（更快）
                            if value.flags['C_CONTIGUOUS'] or value.flags['F_CONTIGUOUS']:
                                data_to_save = value
                            else:
                                data_to_save = np.ascontiguousarray(value)
                        
                        # 处理bool类型（h5py不支持numpy bool_）
                        if data_to_save.dtype == np.bool_:
                            data_to_save = data_to_save.astype(np.uint8)
                        
                        # 获取MATLAB类型信息
                        dtype_type = data_to_save.dtype.type
                        matlab_info = dtype_map.get(dtype_type, ('double', np.float64))
                        matlab_class, _ = matlab_info
                        
                        # 创建数据集（不压缩时直接写入，最快）
                        # 不压缩：直接写入，最快
                        dset = f.create_dataset(key, data=data_to_save)
                        
                        # 设置MATLAB类属性
                        dset.attrs['MATLAB_class'] = matlab_class.encode('utf-8')
                    else:
                        # 标量值：转换为numpy数组
                        if not isinstance(value, (int, float, complex)):
                            value = np.array(value)
                        
                        dset = f.create_dataset(key, data=value)
                        dset.attrs['MATLAB_class'] = 'double'.encode('utf-8')
        print(f'文件保存成功: {filename}')
    except Exception as e:
        print(f'保存文件失败 {filename}: {e}')
        raise

