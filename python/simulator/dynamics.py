#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
斯特林制冷机动力学模拟
根据活塞和位移器的运动学方程进行数值求解，并生成运动学仿真动画

活塞运动学方程：
    m_d * y'' = -m_d*g*sin(θ) + k_d*(Y_sep - Y_d) + p_buff*A_x_rod + 
                p_c*(A_x_d - A_x_rod) - P_e*A_x_d

位移器运动学方程：
    m_p * y''_p = (p_buff - p_c)*(A_x_c - A_x_rod) - k_p*y_p + F_sol
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import odeint
from typing import Dict, Tuple, Optional, List, Callable
import imageio


class StirlingRefrigeratorDynamics:
    """
    斯特林制冷机动力学模拟类
    """
    
    def __init__(self, 
                 # 活塞参数
                 m_d: float = 0.5,           # 活塞质量 (kg)
                 k_d: float = 1000.0,        # 活塞弹簧常数 (N/m)
                 Y_sep: float = 0.0,         # 活塞分离位置 (m)
                 A_x_d: float = 1e-3,        # 活塞横截面积 (m²)
                 A_x_rod: float = 5e-4,      # 活塞杆横截面积 (m²)
                 
                 # 位移器参数
                 m_p: float = 0.3,           # 位移器质量 (kg)
                 k_p: float = 800.0,         # 位移器弹簧常数 (N/m)
                 A_x_c: float = 8e-4,        # 位移器横截面积 (m²)
                 
                 # 系统参数
                 P_e: float = 1e5,           # 环境压力 (Pa)
                 R: float = 287.0,           # 气体常数 (J/(kg·K))
                 gamma: float = 1.4,         # 绝热指数
                 
                 # 初始条件
                 y_d0: float = 0.0,          # 活塞初始位置 (m)
                 y_d_dot0: float = 0.0,      # 活塞初始速度 (m/s)
                 y_p0: float = 0.0,          # 位移器初始位置 (m)
                 y_p_dot0: float = 0.0,      # 位移器初始速度 (m/s)
                 
                 # 工作参数
                 V_buffer: float = 1e-3,     # 缓冲腔体积 (m³)
                 V_compression: float = 5e-4, # 压缩腔初始体积 (m³)
                 T_hot: float = 300.0,       # 热端温度 (K)
                 T_cold: float = 80.0,       # 冷端温度 (K)
                 
                 # 驱动参数
                 omega: float = 50.0,        # 驱动频率 (rad/s)
                 F_sol_amplitude: float = 10.0,  # 驱动力幅值 (N)
                 theta: float = 0.0,         # 安装角度 (rad)
                 
                 # 行程限制参数
                 y_d_max: float = 0.02,      # 活塞最大行程 (m) - 20mm
                 y_d_min: float = -0.02,     # 活塞最小行程 (m) - -20mm
                 y_p_max: float = 0.02,      # 位移器最大行程 (m) - 20mm
                 y_p_min: float = -0.02,     # 位移器最小行程 (m) - -20mm
                 
                 # 阻尼参数
                 c_d: float = 5.0,           # 活塞阻尼系数 (N·s/m)
                 c_p: float = 3.0,           # 位移器阻尼系数 (N·s/m)
                 
                 # 碰撞参数
                 e_d: float = 0.8,           # 活塞碰撞恢复系数 (0-1)
                 e_p: float = 0.8):          # 位移器碰撞恢复系数 (0-1)
        """
        初始化斯特林制冷机动力学参数
        
        参数:
            m_d: 活塞质量
            k_d: 活塞弹簧常数
            Y_sep: 活塞分离位置
            A_x_d: 活塞横截面积
            A_x_rod: 活塞杆横截面积
            m_p: 位移器质量
            k_p: 位移器弹簧常数
            A_x_c: 位移器横截面积
            P_e: 环境压力
            R: 气体常数
            gamma: 绝热指数
            y_d0: 活塞初始位置
            y_d_dot0: 活塞初始速度
            y_p0: 位移器初始位置
            y_p_dot0: 位移器初始速度
            V_buffer: 缓冲腔体积
            V_compression: 压缩腔初始体积
            T_hot: 热端温度
            T_cold: 冷端温度
            omega: 驱动频率
            F_sol_amplitude: 驱动力幅值
            theta: 安装角度
        """
        # 活塞参数
        self.m_d = m_d
        self.k_d = k_d
        self.Y_sep = Y_sep
        self.A_x_d = A_x_d
        self.A_x_rod = A_x_rod
        
        # 位移器参数
        self.m_p = m_p
        self.k_p = k_p
        self.A_x_c = A_x_c
        
        # 系统参数
        self.P_e = P_e
        self.R = R
        self.gamma = gamma
        self.g = 9.81  # 重力加速度
        
        # 初始条件
        self.y_d0 = y_d0
        self.y_d_dot0 = y_d_dot0
        self.y_p0 = y_p0
        self.y_p_dot0 = y_p_dot0
        
        # 工作参数
        self.V_buffer = V_buffer
        self.V_compression = V_compression
        self.T_hot = T_hot
        self.T_cold = T_cold
        
        # 驱动参数
        self.omega = omega
        self.F_sol_amplitude = F_sol_amplitude
        self.theta = theta
        
        # 行程限制参数
        self.y_d_max = y_d_max
        self.y_d_min = y_d_min
        self.y_p_max = y_p_max
        self.y_p_min = y_p_min
        
        # 阻尼参数
        self.c_d = c_d
        self.c_p = c_p
        
        # 碰撞参数
        self.e_d = e_d
        self.e_p = e_p
        
        # 验证初始位置在限制范围内
        if y_d0 > y_d_max or y_d0 < y_d_min:
            raise ValueError(f"Initial displacer position {y_d0} is outside limits [{y_d_min}, {y_d_max}]")
        if y_p0 > y_p_max or y_p0 < y_p_min:
            raise ValueError(f"Initial piston position {y_p0} is outside limits [{y_p_min}, {y_p_max}]")
        
        # 计算初始状态
        self._calculate_initial_pressures()
    
    def _calculate_initial_pressures(self):
        """计算初始压力"""
        # 初始体积
        V_c0 = self.V_compression - self.y_d0 * (self.A_x_d - self.A_x_rod)
        V_d0 = self.V_buffer + self.y_d0 * self.A_x_rod - self.y_p0 * (self.A_x_c - self.A_x_rod)
        
        # 假设初始时压力为环境压力
        self.p_buff0 = self.P_e
        self.p_c0 = self.P_e
    
    def calculate_pressures(self, y_d: float, y_p: float, y_d_dot: float = 0.0) -> Tuple[float, float]:
        """
        计算压缩腔和缓冲腔的压力
        
        参数:
            y_d: 活塞位置
            y_p: 位移器位置
            y_d_dot: 活塞速度（用于考虑压缩过程）
        
        返回:
            (p_buff, p_c): 缓冲腔压力和压缩腔压力
        """
        # 计算体积变化
        V_c = self.V_compression - y_d * (self.A_x_d - self.A_x_rod)
        V_d = self.V_buffer + y_d * self.A_x_rod - y_p * (self.A_x_c - self.A_x_rod)
        
        # 确保体积为正
        V_c = max(V_c, 1e-6)
        V_d = max(V_d, 1e-6)
        
        # 简化的压力计算（理想气体绝热过程）
        # 如果压缩（体积减小），压力增加；如果膨胀（体积增大），压力减小
        if y_d_dot < 0:  # 压缩
            # 绝热压缩
            p_c = self.P_e * ((self.V_compression / V_c) ** self.gamma)
        else:  # 膨胀
            # 等温膨胀（简化）
            p_c = self.P_e * (self.V_compression / V_c)
        
        # 缓冲腔压力（假设是等温过程）
        p_buff = self.P_e * (self.V_buffer / V_d)
        
        return p_buff, p_c
    
    def F_sol(self, t: float) -> float:
        """
        计算驱动力（正弦驱动）
        
        参数:
            t: 时间
        
        返回:
            驱动力
        """
        return self.F_sol_amplitude * np.sin(self.omega * t)
    
    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        """
        计算系统状态变量的导数（用于odeint求解）
        
        状态变量: [y_d, y_d_dot, y_p, y_p_dot]
        
        参数:
            state: 状态向量 [y_d, y_d_dot, y_p, y_p_dot]
            t: 时间
        
        返回:
            导数向量 [y_d_dot, y_d_ddot, y_p_dot, y_p_ddot]
        """
        y_d, y_d_dot, y_p, y_p_dot = state
        
        # 应用行程限制：如果超出限制，限制位置并应用碰撞逻辑
        # 活塞（位移器）
        if y_d > self.y_d_max:
            y_d = self.y_d_max
            if y_d_dot > 0:  # 向上运动时碰撞
                y_d_dot = -self.e_d * y_d_dot  # 反弹，考虑恢复系数
        elif y_d < self.y_d_min:
            y_d = self.y_d_min
            if y_d_dot < 0:  # 向下运动时碰撞
                y_d_dot = -self.e_d * y_d_dot  # 反弹，考虑恢复系数
        
        # 位移器（活塞）
        if y_p > self.y_p_max:
            y_p = self.y_p_max
            if y_p_dot > 0:  # 向上运动时碰撞
                y_p_dot = -self.e_p * y_p_dot  # 反弹，考虑恢复系数
        elif y_p < self.y_p_min:
            y_p = self.y_p_min
            if y_p_dot < 0:  # 向下运动时碰撞
                y_p_dot = -self.e_p * y_p_dot  # 反弹，考虑恢复系数
        
        # 计算压力
        p_buff, p_c = self.calculate_pressures(y_d, y_p, y_d_dot)
        
        # 活塞加速度（从运动学方程求解，添加阻尼项）
        # m_d * y'' = -m_d*g*sin(θ) + k_d*(Y_sep - y_d) + p_buff*A_x_rod + 
        #             p_c*(A_x_d - A_x_rod) - P_e*A_x_d - c_d*y_d_dot
        y_d_ddot = ((-self.m_d * self.g * np.sin(self.theta) +
                     self.k_d * (self.Y_sep - y_d) +
                     p_buff * self.A_x_rod +
                     p_c * (self.A_x_d - self.A_x_rod) -
                     self.P_e * self.A_x_d -
                     self.c_d * y_d_dot) / self.m_d)
        
        # 行程限制力：当接近或超过限制时，施加额外的恢复力
        # 使用软限制（类似弹簧力），当接近限制时逐渐增强
        if y_d >= self.y_d_max * 0.9:  # 接近上限
            limit_penalty = -10.0 * self.k_d * (y_d - self.y_d_max * 0.9)
            y_d_ddot += limit_penalty / self.m_d
        elif y_d <= self.y_d_min * 0.9:  # 接近下限
            limit_penalty = -10.0 * self.k_d * (y_d - self.y_d_min * 0.9)
            y_d_ddot += limit_penalty / self.m_d
        
        # 位移器加速度（从运动学方程求解，添加阻尼项）
        # m_p * y''_p = (p_buff - p_c)*(A_x_c - A_x_rod) - k_p*y_p + F_sol - c_p*y_p_dot
        y_p_ddot = (((p_buff - p_c) * (self.A_x_c - self.A_x_rod) -
                     self.k_p * y_p +
                     self.F_sol(t) -
                     self.c_p * y_p_dot) / self.m_p)
        
        # 行程限制力：当接近或超过限制时，施加额外的恢复力
        if y_p >= self.y_p_max * 0.9:  # 接近上限
            limit_penalty = -10.0 * self.k_p * (y_p - self.y_p_max * 0.9)
            y_p_ddot += limit_penalty / self.m_p
        elif y_p <= self.y_p_min * 0.9:  # 接近下限
            limit_penalty = -10.0 * self.k_p * (y_p - self.y_p_min * 0.9)
            y_p_ddot += limit_penalty / self.m_p
        
        return np.array([y_d_dot, y_d_ddot, y_p_dot, y_p_ddot])
    
    def solve(self, t_span: Tuple[float, float], dt: float = 0.001) -> Dict[str, np.ndarray]:
        """
        求解动力学方程
        
        参数:
            t_span: 时间范围 (t_start, t_end)
            dt: 时间步长
        
        返回:
            包含时间、位置、速度、加速度、压力的字典
        """
        t_start, t_end = t_span
        t = np.arange(t_start, t_end, dt)
        
        # 初始状态
        state0 = np.array([self.y_d0, self.y_d_dot0, self.y_p0, self.y_p_dot0])
        
        # 使用odeint求解，但需要手动应用行程限制
        # 因为odeint可能在步长内超出限制
        solution = []
        state = state0.copy()
        
        for i, t_i in enumerate(t):
            # 计算导数
            dstate = self.derivatives(state, t_i)
            
            # 使用欧拉法手动积分（更精确地应用限制）
            dt_step = dt
            state_new = state + dstate * dt_step
            
            # 应用硬限制
            state_new[0] = np.clip(state_new[0], self.y_d_min, self.y_d_max)
            state_new[2] = np.clip(state_new[2], self.y_p_min, self.y_p_max)
            
            # 如果在限制处，检查速度方向
            if state_new[0] >= self.y_d_max and state_new[1] > 0:
                state_new[1] = -self.e_d * state_new[1]
            elif state_new[0] <= self.y_d_min and state_new[1] < 0:
                state_new[1] = -self.e_d * state_new[1]
            
            if state_new[2] >= self.y_p_max and state_new[3] > 0:
                state_new[3] = -self.e_p * state_new[3]
            elif state_new[2] <= self.y_p_min and state_new[3] < 0:
                state_new[3] = -self.e_p * state_new[3]
            
            solution.append(state_new.copy())
            state = state_new
        
        solution = np.array(solution)
        
        # 提取结果
        y_d = solution[:, 0]
        y_d_dot = solution[:, 1]
        y_p = solution[:, 2]
        y_p_dot = solution[:, 3]
        
        # 计算加速度和压力
        y_d_ddot = np.zeros_like(t)
        y_p_ddot = np.zeros_like(t)
        p_buff = np.zeros_like(t)
        p_c = np.zeros_like(t)
        
        for i, state in enumerate(solution):
            y_d_ddot[i] = self.derivatives(state, t[i])[1]
            y_p_ddot[i] = self.derivatives(state, t[i])[3]
            p_buff[i], p_c[i] = self.calculate_pressures(state[0], state[2], state[1])
        
        return {
            't': t,
            'y_d': y_d,
            'y_d_dot': y_d_dot,
            'y_d_ddot': y_d_ddot,
            'y_p': y_p,
            'y_p_dot': y_p_dot,
            'y_p_ddot': y_p_ddot,
            'p_buff': p_buff,
            'p_c': p_c
        }
    
    def plot_trajectories(self, results: Dict[str, np.ndarray], 
                         save_path: Optional[str] = None,
                         show_plot: bool = True) -> None:
        """
        绘制运动轨迹和参数曲线
        
        参数:
            results: solve()方法返回的结果字典
            save_path: 保存路径（可选）
            show_plot: 是否显示图形
        """
        t = results['t']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Stirling Refrigerator Dynamics Simulation Results', fontsize=16, fontweight='bold')
        
        # Position
        axes[0, 0].plot(t, results['y_d'] * 1000, label='Displacer Position', linewidth=2)
        axes[0, 0].plot(t, results['y_p'] * 1000, label='Piston Position', linewidth=2)
        # 添加行程限制线
        axes[0, 0].axhline(y=self.y_d_max * 1000, color='r', linestyle='--', 
                          alpha=0.5, label=f'Displacer Max ({self.y_d_max*1000:.1f} mm)')
        axes[0, 0].axhline(y=self.y_d_min * 1000, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].axhline(y=self.y_p_max * 1000, color='orange', linestyle='--', 
                          alpha=0.5, label=f'Piston Max ({self.y_p_max*1000:.1f} mm)')
        axes[0, 0].axhline(y=self.y_p_min * 1000, color='orange', linestyle='--', alpha=0.5)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Position (mm)')
        axes[0, 0].set_title('Position vs Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Velocity
        axes[0, 1].plot(t, results['y_d_dot'] * 1000, label='Displacer Velocity', linewidth=2)
        axes[0, 1].plot(t, results['y_p_dot'] * 1000, label='Piston Velocity', linewidth=2)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Velocity (mm/s)')
        axes[0, 1].set_title('Velocity vs Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Acceleration
        axes[0, 2].plot(t, results['y_d_ddot'] * 1000, label='Displacer Acceleration', linewidth=2)
        axes[0, 2].plot(t, results['y_p_ddot'] * 1000, label='Piston Acceleration', linewidth=2)
        axes[0, 2].set_xlabel('Time (s)')
        axes[0, 2].set_ylabel('Acceleration (mm/s²)')
        axes[0, 2].set_title('Acceleration vs Time')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Pressure
        axes[1, 0].plot(t, results['p_buff'] / 1e5, label='Buffer Pressure', linewidth=2)
        axes[1, 0].plot(t, results['p_c'] / 1e5, label='Compression Pressure', linewidth=2)
        axes[1, 0].axhline(y=self.P_e / 1e5, color='r', linestyle='--', label='Ambient Pressure')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Pressure (bar)')
        axes[1, 0].set_title('Pressure vs Time')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Phase diagram (position-velocity)
        axes[1, 1].plot(results['y_d'] * 1000, results['y_d_dot'] * 1000, 
                       label='Displacer Phase', linewidth=2, alpha=0.7)
        axes[1, 1].set_xlabel('Position (mm)')
        axes[1, 1].set_ylabel('Velocity (mm/s)')
        axes[1, 1].set_title('Displacer Phase Diagram')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Piston phase diagram
        axes[1, 2].plot(results['y_p'] * 1000, results['y_p_dot'] * 1000, 
                       label='Piston Phase', linewidth=2, alpha=0.7, color='orange')
        axes[1, 2].set_xlabel('Position (mm)')
        axes[1, 2].set_ylabel('Velocity (mm/s)')
        axes[1, 2].set_title('Piston Phase Diagram')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'Figure saved to: {save_path}')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def create_animation(self, results: Dict[str, np.ndarray],
                        save_path: Optional[str] = None,
                        fps: int = 30,
                        show_animation: bool = False,
                        skip_frames: int = 1) -> Optional[animation.FuncAnimation]:
        """
        创建运动学仿真动画
        
        参数:
            results: solve()方法返回的结果字典
            save_path: GIF保存路径（可选）
            fps: 动画帧率
            show_animation: 是否显示动画窗口
            skip_frames: 跳帧数（用于减少计算量）
        
        返回:
            动画对象（如果save_path为None且show_animation为True）
        """
        t = results['t'][::skip_frames]
        y_d = results['y_d'][::skip_frames] * 1000  # 转换为mm
        y_p = results['y_p'][::skip_frames] * 1000
        p_buff = results['p_buff'][::skip_frames] / 1e5  # 转换为bar
        p_c = results['p_c'][::skip_frames] / 1e5
        
        # 确定坐标范围
        y_min = min(np.min(y_d), np.min(y_p)) - 5
        y_max = max(np.max(y_d), np.max(y_p)) + 5
        
        # Calculate overall vibration amplitude (system center position)
        # System center position represents the overall vibration of the system
        y_center = (y_d + y_p) / 2  # Center position
        y_amplitude = np.abs(y_center - np.mean(y_center))  # Amplitude from center
        # Alternative: relative displacement between piston and displacer
        y_relative = y_d - y_p  # Relative displacement
        
        # Create figure with 3 subplots
        fig = plt.figure(figsize=(20, 8))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1], hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])
        fig.suptitle('Stirling Refrigerator Kinematics Simulation Animation', fontsize=16, fontweight='bold')
        
        # Left plot: Mechanical structure diagram
        ax1.set_xlim(-50, 50)
        ax1.set_ylim(y_min - 20, y_max + 20)
        ax1.set_aspect('equal')
        ax1.set_xlabel('Horizontal Position (mm)', fontsize=12)
        ax1.set_ylabel('Vertical Position (mm)', fontsize=12)
        ax1.set_title('Mechanical Structure Motion', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Draw fixed structure
        # Cylinder walls
        cylinder_width = 30
        y_wall_range = np.array([y_min - 20, y_max + 20])
        ax1.plot([-cylinder_width/2, -cylinder_width/2], y_wall_range, 
                color='gray', linewidth=3, label='Cylinder Wall')
        ax1.plot([cylinder_width/2, cylinder_width/2], y_wall_range, 
                color='gray', linewidth=3)
        
        # Piston and displacer graphical elements
        piston_rect = plt.Rectangle((-10, y_min), 20, 10, fc='blue', ec='black', linewidth=2)
        displacer_rect = plt.Rectangle((-8, y_min), 16, 8, fc='orange', ec='black', linewidth=2)
        piston_text = ax1.text(0, y_min + 5, 'Piston', ha='center', va='center', 
                              fontsize=10, fontweight='bold', color='white')
        displacer_text = ax1.text(0, y_min + 4, 'Displacer', ha='center', va='center',
                                 fontsize=9, fontweight='bold', color='white')
        time_text = ax1.text(0, y_max + 15, '', ha='center', va='center',
                            fontsize=12, fontweight='bold',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax1.add_patch(piston_rect)
        ax1.add_patch(displacer_rect)
        ax1.legend(loc='upper right')
        
        # Right plot: Parameters vs time
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Position (mm)', fontsize=12)
        ax2.set_title('Parameter Variation', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        line_d, = ax2.plot([], [], 'b-', linewidth=2, label='Displacer Position')
        line_p, = ax2.plot([], [], 'orange', linewidth=2, label='Piston Position')
        
        ax2_twin = ax2.twinx()
        ax2_twin.set_ylabel('Pressure (bar)', fontsize=12)
        line_pb, = ax2_twin.plot([], [], 'g--', linewidth=1.5, label='Buffer Pressure', alpha=0.7)
        line_pc, = ax2_twin.plot([], [], 'r--', linewidth=1.5, label='Compression Pressure', alpha=0.7)
        
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.set_xlim(t[0], t[-1])
        ax2.set_ylim(y_min, y_max)
        ax2_twin.set_ylim(0, max(np.max(p_buff), np.max(p_c)) * 1.1)
        
        # Third plot: Overall vibration amplitude
        ax3.set_xlabel('Time (s)', fontsize=12)
        ax3.set_ylabel('Amplitude (mm)', fontsize=12)
        ax3.set_title('Overall System Vibration Amplitude', fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        line_center, = ax3.plot([], [], 'purple', linewidth=2, label='Center Position', alpha=0.7)
        line_amplitude, = ax3.plot([], [], 'r-', linewidth=2, label='Vibration Amplitude', alpha=0.8)
        line_relative, = ax3.plot([], [], 'g--', linewidth=2, label='Relative Displacement', alpha=0.7)
        
        ax3.legend(loc='upper right')
        ax3.set_xlim(t[0], t[-1])
        # Set appropriate y limits for amplitude
        amp_min = min(np.min(y_center), np.min(y_relative), 0) - 1
        amp_max = max(np.max(y_center), np.max(y_relative)) + 1
        ax3.set_ylim(amp_min, amp_max)
        ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # 用于存储历史轨迹
        t_history = []
        y_d_history = []
        y_p_history = []
        p_buff_history = []
        p_c_history = []
        y_center_history = []
        y_amplitude_history = []
        y_relative_history = []
        
        def animate(frame):
            idx = frame
            if idx >= len(t):
                return piston_rect, displacer_rect, piston_text, displacer_text, time_text, \
                       line_d, line_p, line_pb, line_pc, line_center, line_amplitude, line_relative
            
            current_t = t[idx]
            current_y_d = y_d[idx]
            current_y_p = y_p[idx]
            current_p_buff = p_buff[idx]
            current_p_c = p_c[idx]
            current_y_center = y_center[idx]
            current_y_amplitude = y_amplitude[idx]
            current_y_relative = y_relative[idx]
            
            # 更新位置
            piston_rect.set_y(current_y_d)
            displacer_rect.set_y(current_y_p)
            piston_text.set_position((0, current_y_d + 5))
            displacer_text.set_position((0, current_y_p + 4))
            time_text.set_text(f'Time: {current_t:.3f} s\n'
                             f'Piston: {current_y_d:.2f} mm\n'
                             f'Displacer: {current_y_p:.2f} mm\n'
                             f'Buffer: {current_p_buff:.2f} bar\n'
                             f'Compression: {current_p_c:.2f} bar')
            
            # 更新历史轨迹
            t_history.append(current_t)
            y_d_history.append(current_y_d)
            y_p_history.append(current_y_p)
            p_buff_history.append(current_p_buff)
            p_c_history.append(current_p_c)
            y_center_history.append(current_y_center)
            y_amplitude_history.append(current_y_amplitude)
            y_relative_history.append(current_y_relative)
            
            # 只保留最近500个点
            if len(t_history) > 500:
                t_history.pop(0)
                y_d_history.pop(0)
                y_p_history.pop(0)
                p_buff_history.pop(0)
                p_c_history.pop(0)
                y_center_history.pop(0)
                y_amplitude_history.pop(0)
                y_relative_history.pop(0)
            
            # 更新曲线
            line_d.set_data(t_history, y_d_history)
            line_p.set_data(t_history, y_p_history)
            line_pb.set_data(t_history, p_buff_history)
            line_pc.set_data(t_history, p_c_history)
            
            # 更新整体振动幅度曲线
            line_center.set_data(t_history, y_center_history)
            line_amplitude.set_data(t_history, y_amplitude_history)
            line_relative.set_data(t_history, y_relative_history)
            
            return piston_rect, displacer_rect, piston_text, displacer_text, time_text, \
                   line_d, line_p, line_pb, line_pc, line_center, line_amplitude, line_relative
        
        # 创建动画
        anim = animation.FuncAnimation(fig, animate, frames=len(t),
                                     interval=1000/fps, blit=False, repeat=True)
        
        if save_path:
            print(f'Saving animation to {save_path}...')
            try:
                # Save as GIF using imageio
                frames = []
                writer = imageio.get_writer(save_path, fps=fps, loop=0)
                for i in range(0, len(t)):
                    animate(i)
                    fig.canvas.draw()
                    # Convert to numpy array
                    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    writer.append_data(buf)
                writer.close()
                print(f'Animation saved to: {save_path}')
            except Exception as e:
                print(f'Error saving animation: {e}')
                print('Trying to save using PIL...')
                try:
                    # Alternative: Save using PIL
                    from PIL import Image
                    frames = []
                    for i in range(0, len(t), max(1, len(t) // 300)):  # Limit to 300 frames max
                        animate(i)
                        fig.canvas.draw()
                        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                        frames.append(Image.fromarray(buf))
                    frames[0].save(save_path, save_all=True, append_images=frames[1:],
                                 duration=1000/fps, loop=0)
                    print(f'Animation saved to: {save_path}')
                except Exception as e2:
                    print(f'PIL save also failed: {e2}')
        
        if show_animation:
            plt.show()
            return anim
        else:
            if save_path:
                plt.close(fig)
            return None


def main():
    """主函数：演示如何使用斯特林制冷机动力学模拟"""
    # 创建动力学模型实例
    model = StirlingRefrigeratorDynamics(
        # 活塞参数
        m_d=0.5,
        k_d=1000.0,
        Y_sep=0.0,
        A_x_d=1e-3,
        A_x_rod=5e-4,
        
        # 位移器参数
        m_p=0.3,
        k_p=800.0,
        A_x_c=8e-4,
        
        # 初始条件
        y_d0=0.0,
        y_d_dot0=0.0,
        y_p0=0.0,
        y_p_dot0=0.0,
        
        # 驱动参数
        omega=50.0,
        F_sol_amplitude=10.0,
        theta=0.0,
        
        # 行程限制（20mm行程）
        y_d_max=0.02,
        y_d_min=-0.02,
        y_p_max=0.02,
        y_p_min=-0.02,
        
        # 阻尼系数
        c_d=5.0,
        c_p=3.0,
        
        # 碰撞恢复系数
        e_d=0.8,
        e_p=0.8
    )
    
    # Solve dynamics equations
    print('Solving dynamics equations...')
    results = model.solve(t_span=(0.0, 1.0), dt=0.001)
    
    # Plot trajectories
    print('Plotting trajectories...')
    model.plot_trajectories(results, save_path=None, show_plot=True)
    
    # Create animation
    print('Creating animation...')
    model.create_animation(results, 
                          save_path='stirling_animation.gif',
                          fps=30,
                          show_animation=False,
                          skip_frames=5)
    
    print('Simulation completed!')


if __name__ == '__main__':
    main()
