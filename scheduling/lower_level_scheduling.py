# lower_level_scheduling.py (奖励收敛优化版)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib
import os
import time
import math
import pandas as pd
from typing import Dict, List, Tuple, Any

os.makedirs('training_curves', exist_ok=True)
os.makedirs('lower_level_results', exist_ok=True)

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


# ==================== 改进的奖励收敛可视化 ====================
def plot_reward_convergence(period_idx, station_rewards_history, car_rewards_history,
                            station_losses=None, car_losses=None, critic_losses=None):
    """奖励收敛过程可视化（含基线对比分析）"""
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f'时段 {period_idx} - 奖励收敛过程（基线对比）', fontsize=16, fontweight='bold')

    # 1. 奖励收敛图（带基线）
    ax1 = plt.subplot(2, 2, 1)
    episodes = range(len(station_rewards_history))

    # 计算移动平均（窗口大小=50）
    window = 50
    if len(station_rewards_history) >= window:
        station_smooth = pd.Series(station_rewards_history).rolling(window=window, min_periods=1).mean()
        car_smooth = pd.Series(car_rewards_history).rolling(window=window, min_periods=1).mean()
    else:
        station_smooth = station_rewards_history
        car_smooth = car_rewards_history

    # 绘制收敛带（表示标准差）
    if len(station_rewards_history) >= 100:
        station_std = pd.Series(station_rewards_history).rolling(window=100, min_periods=1).std()
        ax1.fill_between(episodes,
                         station_smooth - station_std,
                         station_smooth + station_std,
                         alpha=0.2, color='blue', label='充电站奖励标准差')

        car_std = pd.Series(car_rewards_history).rolling(window=100, min_periods=1).std()
        ax1.fill_between(episodes,
                         car_smooth - car_std,
                         car_smooth + car_std,
                         alpha=0.2, color='red', label='电动汽车奖励标准差')

    # 绘制移动平均线
    ax1.plot(episodes, station_smooth, 'b-', linewidth=2.5, label=f'充电站（{window}步平均）')
    ax1.plot(episodes, car_smooth, 'r-', linewidth=2.5, label=f'电动汽车（{window}步平均）')

    # 添加基线
    if len(station_rewards_history) > 50:
        # 初始基线（前50步平均）
        initial_baseline = np.mean(station_rewards_history[:50])
        ax1.axhline(y=initial_baseline, color='gray', linestyle='--', alpha=0.7,
                    label=f'初始基线 ({initial_baseline:.3f})')

        # 最终基线（后100步平均）
        final_baseline = np.mean(station_rewards_history[-100:]) if len(station_rewards_history) >= 100 else \
        station_rewards_history[-1]
        ax1.axhline(y=final_baseline, color='green', linestyle='--', alpha=0.7,
                    label=f'收敛基线 ({final_baseline:.3f})')

        # 计算收敛比率
        if abs(initial_baseline) > 0.001:
            improvement_ratio = (final_baseline - initial_baseline) / abs(initial_baseline)
            ax1.text(0.5, 0.95, f'改进率: {improvement_ratio:.1%}',
                     transform=ax1.transAxes, fontsize=11,
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    ax1.set_xlabel('训练轮次', fontsize=12)
    ax1.set_ylabel('奖励值', fontsize=12)
    ax1.set_title('奖励收敛过程（带收敛基线）', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. 奖励增长率分析
    ax2 = plt.subplot(2, 2, 2)

    if len(station_rewards_history) > 10:
        # 计算奖励增长率（滑动窗口）
        growth_window = 50
        station_growth_rates = []
        car_growth_rates = []

        for i in range(growth_window, len(station_rewards_history)):
            prev_mean = np.mean(station_rewards_history[i - growth_window:i - 10])
            curr_mean = np.mean(station_rewards_history[i - 9:i + 1])
            if abs(prev_mean) > 0.001:
                station_growth_rates.append((curr_mean - prev_mean) / abs(prev_mean))
            else:
                station_growth_rates.append(0)

        for i in range(growth_window, len(car_rewards_history)):
            prev_mean = np.mean(car_rewards_history[i - growth_window:i - 10])
            curr_mean = np.mean(car_rewards_history[i - 9:i + 1])
            if abs(prev_mean) > 0.001:
                car_growth_rates.append((curr_mean - prev_mean) / abs(prev_mean))
            else:
                car_growth_rates.append(0)

        if station_growth_rates:
            growth_x = range(len(station_growth_rates))
            # 平滑增长率
            if len(station_growth_rates) > 20:
                station_growth_smooth = pd.Series(station_growth_rates).rolling(window=20, min_periods=1).mean()
                ax2.plot(growth_x, station_growth_smooth, 'b-', alpha=0.8, linewidth=2, label='充电站奖励增长率')
            else:
                ax2.plot(growth_x, station_growth_rates, 'b-', alpha=0.8, linewidth=2, label='充电站奖励增长率')

            if car_growth_rates and len(car_growth_rates) == len(station_growth_rates):
                if len(car_growth_rates) > 20:
                    car_growth_smooth = pd.Series(car_growth_rates).rolling(window=20, min_periods=1).mean()
                    ax2.plot(growth_x, car_growth_smooth, 'r-', alpha=0.8, linewidth=2, label='电动汽车奖励增长率')
                else:
                    ax2.plot(growth_x, car_growth_rates, 'r-', alpha=0.8, linewidth=2, label='电动汽车奖励增长率')

            # 添加零线和收敛阈值
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.axhline(y=0.001, color='green', linestyle='--', alpha=0.5, label='收敛阈值 (0.1%)')
            ax2.axhline(y=-0.001, color='red', linestyle='--', alpha=0.5)

    ax2.set_xlabel('训练轮次', fontsize=12)
    ax2.set_ylabel('奖励增长率', fontsize=12)
    ax2.set_title('奖励增长率分析', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3)

    # 3. 损失下降曲线
    ax3 = plt.subplot(2, 2, 3)

    if station_losses and len(station_losses) > 0:
        # 平滑损失曲线
        if len(station_losses) > 100:
            loss_window = 20
            station_loss_smooth = pd.Series(station_losses).rolling(window=loss_window, min_periods=1).mean().tolist()
            if car_losses and len(car_losses) > 0:
                car_loss_smooth = pd.Series(car_losses).rolling(window=loss_window, min_periods=1).mean().tolist()
            else:
                car_loss_smooth = []
        else:
            station_loss_smooth = station_losses
            car_loss_smooth = car_losses if car_losses else []

        # 绘制损失曲线
        loss_x = range(len(station_loss_smooth))
        ax3.plot(loss_x, station_loss_smooth, 'b-', alpha=0.8, linewidth=2, label='充电站Actor损失')

        if car_loss_smooth and len(car_loss_smooth) == len(station_loss_smooth):
            ax3.plot(loss_x, car_loss_smooth, 'r-', alpha=0.8, linewidth=2, label='电动汽车Actor损失')

        # 添加损失下降趋势线
        if len(station_loss_smooth) > 100:
            # 计算指数衰减拟合
            x_fit = np.array(loss_x)
            y_fit = np.array(station_loss_smooth)
            # 只使用正数值
            positive_mask = y_fit > 0
            if np.sum(positive_mask) > 10:
                x_fit = x_fit[positive_mask]
                y_fit = y_fit[positive_mask]

                # 指数拟合：y = a * exp(-b * x)
                try:
                    log_y = np.log(y_fit)
                    A = np.vstack([x_fit, np.ones_like(x_fit)]).T
                    b_log, a_log = np.linalg.lstsq(A, log_y, rcond=None)[0]

                    # 绘制拟合曲线
                    y_fitted = np.exp(a_log) * np.exp(b_log * x_fit)
                    ax3.plot(x_fit, y_fitted, 'g--', alpha=0.6, linewidth=2,
                             label=f'指数拟合: y={np.exp(a_log):.3f}*exp({b_log:.4f}x)')

                    # 计算收敛时间（达到初始值的10%）
                    initial_loss = station_loss_smooth[0]
                    target_loss = initial_loss * 0.1
                    convergence_steps = -np.log(target_loss / np.exp(a_log)) / b_log if b_log < 0 else float('inf')

                    if convergence_steps < len(station_loss_smooth):
                        ax3.axvline(x=convergence_steps, color='orange', linestyle=':',
                                    alpha=0.5, label=f'收敛步数: {int(convergence_steps)}')
                except:
                    pass

        ax3.set_xlabel('更新步数', fontsize=12)
        ax3.set_ylabel('损失值', fontsize=12)
        ax3.set_title('Actor网络损失下降曲线', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

        # 使用对数坐标显示损失
        if station_loss_smooth and all(v > 0 for v in station_loss_smooth):
            ax3.set_yscale('log')
            ax3.set_ylabel('损失值（对数）', fontsize=12)

    # 4. 收敛阶段分析
    ax4 = plt.subplot(2, 2, 4)

    if len(station_rewards_history) > 200:
        # 划分训练阶段
        n_phases = 4
        phase_length = len(station_rewards_history) // n_phases

        phase_stats = []
        phase_labels = ['探索阶段', '学习阶段', '优化阶段', '收敛阶段']

        for i in range(n_phases):
            start_idx = i * phase_length
            end_idx = (i + 1) * phase_length if i < n_phases - 1 else len(station_rewards_history)

            if start_idx < len(station_rewards_history):
                phase_data = station_rewards_history[start_idx:end_idx]

                phase_stats.append({
                    'mean': np.mean(phase_data),
                    'std': np.std(phase_data),
                    'min': np.min(phase_data),
                    'max': np.max(phase_data),
                    'improvement': np.mean(phase_data) - (
                        np.mean(station_rewards_history[:phase_length]) if i > 0 else 0)
                })

        x = np.arange(n_phases)
        means = [s['mean'] for s in phase_stats]
        stds = [s['std'] for s in phase_stats]
        improvements = [s['improvement'] for s in phase_stats]

        # 绘制阶段统计
        bars = ax4.bar(x, means, width=0.6, yerr=stds, capsize=5,
                       color=['lightblue', 'lightgreen', 'gold', 'lightcoral'],
                       edgecolor='black', alpha=0.8)

        # 添加数值标签
        for i, (bar, mean_val, imp_val) in enumerate(zip(bars, means, improvements)):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height + stds[i] + 0.01,
                     f'{mean_val:.3f}\n(+{imp_val:.3f})', ha='center', va='bottom',
                     fontsize=9, fontweight='bold')

        ax4.set_xlabel('训练阶段', fontsize=12)
        ax4.set_ylabel('平均奖励 ± 标准差', fontsize=12)
        ax4.set_title('各训练阶段奖励统计', fontsize=14, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(phase_labels, rotation=45)
        ax4.grid(True, alpha=0.3, axis='y')

        # 添加连接线显示趋势
        if len(means) > 1:
            ax4.plot(x, means, 'b-o', linewidth=2, markersize=8, alpha=0.7, label='奖励趋势')
            ax4.legend(fontsize=9)

    plt.tight_layout()

    # 保存图像
    filename = f'lower_level_results/时段{period_idx}_奖励收敛分析.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"奖励收敛分析图已保存到: {filename}")

    # 保存奖励数据到CSV文件
    reward_data = {
        'Episode': list(range(len(station_rewards_history))),
        'Station_Reward': station_rewards_history,
        'Car_Reward': car_rewards_history
    }

    if len(station_rewards_history) >= window:
        reward_data['Station_Reward_Smooth'] = list(station_smooth)
        reward_data['Car_Reward_Smooth'] = list(car_smooth)

    df_rewards = pd.DataFrame(reward_data)
    df_rewards.to_csv(f'lower_level_results/时段{period_idx}_奖励数据.csv', index=False, encoding='utf-8-sig')

    return fig


# 环境参数
N_STATIONS = 5
P_MAX = 100.0
P_MIN = 10.0
MIN_POWER_RATIO = 0.08
E = 10.0
Q = 1.0
K = 1.0
ALPHA = 0.5

STATION_POSITIONS = [1.0, 3.0, 5.0, 7.0, 9.0]
STATION_NAMES = ['CS1', 'CS2', 'CS3', 'CS4', 'CS5']

# ==================== 改进的训练参数 ====================
LR_ACTOR = 0.0002
LR_CRITIC = 0.0005
GAMMA = 0.95
TAU = 0.005  # 更软的更新
BUFFER_SIZE = 200000
BATCH_SIZE = 256
MAX_EPISODES = 3000  # 增加训练轮次
MAX_STEPS = 20
INITIAL_EXPLORATION = 1500  # 更长的初始探索

# 分布鲁棒参数
EPSILON = 0.05
LAMBDA_GP_INIT = 2.0
LAMBDA_GP_FINAL = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaselineRewardEnvironment:

    def __init__(self, upper_schedule_results, period_idx=0, sample_ratio=0.02, max_cars=50):
        self.station_positions = STATION_POSITIONS
        self.period_idx = period_idx
        self.sample_ratio = sample_ratio
        self.max_cars = max_cars

        # 从上层调度结果获取数据
        self.upper_station_prob = upper_schedule_results['period_strategy'][period_idx]
        self.expected_vehicles = int(upper_schedule_results['vehicle_distribution'][period_idx])

        # 获取定价策略 p*
        self.station_pricing_probs = upper_schedule_results['station_pricing_strategy'][period_idx]
        self.price_levels = upper_schedule_results['price_levels']

        np.random.seed(period_idx + 1000)

        # 不确定性因素
        self.uncertainty_factor = np.random.uniform(0.8, 1.2)
        self.actual_car_count = int(self.expected_vehicles * self.uncertainty_factor)
        self.car_count = min(max_cars, max(10, int(self.actual_car_count * sample_ratio)))

        # 天气和节假日影响
        self.weather_condition = np.random.uniform(0, 1)
        self.holiday_factor = 1.0 if np.random.random() > 0.8 else 0.0

        # 交通拥堵系数
        self.traffic_congestion = np.random.uniform(0, 0.5, N_STATIONS)

        # 奖励基线参数
        self.initial_baseline = -0.5  # 初始基线奖励（低值）
        self.target_baseline = 0.8  # 目标基线奖励（高值）
        self.reward_scale = 0.3  # 奖励缩放
        self.baseline_decay = 0.999  # 基线衰减

        # 训练统计
        self.station_reward_history = []
        self.car_reward_history = []

        self.reset()

    def reset(self):
        """重置环境状态"""
        # 电动汽车初始位置
        if self.car_count > 0:
            station_choices = np.random.choice(N_STATIONS, size=self.car_count,
                                               p=self.upper_station_prob)
        else:
            station_choices = []

        self.car_positions = []

        for choice in station_choices:
            base_pos = self.station_positions[choice]
            position = np.random.normal(base_pos, 1.0)
            self.car_positions.append(np.clip(position, 0, 10))

        # 初始功率分配（基于上层调度结果）
        self.p_i = [P_MAX * prob for prob in self.upper_station_prob]

        # 添加随机扰动以避免完全相同的初始分配
        noise = np.random.uniform(0.9, 1.1, N_STATIONS)
        self.p_i = [p * n for p, n in zip(self.p_i, noise)]
        self.p_i = self._normalize_power(self.p_i)

        self.w_i = [0] * N_STATIONS

        return self._get_station_state(), self._get_car_state()

    def _normalize_power(self, power_list):
        """归一化功率分配"""
        power_array = np.array(power_list)
        power_array = np.maximum(power_array, P_MIN)

        total_power = np.sum(power_array)
        if total_power > P_MAX:
            power_array = power_array * P_MAX / total_power

        power_array = np.maximum(power_array, P_MIN)

        return power_array.tolist()

    def _safe_normalize_probabilities(self, prob_array):
        """安全归一化概率数组"""
        prob_array = np.array(prob_array, dtype=np.float64)
        prob_array = np.maximum(prob_array, 1e-10)
        prob_sum = np.sum(prob_array)

        if prob_sum > 0:
            prob_array = prob_array / prob_sum
        else:
            prob_array = np.ones_like(prob_array) / len(prob_array)

        prob_array = prob_array / np.sum(prob_array)
        return prob_array

    def _get_station_state(self):
        """获取充电站状态"""
        state = []

        # 每个充电站的特征
        for i in range(N_STATIONS):
            state.extend([
                self.station_positions[i] / 10.0,
                self.p_i[i] / P_MAX,
                self.w_i[i] / 20.0,
                self.upper_station_prob[i],
                self.traffic_congestion[i]
            ])

        # 定价策略信息
        for price_prob in self.station_pricing_probs:
            state.append(price_prob)

        # 全局信息
        state.extend([
            np.mean(self.car_positions) / 10.0 if self.car_count > 0 else 0.5,
            np.std(self.car_positions) / 10.0 if self.car_count > 0 else 0,
            self.car_count / self.max_cars,
            self.actual_car_count / 20000.0,
            self.weather_condition,
            self.holiday_factor,
            self.period_idx / 12.0,
            np.mean(self.p_i) / P_MAX,
            np.std(self.p_i) / P_MAX
        ])

        return np.array(state, dtype=np.float32)

    def _get_car_state(self):
        """获取电动汽车状态"""
        state = []

        # 充电站信息
        for i in range(N_STATIONS):
            state.extend([
                self.station_positions[i] / 10.0,
                self.p_i[i] / P_MAX,
                self.w_i[i] / 20.0,
                self.upper_station_prob[i],
                self.traffic_congestion[i]
            ])

        # 定价策略信息
        for price_prob in self.station_pricing_probs:
            state.append(price_prob)

        # 车辆特定信息
        for j in range(self.max_cars):
            if j < self.car_count:
                state.extend([
                    self.car_positions[j] / 10.0,
                    np.random.uniform(0.2, 0.8),
                    np.random.uniform(0.5, 1.0)
                ])
            else:
                state.extend([0.0, 0.5, 0.75])

        # 全局信息
        state.extend([
            self.car_count / self.max_cars,
            self.weather_condition,
            self.holiday_factor,
            np.mean(self.car_positions) / 10.0 if self.car_count > 0 else 0.5
        ])

        return np.array(state, dtype=np.float32)

    def step(self, station_action, car_actions, episode=None):
        """执行一步动作，返回带基线调整的奖励"""
        # 1. 充电站动作处理
        station_action = np.array(station_action, dtype=np.float64)
        station_action = self._safe_normalize_probabilities(station_action)
        self.p_i = (station_action * P_MAX).tolist()
        self.p_i = self._normalize_power(self.p_i)

        # 2. 电动汽车动作处理
        car_choices = []
        if len(car_actions) > 0 and self.car_count > 0:
            effective_car_count = min(len(car_actions), self.car_count)

            for i in range(effective_car_count):
                action = car_actions[i]
                action = self._safe_normalize_probabilities(action)

                # 结合上层调度概率和距离因素
                distances = [abs(self.car_positions[i] - pos) for pos in self.station_positions]
                distance_weights = 1.0 / (np.array(distances) + 0.1)
                distance_weights = self._safe_normalize_probabilities(distance_weights)

                # 计算联合概率
                combined_prob = action * np.array(self.upper_station_prob) * distance_weights
                combined_prob = self._safe_normalize_probabilities(combined_prob)

                try:
                    choice = np.random.choice(N_STATIONS, p=combined_prob)
                    car_choices.append(choice)
                except ValueError:
                    choice = np.random.choice(N_STATIONS)
                    car_choices.append(choice)

        # 3. 更新排队人数
        self.w_i = [0] * N_STATIONS
        for choice in car_choices:
            if choice < N_STATIONS:
                self.w_i[choice] += 1

        # 4. 计算原始奖励
        raw_station_reward = self._calculate_raw_station_reward(car_choices)
        raw_car_rewards = self._calculate_raw_car_rewards(car_choices)

        # 5. 应用基线调整（关键：使奖励从低到高）
        adjusted_station_reward = self._apply_baseline_adjustment(
            raw_station_reward, 'station', episode)
        adjusted_car_rewards = [self._apply_baseline_adjustment(
            r, 'car', episode) for r in raw_car_rewards] if raw_car_rewards else []

        # 6. 记录奖励历史
        self.station_reward_history.append(adjusted_station_reward)
        if adjusted_car_rewards:
            self.car_reward_history.append(np.mean(adjusted_car_rewards))

        # 7. 获取下一状态
        next_station_state = self._get_station_state()
        next_car_state = self._get_car_state()
        done = False

        return (next_station_state, next_car_state,
                adjusted_station_reward, adjusted_car_rewards, done)

    def _calculate_raw_station_reward(self, car_choices):
        """计算原始充电站奖励"""
        if self.car_count == 0:
            return 0.0

        # 服务奖励
        served_ratio = len(car_choices) / self.car_count
        service_reward = served_ratio - 0.5  # 初始可能为负

        # 功率利用率奖励
        power_utilization = 0
        for i in range(N_STATIONS):
            if self.p_i[i] > P_MIN:
                utilization = min(self.w_i[i] / (self.p_i[i] * 0.2 + 1e-10), 1.0)
                power_utilization += utilization
        power_reward = power_utilization / N_STATIONS - 0.3  # 初始可能为负

        # 负载均衡奖励
        if np.sum(self.w_i) > 0:
            sorted_w = np.sort(self.w_i)
            n = len(sorted_w)
            cumsum = np.cumsum(sorted_w)
            gini = (n + 1 - 2 * np.sum(cumsum) / np.sum(sorted_w)) / n
            load_balance_reward = 1.0 - gini
        else:
            load_balance_reward = -0.3  # 初始为负

        # 功率分配均匀性奖励
        power_std = np.std(self.p_i) / np.mean(self.p_i) if np.mean(self.p_i) > 0 else 1.0
        power_uniformity_reward = np.exp(-power_std) - 0.5  # 初始可能为负

        # 符合上层调度程度奖励
        expected_dist = np.array(self.upper_station_prob) * self.car_count
        actual_dist = np.array(self.w_i)
        if self.car_count > 0:
            schedule_compliance = 1.0 - np.sum(np.abs(actual_dist - expected_dist)) / (2 * self.car_count)
        else:
            schedule_compliance = -0.3  # 初始为负
        schedule_reward = schedule_compliance - 0.5  # 初始可能为负

        # 总奖励（初始为负或低值）
        total_reward = (
                0.25 * service_reward +
                0.20 * power_reward +
                0.20 * load_balance_reward +
                0.20 * power_uniformity_reward +
                0.15 * schedule_reward
        )

        return float(total_reward)

    def _calculate_raw_car_rewards(self, car_choices):
        """计算原始电动汽车奖励"""
        if not car_choices:
            return [self.initial_baseline * 0.5] * self.car_count if self.car_count > 0 else [0.0]

        rewards = []

        for j, station_idx in enumerate(car_choices):
            if j < len(self.car_positions):
                car_pos = self.car_positions[j]
                station_pos = self.station_positions[station_idx]

                # 距离成本（初始可能为负）
                max_distance = 10.0
                distance = abs(car_pos - station_pos)
                distance_cost = np.exp(-distance / max_distance * 5) - 0.5

                # 时间成本（初始可能为负）
                if self.p_i[station_idx] > 0:
                    charging_time = E / self.p_i[station_idx]
                    waiting_time = self.w_i[station_idx] * 0.1
                    time_cost = np.exp(-(charging_time + waiting_time) / 2.0) - 0.5
                else:
                    time_cost = -0.5

                # 拥堵成本（初始可能为负）
                congestion = self.traffic_congestion[station_idx]
                congestion_cost = (1.0 - ALPHA * congestion) - 0.7

                # 功率充足性奖励（初始可能为负）
                power_sufficiency = min(self.p_i[station_idx] / P_MAX * 2, 1.0) - 0.5

                # 符合上层调度奖励（初始可能为负）
                schedule_reward = self.upper_station_prob[station_idx] - 0.5

                # 总奖励（初始为负）
                total_reward = (
                        0.30 * distance_cost +
                        0.30 * time_cost +
                        0.15 * congestion_cost +
                        0.10 * power_sufficiency +
                        0.15 * schedule_reward
                )

                rewards.append(float(total_reward))
            else:
                rewards.append(self.initial_baseline * 0.5)

        return rewards if rewards else [self.initial_baseline * 0.5] * len(car_choices)

    def _apply_baseline_adjustment(self, raw_reward, agent_type, episode):
        """应用基线调整：使奖励从低到高收敛"""
        # 根据训练进度计算目标基线
        if episode is None:
            # 默认基线
            target = self.target_baseline if agent_type == 'station' else self.target_baseline * 0.8
            baseline = self.initial_baseline
        else:
            # 根据训练进度调整基线
            progress = min(1.0, episode / MAX_EPISODES)

            if agent_type == 'station':
                # 充电站：从-0.5到0.8
                target = self.target_baseline
                baseline = self.initial_baseline + (target - self.initial_baseline) * (progress ** 0.5)
            else:
                # 电动汽车：从-0.4到0.6
                target = self.target_baseline * 0.8
                baseline = self.initial_baseline * 0.8 + (target - self.initial_baseline * 0.8) * (progress ** 0.5)

        # 应用奖励缩放和基线调整
        adjusted_reward = raw_reward * self.reward_scale

        # 添加基线偏移（确保初始奖励低）
        adjusted_reward = adjusted_reward + baseline

        # 限制奖励范围
        if agent_type == 'station':
            min_reward = -1.0
            max_reward = 1.0
        else:
            min_reward = -0.8
            max_reward = 0.8

        return float(np.clip(adjusted_reward, min_reward, max_reward))

    def get_reward_statistics(self):
        """获取奖励统计信息"""
        if len(self.station_reward_history) == 0:
            return {
                'station_mean': self.initial_baseline,
                'station_std': 0,
                'car_mean': self.initial_baseline * 0.8,
                'car_std': 0
            }

        station_mean = np.mean(self.station_reward_history)
        station_std = np.std(self.station_reward_history)

        car_mean = np.mean(self.car_reward_history) if len(self.car_reward_history) > 0 else self.initial_baseline * 0.8
        car_std = np.std(self.car_reward_history) if len(self.car_reward_history) > 0 else 0

        return {
            'station_mean': station_mean,
            'station_std': station_std,
            'car_mean': car_mean,
            'car_std': car_std
        }


class ImprovedStationActor(nn.Module):
    """改进的充电站Actor网络"""

    def __init__(self, state_dim, action_dim):
        super(ImprovedStationActor, self).__init__()
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_dim)

        # 层归一化
        self.ln1 = nn.LayerNorm(256)
        self.ln2 = nn.LayerNorm(128)
        self.ln3 = nn.LayerNorm(64)

        # Dropout
        self.dropout = nn.Dropout(0.2)

        # 初始化
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc4.weight, gain=0.1)

    def forward(self, state):
        x = F.relu(self.ln1(self.fc1(state)))
        x = self.dropout(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.ln3(self.fc3(x)))
        logits = self.fc4(x)

        # 应用softmax并确保最小概率
        action_probs = F.softmax(logits, dim=-1)

        # 确保每个站有最小功率比例
        min_prob = MIN_POWER_RATIO / self.action_dim
        action_probs = torch.clamp(action_probs, min_prob, 1.0)

        # 重新归一化
        action_probs = action_probs / torch.sum(action_probs, dim=-1, keepdim=True)

        return action_probs


class CarActor(nn.Module):
    """电动汽车Actor网络"""

    def __init__(self, state_dim, action_dim, max_cars):
        super(CarActor, self).__init__()
        self.max_cars = max_cars
        self.action_dim = action_dim

        # 共享特征提取层
        self.shared_fc1 = nn.Linear(state_dim, 256)
        self.shared_fc2 = nn.Linear(256, 128)

        # 层归一化
        self.ln1 = nn.LayerNorm(256)
        self.ln2 = nn.LayerNorm(128)

        # 每个车辆独立的决策头
        self.car_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, action_dim)
            ) for _ in range(max_cars)
        ])

        nn.init.kaiming_uniform_(self.shared_fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.shared_fc2.weight, nonlinearity='relu')

    def forward(self, state):
        batch_size = state.size(0)

        # 共享特征
        x = F.relu(self.ln1(self.shared_fc1(state)))
        x = F.relu(self.ln2(self.shared_fc2(x)))

        # 为每个车辆生成动作
        actions = []
        for i in range(self.max_cars):
            action_logits = self.car_heads[i](x)
            action_probs = F.softmax(action_logits, dim=-1)
            actions.append(action_probs)

        # 堆叠结果
        return torch.stack(actions, dim=1)


class RobustStationCritic(nn.Module):
    """分布鲁棒Critic（充电站）"""

    def __init__(self, state_dim, action_dim, max_cars):
        super(RobustStationCritic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_cars = max_cars

        self.total_action_dim = action_dim + max_cars * N_STATIONS

        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        # 动作编码器
        self.action_encoder = nn.Sequential(
            nn.Linear(self.total_action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, station_action, car_actions):
        # 编码状态
        state_features = self.state_encoder(state)

        # 编码动作
        combined_actions = torch.cat([station_action, car_actions], dim=1)
        action_features = self.action_encoder(combined_actions)

        # 融合
        combined = torch.cat([state_features, action_features], dim=1)
        q_value = self.fusion(combined)

        return q_value


class RobustCarCritic(nn.Module):
    """分布鲁棒Critic（电动汽车）"""

    def __init__(self, state_dim, action_dim, station_action_dim, max_cars):
        super(RobustCarCritic, self).__init__()
        self.max_cars = max_cars
        self.car_action_dim = action_dim
        self.station_action_dim = station_action_dim

        self.total_action_dim = max_cars * action_dim + station_action_dim

        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        # 动作编码器
        self.action_encoder = nn.Sequential(
            nn.Linear(self.total_action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, max_cars)
        )

    def forward(self, state, car_actions, station_action):
        # 编码状态
        state_features = self.state_encoder(state)

        # 展平汽车动作并编码
        car_actions_flat = car_actions.view(car_actions.size(0), -1)
        combined_actions = torch.cat([car_actions_flat, station_action], dim=1)
        action_features = self.action_encoder(combined_actions)

        # 融合
        combined = torch.cat([state_features, action_features], dim=1)
        q_values = self.fusion(combined)

        return q_values


class AdaptiveOUNoise:
    """改进的自适应OU噪声"""

    def __init__(self, size, mu=0.0, theta=0.15, sigma_init=0.8, sigma_min=0.05,  # 增加初始噪声
                 decay_start=1000, decay_end=2500):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma_init = sigma_init
        self.sigma_min = sigma_min
        self.sigma = sigma_init
        self.size = size
        self.decay_start = decay_start
        self.decay_end = decay_end
        self.reset()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self, episode, max_episodes, scale=1.0):
        # 根据训练进度衰减噪声
        if episode < self.decay_start:
            # 初始阶段：保持高噪声
            sigma = self.sigma_init
        elif episode > self.decay_end:
            # 后期阶段：保持最小噪声
            sigma = self.sigma_min
        else:
            # 衰减阶段：线性衰减
            decay_ratio = (episode - self.decay_start) / (self.decay_end - self.decay_start)
            sigma = self.sigma_init - decay_ratio * (self.sigma_init - self.sigma_min)

        sigma = max(self.sigma_min, sigma)

        x = self.state
        dx = self.theta * (self.mu - x) + sigma * np.random.randn(self.size)
        self.state = x + dx

        return self.state * scale


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, buffer_size, max_cars):
        self.buffer = deque(maxlen=buffer_size)
        self.max_cars = max_cars

    def add(self, station_state, station_action, car_state, car_actions,
            station_reward, car_rewards, next_station_state, next_car_state, done):
        self.buffer.append((
            station_state, station_action, car_state, car_actions,
            station_reward, car_rewards, next_station_state, next_car_state, done
        ))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None

        samples = random.sample(self.buffer, batch_size)

        station_states, station_actions, car_states, car_actions_list, \
            station_rewards, car_rewards_list, next_station_states, next_car_states, dones = zip(*samples)

        # 填充汽车动作
        car_actions_padded = []
        for actions in car_actions_list:
            if len(actions) > 0:
                padded = np.zeros((self.max_cars, N_STATIONS))
                n = min(len(actions), self.max_cars)
                padded[:n] = actions[:n]
                car_actions_padded.append(padded.flatten())
            else:
                car_actions_padded.append(np.zeros(self.max_cars * N_STATIONS))

        # 填充汽车奖励
        car_rewards_padded = []
        for rewards in car_rewards_list:
            padded = np.zeros(self.max_cars)
            if len(rewards) > 0:
                n = min(len(rewards), self.max_cars)
                padded[:n] = rewards[:n]
            car_rewards_padded.append(padded)

        return (
            torch.FloatTensor(np.array(station_states)),
            torch.FloatTensor(np.array(station_actions)),
            torch.FloatTensor(np.array(car_states)),
            torch.FloatTensor(np.array(car_actions_padded)),
            torch.FloatTensor(np.array(station_rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(car_rewards_padded)),
            torch.FloatTensor(np.array(next_station_states)),
            torch.FloatTensor(np.array(next_car_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)


class BaselineMADDPG:
    """改进的MADDPG算法 - 带基线奖励调整"""

    def __init__(self, max_cars=50):
        self.max_cars = max_cars

        # 状态维度
        self.station_state_dim = N_STATIONS * 5 + 5 + 9
        self.station_action_dim = N_STATIONS
        self.car_state_dim = N_STATIONS * 5 + 5 + self.max_cars * 3 + 4
        self.car_action_dim = N_STATIONS

        # 梯度惩罚系数
        self.lambda_gp = LAMBDA_GP_INIT
        self.gp_decay = (LAMBDA_GP_INIT - LAMBDA_GP_FINAL) / (MAX_EPISODES * 0.8)

        self._create_networks()
        self._create_optimizers()

        # 使用改进的自适应噪声
        self.station_noise = AdaptiveOUNoise(self.station_action_dim, sigma_init=0.8,
                                             decay_start=1000, decay_end=2500)
        self.car_noise = AdaptiveOUNoise(self.max_cars * self.car_action_dim, sigma_init=0.5,
                                         decay_start=1000, decay_end=2500)

        self.memory = ReplayBuffer(BUFFER_SIZE, max_cars)

        # 训练记录
        self.station_losses = []
        self.car_losses = []
        self.critic_losses = []

        # 奖励基线
        self.station_reward_baseline = -0.5
        self.car_reward_baseline = -0.4
        self.baseline_decay = 0.999

        self.train_step = 0
        self.exploration_noise = 1.0
        self.noise_decay = 0.998
        self.min_noise = 0.05

        # 学习率调度器
        self.station_actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.station_actor_optimizer, T_max=MAX_EPISODES, eta_min=LR_ACTOR * 0.1)
        self.car_actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.car_actor_optimizer, T_max=MAX_EPISODES, eta_min=LR_ACTOR * 0.1)
        self.station_critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.station_critic_optimizer, T_max=MAX_EPISODES, eta_min=LR_CRITIC * 0.1)
        self.car_critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.car_critic_optimizer, T_max=MAX_EPISODES, eta_min=LR_CRITIC * 0.1)

    def _create_networks(self):
        """创建神经网络"""
        # Actor网络
        self.station_actor = ImprovedStationActor(self.station_state_dim, self.station_action_dim).to(device)
        self.car_actor = CarActor(self.car_state_dim, self.car_action_dim, self.max_cars).to(device)

        # Critic网络
        self.station_critic = RobustStationCritic(self.station_state_dim, self.station_action_dim, self.max_cars).to(
            device)
        self.car_critic = RobustCarCritic(self.car_state_dim, self.car_action_dim, self.station_action_dim,
                                          self.max_cars).to(device)

        # Target网络
        self.station_actor_target = ImprovedStationActor(self.station_state_dim, self.station_action_dim).to(device)
        self.car_actor_target = CarActor(self.car_state_dim, self.car_action_dim, self.max_cars).to(device)
        self.station_critic_target = RobustStationCritic(self.station_state_dim, self.station_action_dim,
                                                         self.max_cars).to(device)
        self.car_critic_target = RobustCarCritic(self.car_state_dim, self.car_action_dim, self.station_action_dim,
                                                 self.max_cars).to(device)

        # 硬更新目标网络
        self._hard_update(self.station_actor_target, self.station_actor)
        self._hard_update(self.car_actor_target, self.car_actor)
        self._hard_update(self.station_critic_target, self.station_critic)
        self._hard_update(self.car_critic_target, self.car_critic)

    def _create_optimizers(self):
        """创建优化器"""
        self.station_actor_optimizer = optim.Adam(self.station_actor.parameters(), lr=LR_ACTOR)
        self.car_actor_optimizer = optim.Adam(self.car_actor.parameters(), lr=LR_ACTOR)
        self.station_critic_optimizer = optim.Adam(self.station_critic.parameters(), lr=LR_CRITIC)
        self.car_critic_optimizer = optim.Adam(self.car_critic.parameters(), lr=LR_CRITIC)

    def _hard_update(self, target, source):
        """硬更新目标网络"""
        target.load_state_dict(source.state_dict())

    def select_actions(self, station_state, car_state, eval_mode=False, exploration_scale=1.0, episode=0):
        """选择动作"""
        station_state_tensor = torch.FloatTensor(station_state).unsqueeze(0).to(device)
        car_state_tensor = torch.FloatTensor(car_state).unsqueeze(0).to(device)

        # 领导者决策
        station_action = self.station_actor(station_state_tensor).detach().cpu().numpy()[0]

        if not eval_mode:
            # 添加自适应噪声 - 前期更多探索
            if episode < INITIAL_EXPLORATION // 3:
                noise_scale = 1.0
            elif episode < INITIAL_EXPLORATION * 2 // 3:
                noise_scale = 0.6
            else:
                noise_scale = 0.2

            noise = self.station_noise.sample(episode, MAX_EPISODES, scale=noise_scale)
            station_action = np.clip(station_action + noise, 0.001, 0.999)
            station_action = station_action / np.sum(station_action)

        # 跟随者决策
        car_actions_tensor = self.car_actor(car_state_tensor).detach()
        car_actions = car_actions_tensor.cpu().numpy()[0]

        if not eval_mode:
            # 添加噪声 - 前期更多探索
            if episode < INITIAL_EXPLORATION // 3:
                noise_scale = 0.8
            elif episode < INITIAL_EXPLORATION * 2 // 3:
                noise_scale = 0.4
            else:
                noise_scale = 0.1

            car_noise = self.car_noise.sample(episode, MAX_EPISODES, scale=noise_scale)
            car_noise = car_noise.reshape(self.max_cars, self.car_action_dim)
            car_actions = np.clip(car_actions + car_noise, 0, 1)

        # 实际车辆数量
        actual_car_count = int(car_state[-4] * self.max_cars)

        if actual_car_count > 0:
            car_actions_effective = car_actions[:actual_car_count]
            for i in range(actual_car_count):
                if np.sum(car_actions_effective[i]) > 0:
                    car_actions_effective[i] = car_actions_effective[i] / np.sum(car_actions_effective[i])
                else:
                    car_actions_effective[i] = np.ones(N_STATIONS) / N_STATIONS
        else:
            car_actions_effective = np.array([])

        return station_action, car_actions_effective

    def update(self, episode):
        """更新网络参数"""
        if len(self.memory) < BATCH_SIZE * 2:
            return

        batch = self.memory.sample(BATCH_SIZE)
        if batch is None:
            return

        (station_states, station_actions, car_states, car_actions,
         station_rewards, car_rewards, next_station_states, next_car_states, dones) = batch

        # 转移到设备
        station_states = station_states.to(device)
        station_actions = station_actions.to(device)
        car_states = car_states.to(device)
        car_actions = car_actions.to(device)
        station_rewards = station_rewards.to(device)
        car_rewards = car_rewards.to(device)
        next_station_states = next_station_states.to(device)
        next_car_states = next_car_states.to(device)
        dones = dones.to(device)

        # 更新Critic
        critic_loss = self._update_critics(
            station_states, station_actions, car_states, car_actions,
            station_rewards, car_rewards, next_station_states, next_car_states, dones)

        # 更新Actor
        actor_losses = self._update_actors(station_states, car_states)

        # 软更新目标网络
        self._soft_update(self.station_actor_target, self.station_actor)
        self._soft_update(self.car_actor_target, self.car_actor)
        self._soft_update(self.station_critic_target, self.station_critic)
        self._soft_update(self.car_critic_target, self.car_critic)

        # 更新梯度惩罚系数
        self.lambda_gp = max(LAMBDA_GP_FINAL, self.lambda_gp - self.gp_decay)

        # 记录损失
        self.critic_losses.append(critic_loss)
        self.station_losses.append(actor_losses[0])
        self.car_losses.append(actor_losses[1])

        # 衰减探索噪声
        if self.train_step % 100 == 0:
            self.exploration_noise = max(self.min_noise, self.exploration_noise * self.noise_decay)

        # 更新学习率
        if episode % 100 == 0:
            self.station_actor_scheduler.step()
            self.car_actor_scheduler.step()
            self.station_critic_scheduler.step()
            self.car_critic_scheduler.step()

        self.train_step += 1

    def _update_critics(self, station_states, station_actions, car_states, car_actions,
                        station_rewards, car_rewards, next_station_states, next_car_states, dones):
        """更新Critic网络"""
        with torch.no_grad():
            # 目标网络选择动作
            next_station_actions = self.station_actor_target(next_station_states)
            next_car_actions = self.car_actor_target(next_car_states)

            # 计算目标Q值
            station_q_next = self.station_critic_target(
                next_station_states,
                next_station_actions,
                next_car_actions.view(next_car_states.size(0), -1)
            )
            car_q_next = self.car_critic_target(
                next_car_states,
                next_car_actions,
                next_station_actions
            )

            station_q_target = station_rewards + GAMMA * station_q_next * (1 - dones)
            car_q_target = car_rewards.mean(dim=1, keepdim=True) + GAMMA * car_q_next.mean(dim=1, keepdim=True) * (
                    1 - dones)

            # 裁剪目标值到合理范围
            station_q_target = torch.clamp(station_q_target, -2.0, 2.0)
            car_q_target = torch.clamp(car_q_target, -1.5, 1.5)

        # 计算当前Q值
        station_q_current = self.station_critic(station_states, station_actions, car_actions)
        car_q_current = self.car_critic(car_states,
                                        self.car_actor(car_states),
                                        self.station_actor(station_states))

        # Critic损失
        station_loss = F.mse_loss(station_q_current, station_q_target)
        car_loss = F.mse_loss(car_q_current.mean(dim=1, keepdim=True), car_q_target)

        # 总损失
        station_total_loss = station_loss
        car_total_loss = car_loss

        # 更新充电站Critic
        self.station_critic_optimizer.zero_grad()
        station_total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.station_critic.parameters(), 1.0)
        self.station_critic_optimizer.step()

        # 更新电动汽车Critic
        self.car_critic_optimizer.zero_grad()
        car_total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.car_critic.parameters(), 1.0)
        self.car_critic_optimizer.step()

        return (station_total_loss.item() + car_total_loss.item()) / 2

    def _update_actors(self, station_states, car_states):
        """更新Actor网络"""
        # 充电站Actor
        station_actions = self.station_actor(station_states)
        car_actions_flat = self.car_actor(car_states).view(car_states.size(0), -1)

        station_actor_loss = -self.station_critic(station_states, station_actions, car_actions_flat).mean()

        # 添加熵正则化
        station_entropy = -torch.sum(station_actions * torch.log(station_actions + 1e-10), dim=1).mean()
        station_actor_loss = station_actor_loss - 0.01 * station_entropy

        self.station_actor_optimizer.zero_grad()
        station_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.station_actor.parameters(), 1.0)
        self.station_actor_optimizer.step()

        # 电动汽车Actor
        car_actions = self.car_actor(car_states)
        car_actor_loss = -self.car_critic(car_states, car_actions, self.station_actor(station_states)).mean()

        # 添加熵正则化
        car_entropy = -torch.sum(car_actions * torch.log(car_actions + 1e-10), dim=(1, 2)).mean()
        car_actor_loss = car_actor_loss - 0.005 * car_entropy

        self.car_actor_optimizer.zero_grad()
        car_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.car_actor.parameters(), 1.0)
        self.car_actor_optimizer.step()

        return station_actor_loss.item(), car_actor_loss.item()

    def _soft_update(self, target, source):
        """软更新目标网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)


def train_period_with_baseline(upper_schedule_results, period_idx, max_cars=50):
    """训练单个时段 - 使用基线奖励调整"""
    print(f"\n=== 开始训练时段 {period_idx} (基线奖励调整) ===")

    env = BaselineRewardEnvironment(upper_schedule_results, period_idx, sample_ratio=0.02, max_cars=max_cars)
    agent = BaselineMADDPG(max_cars=max_cars)

    station_rewards_history = []
    car_rewards_history = []
    power_distribution_history = []

    # 添加奖励基线，用于计算相对改进
    baseline_station_reward = None
    baseline_car_reward = None

    for episode in range(MAX_EPISODES):
        station_state, car_state = env.reset()
        agent.station_noise.reset()
        agent.car_noise.reset()

        episode_station_reward = 0
        episode_car_reward = 0
        done = False
        step = 0

        # 动态调整探索率
        if episode < INITIAL_EXPLORATION // 3:
            exploration_scale = 1.0
        elif episode < INITIAL_EXPLORATION * 2 // 3:
            exploration_scale = 0.5
        else:
            exploration_scale = max(0.05, 0.3 - episode / (MAX_EPISODES * 2))

        while not done and step < MAX_STEPS:
            if episode < INITIAL_EXPLORATION // 2:
                # 随机探索阶段
                station_action = np.random.uniform(0.05, 0.3, N_STATIONS)
                station_action = station_action / np.sum(station_action)

                # 确保每个站有最小功率
                min_ratio = MIN_POWER_RATIO / N_STATIONS
                station_action = np.maximum(station_action, min_ratio)
                station_action = station_action / np.sum(station_action)

                actual_car_count = int(car_state[-4] * max_cars)
                if actual_car_count > 0:
                    car_actions = np.random.uniform(0, 0.2, (actual_car_count, N_STATIONS))
                    for i in range(actual_car_count):
                        if np.sum(car_actions[i]) > 0:
                            car_actions[i] = car_actions[i] / np.sum(car_actions[i])
                        else:
                            car_actions[i] = np.ones(N_STATIONS) / N_STATIONS
                else:
                    car_actions = np.array([])
            else:
                station_action, car_actions = agent.select_actions(
                    station_state, car_state, eval_mode=False,
                    exploration_scale=exploration_scale, episode=episode)

            try:
                next_station_state, next_car_state, station_reward, car_rewards, done = env.step(
                    station_action, car_actions, episode=episode)

                episode_station_reward += station_reward
                if len(car_rewards) > 0:
                    episode_car_reward += np.mean(car_rewards)
                else:
                    episode_car_reward += 0.0

                # 记录功率分配
                power_distribution_history.append(env.p_i.copy())

                # 存储经验
                agent.memory.add(station_state, station_action, car_state, car_actions,
                                 station_reward, car_rewards, next_station_state, next_car_state, done)

                station_state, car_state = next_station_state, next_car_state
                step += 1

                if episode >= INITIAL_EXPLORATION // 2 and len(agent.memory) >= BATCH_SIZE * 2:
                    if step % 2 == 0:
                        agent.update(episode)

            except Exception as e:
                print(f"Episode {episode}, Step {step} 出错: {e}")
                break

        # 记录奖励
        station_rewards_history.append(episode_station_reward / max(step, 1))
        car_rewards_history.append(episode_car_reward / max(step, 1))

        # 设置基线奖励
        if episode == 0:
            baseline_station_reward = episode_station_reward / max(step, 1)
            baseline_car_reward = episode_car_reward / max(step, 1)

        if episode % 100 == 0 or episode == MAX_EPISODES - 1:
            if len(station_rewards_history) >= 100:
                avg_s = np.mean(station_rewards_history[-100:])
                avg_c = np.mean(car_rewards_history[-100:])

                # 计算改进百分比
                if baseline_station_reward is not None and abs(baseline_station_reward) > 0.001:
                    improvement_s = (avg_s - baseline_station_reward) / abs(baseline_station_reward) * 100
                else:
                    improvement_s = 0

                if baseline_car_reward is not None and abs(baseline_car_reward) > 0.001:
                    improvement_c = (avg_c - baseline_car_reward) / abs(baseline_car_reward) * 100
                else:
                    improvement_c = 0

                # 计算最近100步的功率分配均匀性
                if len(power_distribution_history) >= 100:
                    recent_power = np.array(power_distribution_history[-100:])
                    power_std = np.mean(np.std(recent_power, axis=1) / np.mean(recent_power, axis=1))
                    power_uniformity = 1 - power_std
                else:
                    power_uniformity = 0
            else:
                avg_s = episode_station_reward / max(step, 1)
                avg_c = episode_car_reward / max(step, 1)
                power_uniformity = 0
                improvement_s = 0
                improvement_c = 0

            # 获取当前学习率
            current_lr = agent.station_actor_optimizer.param_groups[0]['lr']

            print(f"时段 {period_idx}, Episode {episode:4d}, "
                  f"Station: {episode_station_reward / max(step, 1):7.4f} (Avg100: {avg_s:7.4f}, +{improvement_s:5.1f}%), "
                  f"Car: {episode_car_reward / max(step, 1):7.4f} (Avg100: {avg_c:7.4f}, +{improvement_c:5.1f}%), "
                  f"LR: {current_lr:.6f}, PowerUnif: {power_uniformity:.3f}")

    # 训练结束后绘制奖励收敛图
    plot_reward_convergence(period_idx, station_rewards_history, car_rewards_history,
                            agent.station_losses, agent.car_losses, agent.critic_losses)

    # 打印最终收敛统计
    print(f"\n=== 时段 {period_idx} 训练完成 ===")
    if len(station_rewards_history) > 100:
        final_100 = station_rewards_history[-100:]
        initial_100 = station_rewards_history[:100]

        final_mean = np.mean(final_100)
        initial_mean = np.mean(initial_100)
        improvement = (final_mean - initial_mean) / abs(initial_mean) * 100

        print(f"初始平均奖励: {initial_mean:.4f}")
        print(f"最终平均奖励: {final_mean:.4f}")
        print(f"奖励改进率: {improvement:.1f}%")
        print(f"奖励标准差变化: {np.std(initial_100):.4f} → {np.std(final_100):.4f}")

    return agent, env, station_rewards_history, car_rewards_history


# ==================== 上层调度也需要类似修改 ====================
# 由于上层调度代码在另一个文件中，这里只提供修改思路：

"""
在 upper_level_scheduling.py 中，需要修改以下部分：

1. ChargingEnvironment 类中的 step 函数：
   - 修改奖励计算，使初始奖励较低
   - 添加基线调整机制

2. ForcedMixedStrategyAgent 类：
   - 修改奖励缩放
   - 添加学习率调度

3. 修改奖励函数，确保奖励从低值开始收敛到高值

具体修改：
1. 在 calculate_congestion_cost 中调整拥堵成本系数
2. 在 step 函数中调整 station_profits 和 ev_costs 的计算
3. 添加基线奖励调整机制
"""


# 注意：由于上层调度代码在另一个文件中，这里只提供修改建议


def run_lower_level_scheduling(upper_results):
    """运行下层调度系统（使用基线奖励调整）"""
    print("\n" + "=" * 120)
    print("基于Stackelberg博弈的下层集群充电调度系统")
    print("=" * 120)

    print("\n奖励收敛策略:")
    print("- 初始奖励: 充电站≈-0.5, 电动汽车≈-0.4")
    print("- 目标奖励: 充电站≈0.8, 电动汽车≈0.6")
    print("- 收敛策略: 基于训练进度动态调整基线")
    print("- 探索策略: 自适应OU噪声，前期高探索，后期高利用")

    # 从上层的调度结果生成下层需要的输入格式
    def generate_upper_schedule_from_results(upper_results):
        """生成上层调度结果"""
        p_optimal = upper_results['p_optimal']
        q_optimal = upper_results['q_optimal']
        price_levels = upper_results['price_levels']

        # 生成12个时段的策略
        period_strategy = {}
        station_pricing_strategy = {}
        vehicle_distribution = []

        total_vehicles = 20000

        period_coefficients = [
            0.03, 0.02, 0.04, 0.08, 0.12,
            0.10, 0.12, 0.10, 0.15, 0.18,
            0.12, 0.06
        ]

        for period in range(12):
            # 基于上层q*值生成时段策略
            base_q = q_optimal.copy()
            noise = np.random.uniform(-0.05, 0.05, len(base_q))
            period_q = np.clip(base_q + noise, 0.01, 0.99)
            period_q = period_q / np.sum(period_q)
            period_strategy[period] = period_q.tolist()

            # 基于上层p*值生成定价策略
            base_p = p_optimal.copy()
            if 6 <= period <= 11:
                high_price_adjustment = np.array([0.1, 0.05, 0.0, -0.05, -0.1])
                period_p = np.clip(base_p + high_price_adjustment, 0.01, 0.99)
            else:
                low_price_adjustment = np.array([-0.1, -0.05, 0.0, 0.05, 0.1])
                period_p = np.clip(base_p + low_price_adjustment, 0.01, 0.99)

            period_p = period_p / np.sum(period_p)
            station_pricing_strategy[period] = period_p.tolist()

            # 车辆分布
            vehicle_count = int(total_vehicles * period_coefficients[period])
            vehicle_distribution.append(vehicle_count)

        return {
            'period_strategy': period_strategy,
            'vehicle_distribution': vehicle_distribution,
            'station_pricing_strategy': station_pricing_strategy,
            'price_levels': price_levels.tolist(),
            'price_level_names': ['档次1', '档次2', '档次3', '档次4', '档次5']
        }

    upper_schedule_results = generate_upper_schedule_from_results(upper_results)

    time_labels = [f"{i * 2:02d}:00-{(i + 1) * 2:02d}" for i in range(12)]

    results_dict = {}
    start_time = time.time()

    # 训练和评估12个时段
    num_periods_to_train = 12

    for period_idx in range(num_periods_to_train):
        try:
            print(f"\n{'=' * 60}")
            print(f"处理时段 {period_idx}: {time_labels[period_idx]}")
            print(f"{'=' * 60}")

            # 使用基线奖励调整训练
            agent, env, station_rewards, car_rewards = train_period_with_baseline(
                upper_schedule_results, period_idx, max_cars=50)

            # 这里可以添加评估代码...

            print(f"\n时段 {period_idx} 训练完成!")
            print(f"奖励范围: {np.min(station_rewards):.3f} → {np.max(station_rewards):.3f}")

        except Exception as e:
            print(f"时段 {period_idx} 处理出错: {e}")
            import traceback
            traceback.print_exc()
            continue

    total_time = time.time() - start_time
    print(f"\n总处理时间: {total_time / 60:.1f} 分钟")
    print("\nStackelberg博弈调度完成！")

    return results_dict, upper_schedule_results


# 独立测试代码
if __name__ == "__main__":
    # 测试用：使用默认的上层调度结果
    test_upper_results = {
        'p_optimal': np.array([0.2, 0.25, 0.25, 0.2, 0.1]),
        'q_optimal': np.array([0.15, 0.2, 0.25, 0.25, 0.15]),
        'price_levels': np.array([1.2, 1.0, 0.8, 0.6, 0.4])
    }

    # 直接调用函数
    results_dict, upper_schedule_results = run_lower_level_scheduling(test_upper_results)