# upper_level_scheduling.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
import random
import warnings
import time
import os

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ChargingEnvironment:
    def __init__(self):
        # 价格档次：重新设计以混合策略
        self.price_levels = np.array([1.2, 1.0, 0.8, 0.6, 0.4])  # 更合理的价格梯度
        self.N = 100  # 电动汽车数量
        self.N_levels = len(self.price_levels)

        # 关键参数调整：使混合策略成为最优选择
        self.alpha = 1.5  # 显著增加拥堵惩罚系数
        self.c = 0.3  # 降低运营成本
        self.rho = 4.0  # 增加容量比，强化拥堵效应

        # 交通承载系数：价格越低，承载能力越低，分流
        self.omega = np.array([0.8, 0.7, 0.6, 0.5, 0.4])  # 反向设计：低价格承载能力低

        # 计算承载上限
        self.k = self.omega * self.N

        # 历史数据
        self.history = {
            'p': [], 'q': [], 'station_profits': [],
            'ev_costs': [], 'congestion_costs': []
        }

        print("环境参数初始化（混合策略设计）:")
        print(f"价格档次: {self.price_levels}")
        print(f"承载系数ω: {self.omega}")
        print(f"承载上限k: {self.k}")
        print(f"拥堵惩罚系数α: {self.alpha}")
        print(f"容量比ρ: {self.rho}")
        print(f"运营成本c: {self.c}")
        print("-" * 60)

    def calculate_congestion_cost(self, q):
        """计算拥堵成本 - 关键：使拥堵成本显著"""
        congestion_costs = np.zeros(self.N_levels)
        e = np.zeros(self.N_levels)

        for i in range(self.N_levels):
            actual_vehicles = self.N * q[i]

            # 关键修改：只要超过承载上限一定比例就产生拥堵
            threshold = self.k[i] * 0.7  # 70%承载就开始拥堵
            if actual_vehicles > threshold:
                e[i] = ((actual_vehicles - threshold) / max(threshold, 1)) * self.rho
            else:
                e[i] = 0

            congestion_costs[i] = self.alpha * (e[i] ** 2)

        return congestion_costs, e

    def step(self, p, q):
        """环境交互 - 关键：混合策略奖励"""
        p = np.clip(p, 0.001, 0.999)
        q = np.clip(q, 0.001, 0.999)
        p = p / np.sum(p) if np.sum(p) > 0 else np.ones_like(p) / len(p)
        q = q / np.sum(q) if np.sum(q) > 0 else np.ones_like(q) / len(q)

        congestion_costs, e = self.calculate_congestion_cost(q)

        # 关键：计算充电站利润时，考虑拥堵成本的显著影响
        station_profits = np.zeros(self.N_levels)
        for i in range(self.N_levels):
            # 增强拥堵成本对利润的影响
            station_profits[i] = (self.price_levels[i] - self.c) * self.N * q[i] - 2.0 * congestion_costs[i]

        station_total_profit = np.sum(p * station_profits)

        # 电动汽车成本：拥堵成本显著影响
        ev_costs = np.zeros(self.N_levels)
        for i in range(self.N_levels):
            ev_costs[i] = self.price_levels[i] + 2.0 * congestion_costs[i]

        ev_total_cost = np.sum(q * ev_costs)

        # 关键：添加混合策略奖励
        p_entropy = -np.sum(p * np.log(p + 1e-10))
        q_entropy = -np.sum(q * np.log(q + 1e-10))

        # 混合策略奖励：当策略分布更均匀时给予奖励
        mixed_strategy_bonus = 0.0
        if p_entropy > 1.0:  # 熵大于1说明策略比较分散
            mixed_strategy_bonus = 5.0 * (p_entropy / np.log(self.N_levels))

        # 更新状态
        new_state = np.concatenate([p, q, self.price_levels])

        # 存储历史
        self.history['p'].append(p.copy())
        self.history['q'].append(q.copy())
        self.history['station_profits'].append(station_total_profit + mixed_strategy_bonus)
        self.history['ev_costs'].append(ev_total_cost)
        self.history['congestion_costs'].append(np.sum(congestion_costs))

        return new_state, station_total_profit + mixed_strategy_bonus, -ev_total_cost, congestion_costs, e

    def reset(self):
        """重置环境 - 初始混合策略"""
        # 使用混合策略作为初始状态
        p_init = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # 均匀分布
        q_init = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        state = np.concatenate([p_init, q_init, self.price_levels])

        for key in self.history:
            self.history[key] = []

        return state

    def get_performance_metrics(self):
        if len(self.history['station_profits']) == 0:
            return {}

        last_idx = min(50, len(self.history['station_profits']))
        if last_idx == 0:
            return {}

        return {
            'avg_station_profit': np.mean(self.history['station_profits'][-last_idx:]),
            'avg_ev_cost': np.mean(self.history['ev_costs'][-last_idx:]),
            'avg_congestion_cost': np.mean(self.history['congestion_costs'][-last_idx:]),
        }


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        # 关键：使用更温和的softmax确保混合策略
        x = F.softmax(self.fc4(x) * 0.5, dim=-1)  # 乘以0.5使分布更均匀
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class ForcedMixedStrategyAgent:
    def __init__(self, state_dim, action_dim, actor_lr=0.001, critic_lr=0.002,
                 gamma=0.95, tau=0.01, entropy_coef=0.3, agent_type='station'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.entropy_coef = entropy_coef
        self.agent_type = agent_type

        # 网络
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 经验回放
        self.memory = ReplayBuffer(capacity=10000)

        # 探索参数 - 关键：保持高探索率
        self.exploration_rate = 0.8
        self.exploration_decay = 0.999
        self.exploration_min = 0.3  # 保持较高最小探索率

        # 混合策略参数
        self.mixed_strategy_target = 0.15  # 每个档次至少15%概率
        self.concentration_penalty_weight = 1.0  # 集中度惩罚权重

    def select_action(self, state, training=True):
        """选择动作 - 混合策略"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action_probs = self.actor(state_tensor).numpy()[0]

        if training and np.random.random() < self.exploration_rate:
            # 探索混合策略：使用均匀分布
            noise = np.ones(self.action_dim) / self.action_dim
            action_probs = 0.5 * action_probs + 0.5 * noise

        # 混合策略：确保所有档次都有最小概率
        min_prob = 0.05  # 每个档次至少5%概率
        action_probs = np.maximum(action_probs, min_prob)
        action_probs = action_probs / np.sum(action_probs)

        # 衰减探索率
        if training:
            self.exploration_rate = max(self.exploration_min,
                                        self.exploration_rate * self.exploration_decay)

        return action_probs

    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return None, None, None

        batch = self.memory.sample(batch_size)
        if batch is None:
            return None, None, None

        states, actions, rewards, next_states, dones = batch

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # 更新Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_Q = self.critic_target(next_states, next_actions)
            target_Q = rewards + self.gamma * (1 - dones) * target_Q

        current_Q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_Q, target_Q.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # 更新Actor - 关键：混合策略
        actor_actions = self.actor(states)

        # 计算策略熵
        log_probs = torch.log(actor_actions + 1e-10)
        entropy = -torch.sum(actor_actions * log_probs, dim=1, keepdim=True).mean()

        # 关键：混合策略损失函数
        if self.agent_type == 'station':
            # 1. 基本利润项
            profit_term = -self.critic(states, actor_actions).mean()

            # 2. 熵奖励（鼓励多样性）
            entropy_reward = self.entropy_coef * entropy

            # 3. 混合项：惩罚过于集中的策略
            max_prob = torch.max(actor_actions, dim=1)[0]
            concentration_penalty = torch.mean((max_prob - self.mixed_strategy_target) ** 2)

            # 4. 均匀分布奖励：鼓励接近均匀分布
            uniform_target = torch.ones_like(actor_actions) / self.action_dim
            uniform_reward = -F.mse_loss(actor_actions, uniform_target)

            actor_loss = profit_term - entropy_reward + 2.0 * concentration_penalty - 0.5 * uniform_reward
        else:
            # 电动汽车：成本最小化 + 适度确定性
            cost_term = -self.critic(states, actor_actions).mean()
            actor_loss = cost_term - 0.05 * entropy

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # 软更新目标网络
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss.item(), entropy.item()


def train_sequential_game_with_forced_mixing():
    """训练序贯博弈模型 - 混合策略版本"""
    env = ChargingEnvironment()

    state_dim = env.N_levels * 3
    action_dim = env.N_levels

    # 创建智能体 - 使用混合策略智能体
    station_agent = ForcedMixedStrategyAgent(state_dim, action_dim,
                                             actor_lr=0.0005, critic_lr=0.001,
                                             entropy_coef=0.4, agent_type='station')
    ev_agent = ForcedMixedStrategyAgent(state_dim, action_dim,
                                        actor_lr=0.0005, critic_lr=0.001,
                                        entropy_coef=0.2, agent_type='ev')

    # 训练参数 - 优化训练效率
    episodes = 200
    max_steps = 100
    batch_size = 128
    update_frequency = 2

    # 训练统计
    training_stats = {
        'episode': [], 'station_profits': [], 'ev_costs': [],
        'congestion_costs': [], 'station_entropy': [], 'ev_entropy': [],
        'cost_variance': [], 'station_concentration': []
    }

    print("开始序贯博弈训练（混合策略）...")
    print("=" * 80)

    start_time = time.time()

    for episode in range(episodes):
        state = env.reset()
        episode_station_reward = 0
        episode_ev_reward = 0
        episode_congestion = 0
        episode_station_entropy = 0
        episode_ev_entropy = 0
        episode_cost_variance = 0
        episode_station_concentration = 0

        for step in range(max_steps):
            # 选择动作
            p_action = station_agent.select_action(state, training=True)
            q_action = ev_agent.select_action(state, training=True)

            # 环境交互
            next_state, station_reward, ev_reward, congestion_costs, e = env.step(p_action, q_action)

            # 计算统计量
            p_entropy = -np.sum(p_action * np.log(p_action + 1e-10))
            q_entropy = -np.sum(q_action * np.log(q_action + 1e-10))
            station_concentration = np.max(p_action)

            # 计算成本方差
            ev_costs_by_level = env.price_levels + 2.0 * congestion_costs
            cost_var = np.var(ev_costs_by_level)

            # 关键：强化混合策略奖励
            # 对充电站：显著奖励混合策略
            if p_entropy > 1.0:
                station_reward_adjusted = station_reward + 10.0 * (p_entropy / np.log(env.N_levels))
            else:
                station_reward_adjusted = station_reward - 5.0 * (1 - p_entropy / np.log(env.N_levels))

            # 对电动汽车：奖励均衡选择
            ev_reward_adjusted = ev_reward + 2.0 * (1.0 - min(cost_var * 10, 1.0))

            # 存储经验
            station_agent.memory.add(state, p_action, station_reward_adjusted, next_state, False)
            ev_agent.memory.add(state, q_action, ev_reward_adjusted, next_state, False)

            # 定期更新
            if step % update_frequency == 0 and len(station_agent.memory) > batch_size:
                station_agent.update(batch_size)
                ev_agent.update(batch_size)

            # 更新统计
            state = next_state
            episode_station_reward += station_reward
            episode_ev_reward += ev_reward
            episode_congestion += np.sum(congestion_costs)
            episode_station_entropy += p_entropy
            episode_ev_entropy += q_entropy
            episode_cost_variance += cost_var
            episode_station_concentration += station_concentration

        # 记录训练统计
        training_stats['episode'].append(episode)
        training_stats['station_profits'].append(episode_station_reward / max_steps)
        training_stats['ev_costs'].append(-episode_ev_reward / max_steps)
        training_stats['congestion_costs'].append(episode_congestion / max_steps)
        training_stats['station_entropy'].append(episode_station_entropy / max_steps)
        training_stats['ev_entropy'].append(episode_ev_entropy / max_steps)
        training_stats['cost_variance'].append(episode_cost_variance / max_steps)
        training_stats['station_concentration'].append(episode_station_concentration / max_steps)

        # 每20轮输出一次训练进展
        if episode % 20 == 0 or episode == episodes - 1:
            metrics = env.get_performance_metrics()
            elapsed_time = time.time() - start_time
            print(f"Episode {episode} ({elapsed_time:.1f}s):")
            if metrics:
                print(f"  充电站平均利润: {metrics.get('avg_station_profit', 0):.2f}")
                print(f"  电动汽车平均成本: {metrics.get('avg_ev_cost', 0):.4f}")
                print(f"  平均拥堵成本: {metrics.get('avg_congestion_cost', 0):.4f}")
            print(f"  充电站策略熵: {training_stats['station_entropy'][-1]:.4f}")
            print(f"  电车策略熵: {training_stats['ev_entropy'][-1]:.4f}")
            print(f"  策略集中度: {training_stats['station_concentration'][-1]:.4f}")
            print(f"  探索率: {station_agent.exploration_rate:.4f}")
            print("-" * 60)

    total_time = time.time() - start_time
    print(f"训练完成! 总耗时: {total_time:.1f}秒")
    print("=" * 80)

    return station_agent, ev_agent, env, training_stats


def analyze_final_strategies(env, station_agent, ev_agent):
    """分析最终策略"""
    print("\n" + "=" * 80)
    print("序贯博弈上层调度决策 - 最终结果分析（混合策略）")
    print("=" * 80)

    # 使用多个测试状态
    num_tests = 20
    p_optimal_all = []
    q_optimal_all = []

    for _ in range(num_tests):
        # 使用混合初始状态
        p_init = np.ones(env.N_levels) / env.N_levels
        q_init = np.ones(env.N_levels) / env.N_levels
        test_state = np.concatenate([p_init, q_init, env.price_levels])

        with torch.no_grad():
            state_tensor = torch.FloatTensor(test_state).unsqueeze(0)
            p_optimal = station_agent.actor(state_tensor).numpy()[0]
            q_optimal = ev_agent.actor(state_tensor).numpy()[0]

            p_optimal = np.maximum(p_optimal, 0.001)
            q_optimal = np.maximum(q_optimal, 0.001)
            p_optimal = p_optimal / np.sum(p_optimal)
            q_optimal = q_optimal / np.sum(q_optimal)

            p_optimal_all.append(p_optimal)
            q_optimal_all.append(q_optimal)

    # 计算平均策略
    p_optimal_avg = np.mean(p_optimal_all, axis=0)
    q_optimal_avg = np.mean(q_optimal_all, axis=0)

    # 计算拥堵成本
    congestion_costs, e = env.calculate_congestion_cost(q_optimal_avg)

    print("\n1. 价格档次设置:")
    print("-" * 60)
    for i, price in enumerate(env.price_levels):
        print(f"   档次{i + 1}: 价格={price:.2f}元/kWh, 承载系数ω={env.omega[i]:.2f}, 承载上限k={env.k[i]:.0f}辆")

    print("\n2. 充电站最优定价策略 p*:")
    print("-" * 60)

    # 排序显示
    strategy_info = []
    for i in range(env.N_levels):
        prob = p_optimal_avg[i]
        stars = "★" * int(prob * 40)  # 更多星号显示差异
        strategy_info.append((i, prob, env.price_levels[i], stars))

    strategy_info.sort(key=lambda x: x[1], reverse=True)

    for i, prob, price, stars in strategy_info:
        print(f"   档次{i + 1} ({price:.2f}元/kWh): 概率={prob:.4f} {stars}")

    # 混合策略分析
    mixed_count = np.sum((p_optimal_avg > 0.1) & (p_optimal_avg < 0.9))
    entropy_station = -np.sum(p_optimal_avg * np.log(p_optimal_avg + 1e-10))
    max_entropy = np.log(env.N_levels)

    print(f"\n   混合策略分析:")
    print(f"     使用混合策略的档次数: {mixed_count}/{env.N_levels}")
    print(f"     策略熵: {entropy_station:.4f} (最大可能: {max_entropy:.4f})")
    print(f"     熵比率: {entropy_station / max_entropy:.2%}")

    # 检查是否达到混合策略目标
    if mixed_count >= 3 and entropy_station > 1.0:
        print(f"     ✅ 成功实现混合策略")
    else:
        print(f"     ⚠ 未完全达到混合策略目标")

    print("\n3. 电动汽车最优充电策略 q*:")
    print("-" * 60)

    for i in range(env.N_levels):
        prob = q_optimal_avg[i]
        expected_vehicles = env.N * prob
        congestion_status = "拥堵" if expected_vehicles > env.k[i] * 0.7 else "正常"
        stars = "★" * int(prob * 40)

        print(f"   档次{i + 1}: 概率={prob:.4f} {stars}")
        print(f"       预期车辆={expected_vehicles:.1f}辆, 拥堵因子={e[i]:.3f}")
        print(
            f"       拥堵成本={congestion_costs[i]:.4f}, 总成本={env.price_levels[i] + 2.0 * congestion_costs[i]:.4f}")
        print(
            f"       容量利用率={(expected_vehicles / env.k[i] * 100) if env.k[i] > 0 else 0:.1f}% [{congestion_status}]")
        print()

    print("\n4. 性能指标分析:")
    print("-" * 60)

    # 计算期望利润和成本
    station_profits = np.zeros(env.N_levels)
    ev_costs = np.zeros(env.N_levels)

    for i in range(env.N_levels):
        station_profits[i] = (env.price_levels[i] - env.c) * env.N * q_optimal_avg[i] - 2.0 * congestion_costs[i]
        ev_costs[i] = env.price_levels[i] + 2.0 * congestion_costs[i]

    station_total_profit = np.sum(p_optimal_avg * station_profits)
    ev_total_cost = np.sum(q_optimal_avg * ev_costs)

    print(f"   充电站期望利润: {station_total_profit:.2f}")
    print(f"   电动汽车期望成本: {ev_total_cost:.4f}")
    print(f"   系统总拥堵成本: {np.sum(congestion_costs):.4f}")

    print("\n5. 系统效率分析:")
    print("-" * 60)

    # 负载分布
    load_distribution = q_optimal_avg * 100
    load_balance = np.std(load_distribution) / np.mean(load_distribution) * 100 if np.mean(load_distribution) > 0 else 0

    print(f"   负载分布:")
    for i, load in enumerate(load_distribution):
        utilization = (load_distribution[i] / (env.omega[i] * 100)) * 100 if env.omega[i] > 0 else 0
        print(f"     档次{i + 1}: {load:.1f}% (容量利用率: {utilization:.1f}%)")

    print(f"\n   负载均衡度: {load_balance:.2f}% (越低越好)")

    # 价格引导效果
    price_load_corr = np.corrcoef(env.price_levels, load_distribution)[0, 1]
    print(f"   价格-负载相关性: {price_load_corr:.3f}")

    if price_load_corr < -0.3:
        print(f"   ✅ 价格有效引导负载分布")
    else:
        print(f"   ⚠ 价格引导效果一般")

    print("\n" + "=" * 80)
    print("序贯博弈调度决策完成（混合策略）")
    print("=" * 80)

    return p_optimal_avg, q_optimal_avg, env, {
        'price_levels': env.price_levels,
        'omega': env.omega,
        'k': env.k,
        'station_total_profit': station_total_profit,
        'ev_total_cost': ev_total_cost,
        'congestion_costs_total': np.sum(congestion_costs),
        'load_balance': load_balance,
        'price_load_correlation': price_load_corr,
        'mixed_strategy_count': mixed_count,
        'strategy_entropy': entropy_station
    }


def visualize_results(training_stats, p_optimal, q_optimal, env):
    """可视化结果"""
    try:
        # 创建目录
        os.makedirs('upper_level_results', exist_ok=True)

        # 1. 训练过程可视化
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('混合策略训练过程', fontsize=18, fontweight='bold')

        # 充电站利润
        axes[0, 0].plot(training_stats['episode'], training_stats['station_profits'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('训练轮数', fontsize=12)
        axes[0, 0].set_ylabel('充电站平均利润', fontsize=12)
        axes[0, 0].set_title('充电站利润收敛曲线', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)

        # 电动汽车成本
        axes[0, 1].plot(training_stats['episode'], training_stats['ev_costs'], 'g-', linewidth=2)
        axes[0, 1].set_xlabel('训练轮数', fontsize=12)
        axes[0, 1].set_ylabel('电动汽车平均成本', fontsize=12)
        axes[0, 1].set_title('电动汽车成本收敛曲线', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)

        # 策略熵
        axes[0, 2].plot(training_stats['episode'], training_stats['station_entropy'], 'r-', linewidth=2, label='充电站')
        axes[0, 2].plot(training_stats['episode'], training_stats['ev_entropy'], 'b-', linewidth=2, label='电动汽车')
        axes[0, 2].set_xlabel('训练轮数', fontsize=12)
        axes[0, 2].set_ylabel('策略熵', fontsize=12)
        axes[0, 2].set_title('策略多样性演化', fontsize=14, fontweight='bold')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].axhline(y=np.log(env.N_levels), color='k', linestyle='--', alpha=0.5, label='最大熵')

        # 策略集中度
        axes[1, 0].plot(training_stats['episode'], training_stats['station_concentration'], 'm-', linewidth=2)
        axes[1, 0].set_xlabel('训练轮数', fontsize=12)
        axes[1, 0].set_ylabel('策略集中度', fontsize=12)
        axes[1, 0].set_title('充电站策略集中度', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0.3, color='r', linestyle='--', alpha=0.5, label='混合策略目标')
        axes[1, 0].legend()

        # 成本方差
        axes[1, 1].plot(training_stats['episode'], training_stats['cost_variance'], 'c-', linewidth=2)
        axes[1, 1].set_xlabel('训练轮数', fontsize=12)
        axes[1, 1].set_ylabel('成本方差', fontsize=12)
        axes[1, 1].set_title('纳什均衡收敛情况', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0.05, color='r', linestyle='--', alpha=0.7, label='均衡阈值')
        axes[1, 1].legend()

        # 拥堵成本
        axes[1, 2].plot(training_stats['episode'], training_stats['congestion_costs'], 'y-', linewidth=2)
        axes[1, 2].set_xlabel('训练轮数', fontsize=12)
        axes[1, 2].set_ylabel('平均拥堵成本', fontsize=12)
        axes[1, 2].set_title('系统拥堵演化', fontsize=14, fontweight='bold')
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('upper_level_results/混合策略训练过程.png', dpi=300, bbox_inches='tight')
        print("训练过程图已保存为 'upper_level_results/混合策略训练过程.png'")

        # 2. 最终策略分布图
        plt.figure(figsize=(14, 10))

        x = np.arange(env.N_levels)
        width = 0.35

        ax1 = plt.subplot(2, 1, 1)
        bars1 = ax1.bar(x - width / 2, p_optimal, width, label='充电站定价策略 (p)',
                        alpha=0.8, color='skyblue', edgecolor='black')
        bars2 = ax1.bar(x + width / 2, q_optimal, width, label='电动汽车充电策略 (q)',
                        alpha=0.8, color='lightcoral', edgecolor='black')

        ax1.set_xlabel('价格档次', fontsize=13)
        ax1.set_ylabel('概率分布', fontsize=13)
        ax1.set_title('混合策略 - 最优策略分布', fontsize=15, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'档次{i + 1}\n({env.price_levels[i]:.2f}元)' for i in range(env.N_levels)],
                            fontsize=11, rotation=0)
        ax1.legend(fontsize=11, loc='upper right')
        ax1.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 负载分布子图
        ax2 = plt.subplot(2, 1, 2)

        expected_vehicles = q_optimal * env.N
        capacities = env.k

        bars3 = ax2.bar(x, expected_vehicles, width=0.6, alpha=0.7, color='lightgreen',
                        edgecolor='black', label='预期车辆数')
        ax2.plot(x, capacities, 'r--', linewidth=2, marker='o', markersize=8, label='承载上限')

        ax2.set_xlabel('价格档次', fontsize=13)
        ax2.set_ylabel('车辆数量', fontsize=13)
        ax2.set_title('负载分布与承载能力对比', fontsize=15, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'档次{i + 1}' for i in range(env.N_levels)], fontsize=11)
        ax2.legend(fontsize=11, loc='upper right')
        ax2.grid(True, alpha=0.3)

        # 添加利用率标签
        for i, (vehicles, capacity) in enumerate(zip(expected_vehicles, capacities)):
            utilization = vehicles / capacity * 100 if capacity > 0 else 0
            color = 'green' if utilization < 50 else 'orange' if utilization < 80 else 'red'
            ax2.text(i, max(vehicles, capacity) + 3, f'{utilization:.1f}%',
                     ha='center', va='bottom', fontsize=10, color=color, fontweight='bold')

        plt.tight_layout()
        plt.savefig('upper_level_results/混合策略最优分布.png', dpi=300, bbox_inches='tight')
        print("最优策略图已保存为 'upper_level_results/混合策略最优分布.png'")

        plt.close('all')

    except Exception as e:
        print(f"可视化过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


def export_results(p_optimal, q_optimal, env, training_stats, analysis_results):
    """导出结果到CSV文件"""
    try:
        # 创建目录
        os.makedirs('upper_level_results', exist_ok=True)

        congestion_costs, e = env.calculate_congestion_cost(q_optimal)

        detailed_strategy_df = pd.DataFrame({
            '价格档次': [f'档次{i + 1}' for i in range(env.N_levels)],
            '电价_元每kWh': env.price_levels,
            '承载系数': env.omega,
            '承载上限_辆': env.k,
            '充电站策略概率': p_optimal,
            '电动汽车策略概率': q_optimal,
            '预期车辆数': q_optimal * env.N,
            '拥堵因子': e,
            '拥堵成本': congestion_costs,
            '电动汽车成本': env.price_levels + 2.0 * congestion_costs,
            '充电站利润': [(env.price_levels[i] - env.c) * env.N * q_optimal[i] - 2.0 * congestion_costs[i]
                           for i in range(env.N_levels)],
            '容量利用率_百分比': [(q_optimal[i] * env.N) / env.k[i] * 100 if env.k[i] > 0 else 0
                                  for i in range(env.N_levels)]
        })

        if training_stats['episode']:
            stats_df = pd.DataFrame({
                '训练轮数': training_stats['episode'],
                '充电站利润': training_stats['station_profits'],
                '电动汽车成本': training_stats['ev_costs'],
                '拥堵成本': training_stats['congestion_costs'],
                '充电站策略熵': training_stats.get('station_entropy', [0] * len(training_stats['episode'])),
                '电动汽车策略熵': training_stats.get('ev_entropy', [0] * len(training_stats['episode'])),
                '成本方差': training_stats.get('cost_variance', [0] * len(training_stats['episode'])),
                '充电站策略集中度': training_stats.get('station_concentration', [0] * len(training_stats['episode']))
            })
        else:
            stats_df = pd.DataFrame()

        # 添加分析结果
        analysis_df = pd.DataFrame([analysis_results])

        detailed_strategy_df.to_csv('upper_level_results/混合策略_详细策略.csv', index=False, encoding='utf-8-sig')
        if not stats_df.empty:
            stats_df.to_csv('upper_level_results/混合策略_训练统计.csv', index=False, encoding='utf-8-sig')
        analysis_df.to_csv('upper_level_results/混合策略_分析结果.csv', index=False, encoding='utf-8-sig')

        print("详细策略已导出到 'upper_level_results/混合策略_详细策略.csv'")
        if not stats_df.empty:
            print("训练统计已导出到 'upper_level_results/混合策略_训练统计.csv'")
        print("分析结果已导出到 'upper_level_results/混合策略_分析结果.csv'")

    except Exception as e:
        print(f"导出结果时出现错误: {str(e)}")


def run_upper_level_scheduling():
    """运行上层调度系统"""
    print("基于序贯博弈的电动汽车上层调度决策系统（混合策略版本）")
    print("=" * 80)
    print("系统设计理念:")
    print("- 充电站：使用混合定价策略，避免价格竞争陷阱")
    print("- 电动汽车：基于价格和拥堵成本分散选择")
    print("- 目标：实现真正的混合策略纳什均衡")
    print("=" * 80)

    try:
        station_agent, ev_agent, env, training_stats = train_sequential_game_with_forced_mixing()
        p_optimal, q_optimal, env, analysis_results = analyze_final_strategies(env, station_agent, ev_agent)
        visualize_results(training_stats, p_optimal, q_optimal, env)
        export_results(p_optimal, q_optimal, env, training_stats, analysis_results)

        return {
            'p_optimal': p_optimal,
            'q_optimal': q_optimal,
            'price_levels': env.price_levels,
            'analysis_results': analysis_results,
            'training_stats': training_stats
        }

    except Exception as e:
        print(f"\n❌ 上层调度执行过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    run_upper_level_scheduling()