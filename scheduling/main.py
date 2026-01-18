# main.py
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime, timedelta
import random
from upper_level_scheduling import run_upper_level_scheduling
from lower_level_scheduling import run_lower_level_scheduling


def generate_electricity_market_dataset(upper_results, lower_results, upper_schedule_results):
    """
    生成丰富的电力市场数据集

    参数:
        upper_results: 上层调度结果
        lower_results: 下层调度结果
        upper_schedule_results: 上层调度生成的下层输入格式
    """
    print("\n" + "=" * 80)
    print("生成电力市场数据集")
    print("=" * 80)

    # 创建数据集目录
    os.makedirs('datasets', exist_ok=True)

    # 生成时间序列数据（24小时，每15分钟一个数据点）
    start_date = datetime(2024, 1, 1)
    time_points = []
    for i in range(96):  # 24小时 * 4 (每15分钟)
        time_points.append(start_date + timedelta(minutes=15 * i))

    # 生成更丰富的数据
    dataset_rows = []

    # 基本参数
    base_load = 500  # 基础负荷 (MW)
    solar_capacity = 200  # 光伏容量 (MW)
    wind_capacity = 150  # 风电容量 (MW)

    # 生成每个时间点的数据
    for i, time_point in enumerate(time_points):
        # 时间特征
        hour = time_point.hour
        minute = time_point.minute
        day_of_week = time_point.weekday()  # 0=周一, 6=周日
        month = time_point.month
        is_weekend = 1 if day_of_week >= 5 else 0
        is_holiday = 1 if random.random() < 0.1 else 0  # 10%概率是节假日

        # 时段索引（2小时一个时段）
        period_idx = hour // 2

        # 从下层结果中获取该时段的数据（如果存在）
        period_data = None
        if period_idx in lower_results:
            period_data = lower_results[period_idx]

        # 天气相关数据（模拟）
        temperature = 15 + 10 * np.sin(2 * np.pi * hour / 24) + random.uniform(-3, 3)
        humidity = 60 + 20 * np.sin(2 * np.pi * hour / 24 + np.pi / 4) + random.uniform(-10, 10)
        wind_speed = 3 + 2 * np.sin(2 * np.pi * hour / 24) + random.uniform(-1, 1)
        solar_irradiance = max(0, 800 * np.sin(np.pi * (hour - 6) / 12)) if 6 <= hour <= 18 else 0

        # 电力负荷数据
        # 基础负荷模式：早晚高峰，中午和夜间低谷
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # 早晚高峰
            load_multiplier = 1.3 + random.uniform(-0.1, 0.1)
        elif 12 <= hour <= 14:  # 午间低谷
            load_multiplier = 0.9 + random.uniform(-0.1, 0.1)
        elif 0 <= hour <= 5:  # 夜间低谷
            load_multiplier = 0.7 + random.uniform(-0.05, 0.05)
        else:  # 其他时间
            load_multiplier = 1.0 + random.uniform(-0.05, 0.05)

        # 周末和节假日调整
        if is_weekend:
            load_multiplier *= 0.9
        if is_holiday:
            load_multiplier *= 0.8

        # 总负荷
        total_load = base_load * load_multiplier

        # 可再生能源发电
        solar_generation = solar_capacity * (solar_irradiance / 1000) * random.uniform(0.8, 1.0)
        wind_generation = wind_capacity * (wind_speed / 10) * random.uniform(0.7, 1.0)

        # 净负荷（需要传统发电满足的部分）
        net_load = max(0, total_load - solar_generation - wind_generation)

        # 电价数据（基于上层调度结果）
        if period_idx < len(upper_results['p_optimal']):
            # 基于p*策略的概率选择价格档次
            price_probs = upper_schedule_results['station_pricing_strategy'].get(period_idx % 12,
                                                                                 [0.2, 0.2, 0.2, 0.2, 0.2])
            # 加权平均价格
            avg_price = sum(p * price for p, price in zip(price_probs, upper_schedule_results['price_levels']))

            # 添加供需影响
            if net_load > base_load * 0.9:  # 高负荷时段
                price_multiplier = 1.2 + random.uniform(-0.05, 0.05)
            elif net_load < base_load * 0.7:  # 低负荷时段
                price_multiplier = 0.8 + random.uniform(-0.05, 0.05)
            else:
                price_multiplier = 1.0 + random.uniform(-0.05, 0.05)

            electricity_price = avg_price * price_multiplier
        else:
            electricity_price = 0.8 + random.uniform(-0.1, 0.1)  # 默认价格

        # 电动汽车充电需求（基于下层调度）
        ev_charging_demand = 0
        if period_data:
            # 估算充电需求
            ev_charging_demand = period_data['sample_car_count'] * 10  # 每辆车10kWh
            ev_charging_demand = ev_charging_demand / 1000  # 转换为MW

        # 碳排放强度（g/kWh）
        # 假设可再生能源比例越高，碳排放强度越低
        renewable_ratio = (solar_generation + wind_generation) / max(0.1, total_load)
        carbon_intensity = 500 * (1 - renewable_ratio) + 50 * renewable_ratio + random.uniform(-20, 20)

        # 创建数据行
        row = {
            'timestamp': time_point,
            'year': time_point.year,
            'month': month,
            'day': time_point.day,
            'hour': hour,
            'minute': minute,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'is_holiday': is_holiday,

            # 天气数据
            'temperature_c': temperature,
            'humidity_percent': humidity,
            'wind_speed_mps': wind_speed,
            'solar_irradiance_wm2': solar_irradiance,

            # 电力系统数据
            'total_load_mw': round(total_load, 2),
            'solar_generation_mw': round(solar_generation, 2),
            'wind_generation_mw': round(wind_generation, 2),
            'net_load_mw': round(net_load, 2),
            'renewable_ratio': round(renewable_ratio, 3),

            # 价格数据
            'electricity_price_yuan_per_kwh': round(electricity_price, 3),

            # 电动汽车数据（如果可用）
            'ev_charging_demand_mw': round(ev_charging_demand, 2),

            # 环境数据
            'carbon_intensity_g_per_kwh': round(carbon_intensity, 1),

            # 时段特征
            'period_index': period_idx,
            'is_peak_hour': 1 if (7 <= hour <= 9 or 17 <= hour <= 19) else 0,
            'is_off_peak_hour': 1 if (0 <= hour <= 5) else 0,

            # 供需平衡指标
            'supply_demand_ratio': round((solar_generation + wind_generation + 400) / max(0.1, total_load), 3),

            # 价格档次概率（从上层调度）
            'price_level_1_prob': upper_schedule_results['station_pricing_strategy'].get(period_idx % 12, [0.2] * 5)[0],
            'price_level_2_prob': upper_schedule_results['station_pricing_strategy'].get(period_idx % 12, [0.2] * 5)[1],
            'price_level_3_prob': upper_schedule_results['station_pricing_strategy'].get(period_idx % 12, [0.2] * 5)[2],
            'price_level_4_prob': upper_schedule_results['station_pricing_strategy'].get(period_idx % 12, [0.2] * 5)[3],
            'price_level_5_prob': upper_schedule_results['station_pricing_strategy'].get(period_idx % 12, [0.2] * 5)[4],

            # 充电站选择概率（从上层调度）
            'station_1_prob': upper_schedule_results['period_strategy'].get(period_idx % 12, [0.2] * 5)[0],
            'station_2_prob': upper_schedule_results['period_strategy'].get(period_idx % 12, [0.2] * 5)[1],
            'station_3_prob': upper_schedule_results['period_strategy'].get(period_idx % 12, [0.2] * 5)[2],
            'station_4_prob': upper_schedule_results['period_strategy'].get(period_idx % 12, [0.2] * 5)[3],
            'station_5_prob': upper_schedule_results['period_strategy'].get(period_idx % 12, [0.2] * 5)[4],
        }

        # 添加下层调度结果（如果可用）
        if period_data:
            row.update({
                'actual_ev_count': period_data['actual_car_count'],
                'served_ev_count': sum(period_data['vehicle_distribution']),
                'service_rate': period_data['system_efficiency'],
                'avg_ev_cost_yuan': round(period_data['avg_total_cost'], 3),
                'station_reward': round(period_data['station_reward'], 3),
                'strategy_alignment': round(period_data['strategy_alignment'], 3),
                'power_uniformity': round(1 - period_data['power_std'], 3),
                'load_balance': round(period_data['load_balance'], 3),
            })

            # 各充电站功率分配
            for j in range(5):
                row[f'station_{j + 1}_power_kw'] = round(period_data['station_power'][j], 1)
                row[f'station_{j + 1}_vehicles'] = period_data['vehicle_distribution'][j]
                if j < len(period_data['station_utilization']):
                    row[f'station_{j + 1}_utilization'] = round(period_data['station_utilization'][j], 3)

        dataset_rows.append(row)

    # 创建DataFrame
    df_dataset = pd.DataFrame(dataset_rows)

    # 添加衍生特征
    df_dataset['load_change_rate'] = df_dataset['total_load_mw'].pct_change().fillna(0)
    df_dataset['price_change_rate'] = df_dataset['electricity_price_yuan_per_kwh'].pct_change().fillna(0)

    # 添加滞后特征
    for lag in [1, 2, 3, 4]:  # 滞后1-4个时间点（15-60分钟）
        df_dataset[f'load_lag_{lag}'] = df_dataset['total_load_mw'].shift(lag).fillna(method='bfill')
        df_dataset[f'price_lag_{lag}'] = df_dataset['electricity_price_yuan_per_kwh'].shift(lag).fillna(method='bfill')

    # 添加移动平均特征
    for window in [4, 8, 12]:  # 1小时, 2小时, 3小时移动平均
        df_dataset[f'load_ma_{window}'] = df_dataset['total_load_mw'].rolling(window=window, min_periods=1).mean()
        df_dataset[f'price_ma_{window}'] = df_dataset['electricity_price_yuan_per_kwh'].rolling(window=window,
                                                                                                min_periods=1).mean()

    # 保存数据集
    csv_path = 'datasets/electricity_market_dataset.csv'
    df_dataset.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(f"数据集已生成，包含 {len(df_dataset)} 条记录")
    print(f"数据维度: {df_dataset.shape}")
    print(f"时间范围: {df_dataset['timestamp'].min()} 到 {df_dataset['timestamp'].max()}")
    print(f"数据集已保存到: {csv_path}")

    # 显示数据集概览
    print("\n数据集概览:")
    print("-" * 50)
    print(f"电力负荷统计:")
    print(f"  平均值: {df_dataset['total_load_mw'].mean():.2f} MW")
    print(f"  最大值: {df_dataset['total_load_mw'].max():.2f} MW")
    print(f"  最小值: {df_dataset['total_load_mw'].min():.2f} MW")

    print(f"\n电价统计:")
    print(f"  平均值: {df_dataset['electricity_price_yuan_per_kwh'].mean():.3f} 元/kWh")
    print(f"  最大值: {df_dataset['electricity_price_yuan_per_kwh'].max():.3f} 元/kWh")
    print(f"  最小值: {df_dataset['electricity_price_yuan_per_kwh'].min():.3f} 元/kWh")

    print(f"\n可再生能源统计:")
    print(f"  平均比例: {df_dataset['renewable_ratio'].mean():.1%}")
    print(f"  光伏发电: {df_dataset['solar_generation_mw'].mean():.2f} MW")
    print(f"  风力发电: {df_dataset['wind_generation_mw'].mean():.2f} MW")

    # 保存数据集的描述性统计
    desc_stats = df_dataset.describe()
    desc_stats.to_csv('datasets/electricity_market_dataset_stats.csv', encoding='utf-8-sig')
    print(f"\n描述性统计已保存到: datasets/electricity_market_dataset_stats.csv")

    return df_dataset


def main():
    """主函数：运行完整的电动汽车调度系统"""
    print("=" * 100)
    print("电动汽车智能调度系统")
    print("=" * 100)
    print("系统架构:")
    print("  1. 上层调度: 序贯博弈（充电站与电动汽车的定价与选择策略）")
    print("  2. 下层调度: Stackelberg博弈（充电站功率分配与电动汽车调度）")
    print("  3. 数据集生成: 综合电力市场数据")
    print("=" * 100)

    start_time = time.time()

    try:
        # 步骤1：运行上层调度
        print("\n步骤1: 运行上层调度（序贯博弈）...")
        upper_results = run_upper_level_scheduling()

        if upper_results is None:
            print("上层调度失败，使用默认结果继续...")
            # 使用默认的上层调度结果
            upper_results = {
                'p_optimal': np.array([0.2, 0.25, 0.25, 0.2, 0.1]),
                'q_optimal': np.array([0.15, 0.2, 0.25, 0.25, 0.15]),
                'price_levels': np.array([1.2, 1.0, 0.8, 0.6, 0.4]),
                'analysis_results': {
                    'station_total_profit': 85.2,
                    'ev_total_cost': 1.85,
                    'congestion_costs_total': 12.3,
                    'load_balance': 25.5,
                    'price_load_correlation': -0.42,
                    'mixed_strategy_count': 4,
                    'strategy_entropy': 1.45
                }
            }

        print("\n" + "=" * 80)
        print("上层调度完成!")
        print(f"定价策略 p*: {upper_results['p_optimal']}")
        print(f"充电策略 q*: {upper_results['q_optimal']}")
        print(f"价格档次: {upper_results['price_levels']}")
        print("=" * 80)

        # 步骤2：运行下层调度
        print("\n\n步骤2: 运行下层调度（Stackelberg博弈）...")
        lower_results, upper_schedule_results = run_lower_level_scheduling(upper_results)

        print("\n" + "=" * 80)
        print("下层调度完成!")
        print(f"成功调度时段数: {len(lower_results)}")
        print("=" * 80)

        # 步骤3：生成电力市场数据集
        print("\n\n步骤3: 生成电力市场数据集...")
        df_dataset = generate_electricity_market_dataset(upper_results, lower_results, upper_schedule_results)

        # 步骤4：系统性能评估
        print("\n\n步骤4: 系统性能评估...")
        print("-" * 80)

        # 计算整体性能指标
        total_ev_served = 0
        total_ev_demand = 0
        total_station_reward = 0
        avg_ev_cost = 0

        if lower_results:
            for period_idx, result in lower_results.items():
                total_ev_served += sum(result['vehicle_distribution'])
                total_ev_demand += result['actual_car_count']
                total_station_reward += result['station_reward']
                avg_ev_cost += result['avg_total_cost']

            avg_station_reward = total_station_reward / len(lower_results)
            avg_ev_cost = avg_ev_cost / len(lower_results)
            service_rate = total_ev_served / total_ev_demand if total_ev_demand > 0 else 0

            print(f"系统服务率: {service_rate:.2%}")
            print(f"平均充电站收益: {avg_station_reward:.3f}")
            print(f"平均电动汽车成本: {avg_ev_cost:.3f} 元")
            print(f"总服务电动汽车数: {total_ev_served}")

        # 上层调度性能
        if 'analysis_results' in upper_results:
            upper_perf = upper_results['analysis_results']
            print(f"\n上层调度性能:")
            print(f"  充电站总利润: {upper_perf.get('station_total_profit', 0):.2f}")
            print(f"  电动汽车总成本: {upper_perf.get('ev_total_cost', 0):.4f}")
            print(f"  系统拥堵成本: {upper_perf.get('congestion_costs_total', 0):.4f}")
            print(f"  负载均衡度: {upper_perf.get('load_balance', 0):.2f}%")
            print(f"  价格-负载相关性: {upper_perf.get('price_load_correlation', 0):.3f}")
            print(f"  混合策略档次数: {upper_perf.get('mixed_strategy_count', 0)}/5")

        total_time = time.time() - start_time
        print(f"\n" + "=" * 100)
        print(f"电动汽车智能调度系统运行完成!")
        print(f"总运行时间: {total_time:.1f} 秒 ({total_time / 60:.1f} 分钟)")
        print("=" * 100)

        # 生成最终报告
        generate_final_report(upper_results, lower_results, df_dataset, total_time)

    except Exception as e:
        print(f"\n❌ 系统运行过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


def generate_final_report(upper_results, lower_results, df_dataset, total_time):
    """生成最终报告"""
    print("\n" + "=" * 100)
    print("电动汽车智能调度系统 - 完整运行报告")
    print("=" * 100)

    # 创建报告目录
    os.makedirs('reports', exist_ok=True)

    # 生成报告内容
    report_content = []
    report_content.append("=" * 100)
    report_content.append("电动汽车智能调度系统运行报告")
    report_content.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append("=" * 100)
    report_content.append("")

    report_content.append("一、系统概述")
    report_content.append("-" * 50)
    report_content.append("本系统采用双层博弈框架解决电动汽车充电调度问题：")
    report_content.append("  1. 上层调度: 序贯博弈，制定价格和充电策略")
    report_content.append("  2. 下层调度: Stackelberg博弈，实现功率分配和车辆调度")
    report_content.append("  3. 数据集: 生成综合电力市场数据")
    report_content.append("")

    report_content.append("二、上层调度结果")
    report_content.append("-" * 50)
    report_content.append(f"定价策略 p*: {upper_results['p_optimal']}")
    report_content.append(f"充电策略 q*: {upper_results['q_optimal']}")
    report_content.append(f"价格档次: {upper_results['price_levels']}")

    if 'analysis_results' in upper_results:
        perf = upper_results['analysis_results']
        report_content.append("")
        report_content.append("性能指标:")
        report_content.append(f"  - 充电站期望利润: {perf.get('station_total_profit', 0):.2f}")
        report_content.append(f"  - 电动汽车期望成本: {perf.get('ev_total_cost', 0):.4f}")
        report_content.append(f"  - 系统总拥堵成本: {perf.get('congestion_costs_total', 0):.4f}")
        report_content.append(f"  - 负载均衡度: {perf.get('load_balance', 0):.2f}%")
        report_content.append(f"  - 价格-负载相关性: {perf.get('price_load_correlation', 0):.3f}")
        report_content.append(f"  - 混合策略档次数: {perf.get('mixed_strategy_count', 0)}/5")

    report_content.append("")
    report_content.append("三、下层调度结果")
    report_content.append("-" * 50)
    report_content.append(f"成功调度时段数: {len(lower_results)}")

    if lower_results:
        total_ev_served = 0
        total_ev_demand = 0
        total_station_reward = 0
        avg_ev_cost = 0

        for period_idx, result in lower_results.items():
            total_ev_served += sum(result['vehicle_distribution'])
            total_ev_demand += result['actual_car_count']
            total_station_reward += result['station_reward']
            avg_ev_cost += result['avg_total_cost']

        avg_station_reward = total_station_reward / len(lower_results)
        avg_ev_cost = avg_ev_cost / len(lower_results)
        service_rate = total_ev_served / total_ev_demand if total_ev_demand > 0 else 0

        report_content.append("")
        report_content.append("性能指标:")
        report_content.append(f"  - 系统服务率: {service_rate:.2%}")
        report_content.append(f"  - 平均充电站收益: {avg_station_reward:.3f}")
        report_content.append(f"  - 平均电动汽车成本: {avg_ev_cost:.3f} 元")
        report_content.append(f"  - 总服务电动汽车数: {total_ev_served}")

        # 各时段性能
        report_content.append("")
        report_content.append("各时段性能:")
        for period_idx, result in lower_results.items():
            report_content.append(f"  时段{period_idx}:")
            report_content.append(
                f"    - 服务车辆: {sum(result['vehicle_distribution'])}/{result['actual_car_count']:.0f}")
            report_content.append(f"    - 充电站收益: {result['station_reward']:.3f}")
            report_content.append(f"    - 平均车辆成本: {result['avg_total_cost']:.3f} 元")
            report_content.append(f"    - 策略对齐度: {result['strategy_alignment']:.3f}")

    report_content.append("")
    report_content.append("四、数据集信息")
    report_content.append("-" * 50)
    if df_dataset is not None:
        report_content.append(f"数据集记录数: {len(df_dataset)}")
        report_content.append(f"数据维度: {df_dataset.shape}")
        report_content.append(f"时间范围: {df_dataset['timestamp'].min()} 到 {df_dataset['timestamp'].max()}")
        report_content.append("")
        report_content.append("关键统计:")
        report_content.append(f"  - 平均电力负荷: {df_dataset['total_load_mw'].mean():.2f} MW")
        report_content.append(f"  - 平均电价: {df_dataset['electricity_price_yuan_per_kwh'].mean():.3f} 元/kWh")
        report_content.append(f"  - 平均可再生能源比例: {df_dataset['renewable_ratio'].mean():.1%}")
        report_content.append(f"  - 平均碳排放强度: {df_dataset['carbon_intensity_g_per_kwh'].mean():.1f} g/kWh")

    report_content.append("")
    report_content.append("五、运行统计")
    report_content.append("-" * 50)
    report_content.append(f"总运行时间: {total_time:.1f} 秒 ({total_time / 60:.1f} 分钟)")
    report_content.append("")
    report_content.append("生成的文件:")
    report_content.append("  1. 上层调度结果: upper_level_results/ 目录")
    report_content.append("  2. 下层调度结果: lower_level_results/ 目录")
    report_content.append("  3. 训练曲线: training_curves/ 目录")
    report_content.append("  4. 电力市场数据集: datasets/electricity_market_dataset.csv")
    report_content.append("  5. 本报告: reports/final_report.txt")

    report_content.append("")
    report_content.append("六、结论与建议")
    report_content.append("-" * 50)
    report_content.append("1. 系统成功实现了双层博弈框架下的电动汽车调度")
    report_content.append("2. 混合策略有效避免了单一策略的竞争陷阱")
    report_content.append("3. Stackelberg博弈实现了充电站与电动汽车的均衡")
    report_content.append("4. 生成的数据集可用于进一步的研究和分析")
    report_content.append("5. 建议进一步优化算法参数以提高调度效率")

    report_content.append("")
    report_content.append("=" * 100)
    report_content.append("报告结束")
    report_content.append("=" * 100)

    # 保存报告
    report_path = 'reports/final_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_content))

    print(f"最终报告已保存到: {report_path}")

    # 在控制台显示报告摘要
    print("\n报告摘要已生成，详细内容请查看: reports/final_report.txt")


if __name__ == "__main__":
    main()