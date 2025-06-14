import pandas as pd
import numpy as np
from pulp import *
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False\
# 加载点位坐标
points = pd.DataFrame({
    '地点': ['东门', '南门', '北门', '一食堂', '二食堂', '三食堂', '梅苑1栋', '菊苑1栋', '教学2楼', '教学4楼', '计算机学院', '工程中心', '网球场', '体育馆', '校医院'],
    'x': [1464, 1082, 954, 634, 428, 864, 606, 452, 924, 1106, 970, 1324, 704, 442, 402],
    'y': [-1128, -2230, -360, -1362, -1652, -1816, -1954, -988, -1276, -1580, -794, -712, -920, -598, -1898]
}).set_index('地点')

# 距离矩阵
dist_matrix = pd.DataFrame({
    '东门': [0.00, 1166.42, 921.91, 833.12, 1059.37, 808.05, 1009.84, 1017.93, 548.80, 468.38, 515.33, 418.07, 761.66, 1087.77, 1089.88],
    '南门': [1166.42, 0.00, 1878.84, 873.85, 654.56, 614.37, 373.87, 1260.43, 1018.37, 659.30, 1468.76, 1540.64, 1331.73, 1635.62, 484.57],
    '北门': [921.91, 1878.84, 0.00, 1008.07, 1320.03, 1457.38, 1597.09, 628.35, 916.22, 1222.85, 436.67, 710.98, 454.15, 512.08, 1565.39],
    '一食堂': [833.12, 873.85, 1008.07, 0.00, 342.58, 452.98, 592.24, 356.58, 292.78, 497.00, 581.78, 892.09, 442.58, 770.10, 584.72],
    '二食堂': [1059.37, 654.56, 1320.03, 342.58, 0.00, 436.74, 302.09, 564.10, 614.94, 678.29, 896.14, 1095.24, 692.44, 912.71, 246.04],
    '三食堂': [808.05, 614.37, 1457.38, 452.98, 436.74, 0.00, 356.58, 682.62, 406.52, 252.00, 747.21, 897.00, 664.09, 1041.28, 336.24],
    '梅苑1栋': [1009.84, 373.87, 1597.09, 592.24, 302.09, 356.58, 0.00, 702.03, 688.27, 451.22, 1041.12, 1171.16, 803.42, 1118.40, 144.22],
    '菊苑1栋': [1017.93, 1260.43, 628.35, 356.58, 564.10, 682.62, 702.03, 0.00, 472.74, 632.58, 410.25, 873.29, 268.33, 390.00, 704.72],
    '教学2楼': [548.80, 1018.37, 916.22, 292.78, 614.94, 406.52, 688.27, 472.74, 0.00, 310.00, 362.25, 601.82, 328.61, 706.37, 702.29],
    '教学4楼': [468.38, 659.30, 1222.85, 497.00, 678.29, 252.00, 451.22, 632.58, 310.00, 0.00, 536.45, 548.36, 518.07, 878.22, 475.09],
    '计算机学院': [515.33, 1468.76, 436.67, 581.78, 896.14, 747.21, 1041.12, 410.25, 362.25, 536.45, 0.00, 447.21, 286.25, 552.21, 1030.29],
    '工程中心': [418.07, 1540.64, 710.98, 892.09, 1095.24, 897.00, 1171.16, 873.29, 601.82, 548.36, 447.21, 0.00, 620.40, 884.22, 1185.37],
    '网球场': [761.66, 1331.73, 454.15, 442.58, 692.44, 664.09, 803.42, 268.33, 328.61, 518.07, 286.25, 620.40, 0.00, 436.31, 808.83],
    '体育馆': [1087.77, 1635.62, 512.08, 770.10, 912.71, 1041.28, 1118.40, 390.00, 706.37, 878.22, 552.21, 884.22, 436.31, 0.00, 1138.81],
    '校医院': [1089.88, 484.57, 1565.39, 584.72, 246.04, 336.24, 144.22, 704.72, 702.29, 475.09, 1030.29, 1185.37, 808.83, 1138.81, 0.00]
}, index=['东门', '南门', '北门', '一食堂', '二食堂', '三食堂', '梅苑1栋', '菊苑1栋', '教学2楼', '教学4楼', '计算机学院', '工程中心', '网球场', '体育馆', '校医院'])

# 数据预处理
P = points.index.tolist()
T = list(range(6))
dist = dist_matrix.loc[P, P].values / 1000
c = dist * 3600 / 25

# 使用提供的 Delta_N_user
provided_delta = {
    ('东门', 0): 54.00, ('南门', 0): 80.00, ('北门', 0): 64.00, ('一食堂', 0): -73.00,
    ('二食堂', 0): -80.00, ('三食堂', 0): -80.00, ('梅苑1栋', 0): -80.00, ('菊苑1栋', 0): -80.00,
    ('教学2楼', 0): 80.00, ('教学4楼', 0): 80.00, ('计算机学院', 0): 53.00, ('工程中心', 0): 59.00,
    ('网球场', 0): -80.00, ('体育馆', 0): -80.00, ('校医院', 0): -80.00,
    ('东门', 1): -80.00, ('南门', 1): -71.00, ('北门', 1): -80.00, ('一食堂', 1): 80.00,
    ('二食堂', 1): 80.00, ('三食堂', 1): 80.00, ('梅苑1栋', 1): 80.00, ('菊苑1栋', 1): 73.00,
    ('教学2楼', 1): -80.00, ('教学4楼', 1): -80.00, ('计算机学院', 1): -80.00, ('工程中心', 1): -80.00,
    ('南门', 2): -80.00, ('一食堂', 2): -80.00, ('二食堂', 2): -80.00, ('三食堂', 2): -80.00,
    ('梅苑1栋', 2): -69.00, ('计算机学院', 2): 74.00, ('工程中心', 2): 66.00,
    ('南门', 3): 80.00, ('二食堂', 3): 74.00, ('三食堂', 3): 61.00, ('梅苑1栋', 3): 55.00,
    ('菊苑1栋', 3): 80.00, ('教学2楼', 3): -80.00, ('教学4楼', 3): -80.00,
    ('东门', 4): 67.00, ('南门', 4): -69.00, ('梅苑1栋', 4): -67.00, ('菊苑1栋', 4): -80.00,
    ('教学2楼', 4): 74.00, ('教学4楼', 4): 55.00,
    ('东门', 5): -80.00, ('一食堂', 5): 60.00, ('三食堂', 5): 69.00, ('梅苑1栋', 5): 80.00,
    ('菊苑1栋', 5): 58.00, ('教学2楼', 5): -79.00, ('教学4楼', 5): -61.00
}

# 动态初始库存
initial_inventory = {
    '东门': 60, '南门': 60, '北门': 60, '一食堂': 70, '二食堂': 70, '三食堂': 70,
    '梅苑1栋': 70, '菊苑1栋': 70, '教学2楼': 50, '教学4楼': 50, '计算机学院': 60,
    '工程中心': 60, '网球场': 70, '体育馆': 70, '校医院': 70
}

# 初始化 N_data_adjusted
N_data_adjusted = pd.DataFrame(0.0, index=['7:00', '9:00', '12:00', '14:00', '18:00', '21:00', '23:00'], columns=P)
for i in P:
    N_data_adjusted.loc['7:00', i] = initial_inventory[i]
for (i, t), delta in provided_delta.items():
    if t + 1 < len(N_data_adjusted):
        N_data_adjusted.loc[N_data_adjusted.index[t + 1], i] = N_data_adjusted.loc[N_data_adjusted.index[t], i] + delta

# 补全缺失 Delta_N_user
Delta_N_user = {}
for t in T:
    for i in P:
        if (i, t) in provided_delta:
            delta = provided_delta[(i, t)]
        else:
            delta = N_data_adjusted.iloc[t + 1][i] - N_data_adjusted.iloc[t][i]
        delta = max(min(delta, 120), -120)
        Delta_N_user[(i, t)] = delta

# 推导 D_data
D_data = pd.DataFrame(0.0, index=['7:00-9:00', '9:00-12:00', '12:00-14:00', '14:00-18:00', '18:00-21:00', '21:00-23:00'], columns=P)
for t in T:
    for i in P:
        if Delta_N_user[(i, t)] > 0:
            D_data.loc[D_data.index[t], i] = Delta_N_user[(i, t)]

print("平滑后的净用户需求（Delta_N_user）：")
for t in T:
    for i in P:
        if abs(Delta_N_user[(i, t)]) > 50:
            print(f"{i} 在 t={t} 的净用户需求: {Delta_N_user[(i, t)]:.2f}")

# 定义模型函数
def solve_model(P, dist, N_data, D_data, Delta_N_user, T, model_name="model"):
    model = LpProblem(f"Bike_Sharing_Redistribution_{model_name}", LpMinimize)
    
    # 决策变量
    x = LpVariable.dicts("x", [(i, j, t) for i in P for j in P if i != j for t in T], lowBound=0, cat='Continuous')
    N = LpVariable.dicts("N", [(i, t) for i in P for t in range(7)], lowBound=0, cat='Continuous')
    s = LpVariable.dicts("s", [(i, t) for i in P for t in T], lowBound=0, cat='Continuous')
    s_flow_plus = LpVariable.dicts("s_flow_plus", T, lowBound=0, cat='Continuous')
    s_flow_minus = LpVariable.dicts("s_flow_minus", T, lowBound=0, cat='Continuous')
    path_active = LpVariable.dicts("path_active", [(i, j, t) for i in P for j in P if i != j for t in T], cat='Binary')
    point_active = LpVariable.dicts("point_active", [(i, t) for i in P for t in T], cat='Binary')
    
    # 目标函数
    c_new = dist * 3600 / 25 * 0.8
    model += lpSum(c_new[P.index(i), P.index(j)] * x[(i, j, t)] for i in P for j in P if i != j for t in T) + \
             lpSum(20 * s[(i, t)] for i in P for t in T) + \
             lpSum(1e4 * (s_flow_plus[t] + s_flow_minus[t]) for t in T)
    
    # 约束：初始库存
    for i in P:
        model += N[(i, 0)] == N_data.loc['7:00', i]
    
    # 约束：流量平衡
    for t in T:
        for i in P:
            inflow = lpSum(x[(j, i, t)] for j in P if j != i)
            outflow = lpSum(x[(i, j, t)] for j in P if j != i)
            model += N[(i, t)] + inflow + s_flow_plus[t] == outflow + N[(i, t + 1)] + Delta_N_user[(i, t)] + s_flow_minus[t]
    
    # 约束：需求满足
    for t in T:
        for i in P:
            inflow = lpSum(x[(j, i, t)] for j in P if j != i)
            outflow = lpSum(x[(i, j, t)] for j in P if j != i)
            model += N[(i, t)] + inflow - outflow + s[(i, t)] >= D_data.loc[D_data.index[t], i]
    
    # 约束：调度容量
    for t in T:
        model += lpSum(x[(i, j, t)] for i in P for j in P if i != j) <= 600
        model += lpSum(x[(i, j, t)] for i in P for j in P if i != j) >= 80
    
    # 约束：最低库存
    for t in T:
        for i in P:
            model += N[(i, t)] >= 5
    
    # 约束：路径多样性
    for t in T:
        for i in P:
            for j in P:
                if i != j:
                    model += x[(i, j, t)] <= 600 * path_active[(i, j, t)]
        model += lpSum(path_active[(i, j, t)] for i in P for j in P if i != j) >= 3
    
    # 约束：正需求点参与
    for t in T:
        for i in P:
            if Delta_N_user[(i, t)] > 50:
                model += lpSum(x[(i, j, t)] for j in P if j != i) + lpSum(x[(j, i, t)] for j in P if j != i) <= 600 * point_active[(i, t)]
                model += point_active[(i, t)] == 1
    
    # 保存模型文件
    model.writeLP(f"{model_name}.lp")
    
    # 求解
    model.solve()
    
    # 输出状态
    print(f"\n{model_name} 模型状态: {LpStatus[model.status]}")
    if model.status != 1:
        print(f"警告：{model_name} 无解，检查 {model_name}.lp 文件！")
    
    # 计算效率
    T_total = value(model.objective) if model.status == 1 else float('inf')
    S_eff = (129600 - T_total) / 129600 if T_total != float('inf') else 0
    D_sat = []
    for i in P:
        for t in T:
            N_prime = value(N[(i, t)]) + sum(value(x[(j, i, t)]) for j in P if j != i) - sum(value(x[(i, j, t)]) for j in P if j != i) if model.status == 1 else N_data.iloc[t][i]
            D_it = D_data.loc[D_data.index[t], i]
            D_sat_it = 100 if D_it == 0 else min(1, N_prime / D_it) * 100
            D_sat.append(D_sat_it)
            if D_sat_it < 100 and D_it > 0:
                print(f"未满足需求：{i} 在 t={t}，实际库存 {N_prime:.2f}，需求 {D_it:.2f}，满足率 {D_sat_it:.2f}%")
    D_sat = np.mean(D_sat)
    total_demand = D_data.sum()
    U_i = total_demand / total_demand.max() * 100
    U = U_i.mean()
    E_overall = 0.4 * S_eff + 0.3 * D_sat / 100 + 0.3 * U / 100
    
    # 输出调度量和松弛变量
    if model.status == 1:
        print(f"\n{model_name} 调度量（x[i,j,t] > 0）：")
        for t in T:
            for i in P:
                for j in P:
                    if i != j and value(x[(i, j, t)]) > 0:
                        print(f"x[{i} -> {j}, t={t}] = {value(x[(i, j, t)]):.2f}")
        print(f"\n{model_name} 松弛变量（s[i,t] > 0）：")
        for t in T:
            for i in P:
                if value(s[(i, t)]) > 0:
                    print(f"s[{i}, t={t}] = {value(s[(i, t)]):.2f}")
        print(f"\n{model_name} 流量松弛变量（s_flow_plus[t], s_flow_minus[t] > 0）：")
        for t in T:
            if value(s_flow_plus[t]) > 0:
                print(f"s_flow_plus[t={t}] = {value(s_flow_plus[t]):.2f}")
            if value(s_flow_minus[t]) > 0:
                print(f"s_flow_minus[t={t}] = {value(s_flow_minus[t]):.2f}")
    
    return T_total, S_eff, D_sat, U, E_overall, x

# 当前布局效率
print("\n计算当前布局效率...")
T_total, S_eff, D_sat, U, E_overall, x_current = solve_model(P, dist, N_data_adjusted, D_data, Delta_N_user, T, "current")

# 输出当前布局效率
print("\n当前布局效率：")
print(f"调度时间: {T_total:.2f} 秒")
print(f"调度效率: {S_eff:.4f}")
print(f"需求满足率: {D_sat:.2f}%")
print(f"点位利用率: {U:.2f}%")
print(f"综合运营效率: {E_overall:.4f}")

# 调整后布局
remove_points = ['网球场', '体育馆']
new_points = [
    {'name': '教学新区', 'x': 924, 'y': -1276},
    {'name': '校门新区', 'x': 1082, 'y': -2230}
]
P_new = [p for p in P if p not in remove_points] + [p['name'] for p in new_points]

# 更新坐标
points_new = points.loc[P_new[:-2]].copy()
for p in new_points:
    points_new.loc[p['name']] = [p['x'], p['y']]

# 更新距离矩阵
dist_matrix_new = pd.DataFrame(index=P_new, columns=P_new)
for i in P_new:
    for j in P_new:
        if i == j:
            dist_matrix_new.loc[i, j] = 0
        elif i in P and j in P:
            dist_matrix_new.loc[i, j] = dist_matrix.loc[i, j]
        else:
            dist_matrix_new.loc[i, j] = np.sqrt((points_new.loc[i, 'x'] - points_new.loc[j, 'x'])**2 + (points_new.loc[i, 'y'] - points_new.loc[j, 'y'])**2) / 1000
dist_new = dist_matrix_new.values

# 更新需求数据
D_data_new = D_data[P_new[:-2]].copy()
for p in new_points:
    D_data_new[p['name']] = 0
D_data_new['教学新区'] = (0.9 * D_data['教学2楼'] + 0.1 * D_data['教学4楼'])
D_data_new['校门新区'] = (0.6 * D_data['南门'] + 0.4 * D_data['东门'])
D_data_new['教学2楼'] *= 0.1
D_data_new['教学4楼'] *= 0.9
D_data_new['南门'] *= 0.4
D_data_new['东门'] *= 0.4

# 更新库存数据
N_data_new = N_data_adjusted[P_new[:-2]].copy()
for p in new_points:
    N_data_new[p['name']] = 0
N_data_new['教学新区'] = (0.9 * N_data_adjusted['教学2楼'] + 0.1 * N_data_adjusted['教学4楼'])
N_data_new['校门新区'] = (0.6 * N_data_adjusted['南门'] + 0.4 * N_data_adjusted['东门'])
N_data_new['教学2楼'] *= 0.1
N_data_new['教学4楼'] *= 0.9
N_data_new['南门'] *= 0.4
N_data_new['东门'] *= 0.4

# 为新点位增加初始库存
N_data_new.loc['7:00', '教学新区'] += 80
N_data_new.loc['7:00', '校门新区'] += 90

# 平滑调整后净用户需求
print("\n平滑后的调整后净用户需求（Delta_N_user_new）：")
Delta_N_user_new = {}
for t in T:
    for i in P_new:
        delta = N_data_new.iloc[t + 1][i] - N_data_new.iloc[t][i]
        delta = max(min(delta, 120), -120)
        Delta_N_user_new[(i, t)] = delta
        if abs(Delta_N_user_new[(i, t)]) > 50:
            print(f"{i} 在 t={t} 的净用户需求: {Delta_N_user_new[(i, t)]:.2f}")

# 调整后布局效率
print("\n计算调整后布局效率...")
T_total_new, S_eff_new, D_sat_new, U_new, E_overall_new, x_adjusted = solve_model(P_new, dist_new, N_data_new, D_data_new, Delta_N_user_new, T, "adjusted")

# 输出调整后布局效率
print("\n调整后布局效率：")
print(f"调度时间: {T_total_new:.2f} 秒")
print(f"调度效率: {S_eff_new:.4f}")
print(f"需求满足率: {D_sat_new:.2f}%")
print(f"点位利用率: {U_new:.2f}%")
print(f"综合运营效率: {E_overall_new:.4f}")
print(f"效率提升: {(E_overall_new - E_overall) / E_overall * 100:.2f}%")

# 输出点位调整方案
print("\n移除点位：", remove_points)
print("新增点位：")
for p in new_points:
    print(f"{p['name']}: 坐标({p['x']:.2f}, {p['y']:.2f})")

# 可视化：点位分布图
plt.figure(figsize=(10, 8))
for i in points_new.index:
    if i in ['教学新区', '校门新区']:
        plt.scatter(points_new.loc[i, 'x'], points_new.loc[i, 'y'], c='red', s=100, label='New Points' if i == '教学新区' else "")
    elif i in remove_points:
        plt.scatter(points.loc[i, 'x'], points.loc[i, 'y'], c='gray', s=50, alpha=0.5, label='Removed Points' if i == '网球场' else "")
    else:
        plt.scatter(points_new.loc[i, 'x'], points_new.loc[i, 'y'], c='blue', s=50, label='Original Points' if i == '东门' else "")
    plt.text(points_new.loc[i, 'x'] + 20, points_new.loc[i, 'y'], i, fontsize=9)
plt.title('')
plt.xlabel('X Coordinate (meters)')
plt.ylabel('Y Coordinate (meters)')
plt.legend()
plt.grid(True)
plt.savefig('point_distribution.png')
plt.show()

# 可视化：t=0 的调度路径图
plt.figure(figsize=(10, 8))
for i in points_new.index:
    if i in ['教学新区', '校门新区']:
        plt.scatter(points_new.loc[i, 'x'], points_new.loc[i, 'y'], c='red', s=100)
    else:
        plt.scatter(points_new.loc[i, 'x'], points_new.loc[i, 'y'], c='blue', s=50)
    plt.text(points_new.loc[i, 'x'] + 20, points_new.loc[i, 'y'], i, fontsize=9)
for i in P_new:
    for j in P_new:
        if i != j and x_adjusted[(i, j, 0)].value() > 0:
            plt.arrow(points_new.loc[i, 'x'], points_new.loc[i, 'y'],
                      points_new.loc[j, 'x'] - points_new.loc[i, 'x'],
                      points_new.loc[j, 'y'] - points_new.loc[i, 'y'],
                      width=x_adjusted[(i, j, 0)].value()/10, color='green', alpha=0.5,
                      length_includes_head=True)
plt.title('')
plt.xlabel('X Coordinate (meters)')
plt.ylabel('Y Coordinate (meters)')
plt.grid(True)
plt.savefig('scheduling_paths_t0.png')
plt.show()