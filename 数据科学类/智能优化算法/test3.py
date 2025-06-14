# 核心策略：约束驱动初始解+引导式搜索+自适应算子+动态惩罚+高效扰动+并行优化

import numpy as np
import pandas as pd
import xlsxwriter
import math
import copy
import random
import time
from collections import deque
from multiprocessing import Pool, cpu_count

# Sol类，表示一个可行解
class Sol():
    def __init__(self):
        self.chrom = None
        self.routes = None
        self.obj = float('inf')
        self.distances = None
        self.overD = None
        self.overT = None

# Node类，表示一个节点
class Node():
    def __init__(self):
        self.id = 0
        self.name = ''
        self.x_coord = 0
        self.y_coord = 0
        self.demand = 0
        self.et = 0
        self.lt = 0
        self.st = 0

# Model类，存储算法参数
class Model():
    def __init__(self):
        self.best_sol = None
        self.depot = None
        self.customer = []
        self.number = 0
        self.SeqID = []
        self.opt_type = 1
        self.capacity = 0
        self.v = 1
        self.vehicle_num = 150
        self.fleet = None
        self.p_d = 50  # 容量惩罚因子
        self.p_t = 500  # 时间窗惩罚因子
        self.distance_matrix = None
        self.operator_weights = {'2opt': 0.2, 'insert': 0.1, 'swap': 0.1, 'relocate': 0.15, 
                                'cross': 0.1, 'or_opt': 0.1, 'two_opt_star': 0.25, 'reverse': 0.05, 'intra': 0.05}

# 读取Excel文件并预计算距离矩阵
def readExcel(filepath, model, capacity):
    model.capacity = capacity
    DF = pd.read_excel(filepath)
    id_NO = -1
    for i in range(DF.shape[0]):
        node = Node()
        node.id = id_NO
        node.name = f'C{i-1}'
        node.x_coord = DF['x_coord'][i]
        node.y_coord = DF['y_coord'][i]
        node.demand = DF['demand'][i]
        node.et = DF['et'][i]
        node.lt = DF['lt'][i]
        node.st = DF['st'][i]
        if i == 0:
            model.depot = node
        else:
            model.customer.append(node)
            model.SeqID.append(id_NO)
        id_NO += 1
    model.number = len(model.customer)
    model.fleet = list(range(model.number, model.number + model.vehicle_num))

    n = model.number + 1
    model.distance_matrix = np.zeros((n, n))
    nodes = [model.depot] + model.customer
    for i in range(n):
        for j in range(i + 1, n):
            dist = math.sqrt((nodes[i].x_coord - nodes[j].x_coord)**2 +
                             (nodes[i].y_coord - nodes[j].y_coord)**2) / model.v
            model.distance_matrix[i][j] = model.distance_matrix[j][i] = dist

# 改进的Clarke-Wright节约算法（仿OR-Tools约束优先）
def clarke_wright_init(model):
    sol = Sol()
    customers = list(range(model.number))
    routes = [[c] for c in customers]
    savings = []

    depot = model.depot
    for i in range(model.number):
        for j in range(i + 1, model.number):
            ci, cj = model.customer[i], model.customer[j]
            dist_di = model.distance_matrix[0][i + 1]
            dist_dj = model.distance_matrix[0][j + 1]
            dist_ij = model.distance_matrix[i + 1][j + 1]
            saving = dist_di + dist_dj - dist_ij
            time_window_i = ci.lt - ci.et
            time_window_j = cj.lt - cj.et
            saving += min(time_window_i, time_window_j) * 0.02
            saving -= (ci.demand + cj.demand) / model.capacity * 0.1
            savings.append((saving, i, j))
    savings.sort(reverse=True)

    while savings:
        _, i, j = savings.pop(0)
        route_i, route_j = None, None
        for r in routes:
            if i in r:
                route_i = r
            if j in r:
                route_j = r
        if route_i == route_j or not route_i or not route_j:
            continue

        new_route = route_i + route_j
        load = sum(model.customer[c].demand for c in new_route)
        if load > model.capacity:
            continue

        current_time = 0
        prev = 0
        feasible = True
        for c in new_route:
            travel_time = model.distance_matrix[prev][c + 1]
            arrival = max(current_time + travel_time, model.customer[c].et)
            if arrival > model.customer[c].lt:
                feasible = False
                break
            current_time = arrival + model.customer[c].st
            prev = c + 1
        if feasible:
            travel_time_back = model.distance_matrix[prev][0]
            return_time = current_time + travel_time_back
            if return_time > model.depot.lt:
                feasible = False
        if not feasible:
            continue

        routes.remove(route_i)
        routes.remove(route_j)
        routes.append(new_route)

    routes = [two_opt(route, model) for route in routes if route]

    chrom = []
    vehicle_count = 0
    for route in routes:
        chrom.extend(route)
        chrom.append(model.number + vehicle_count)
        vehicle_count += 1
    while len(chrom) < model.number + model.vehicle_num:
        chrom.append(model.number + vehicle_count)
        vehicle_count += 1

    sol.chrom = chrom[:model.number + model.vehicle_num]
    sol.routes = routes
    return sol

# 切分染色体为路径
def split_routes(chrom, model):
    routes = []
    current_route = []
    for i in chrom:
        if i < model.number:
            current_route.append(i)
        else:
            if current_route:
                routes.append(current_route)
                current_route = []
    if current_route:
        routes.append(current_route)
    return [route for route in routes if route]

# RouteInfo 类（计算路径信息）
class RouteInfo:
    def __init__(self, route, model):
        self.route = route
        self.model = model
        self.distance = 0
        self.overD = 0
        self.overT = 0
        self.calculate()

    def calculate(self):
        if not self.route:
            return
        self.overD = max(sum(self.model.customer[c].demand for c in self.route) - self.model.capacity, 0)
        if self.overD > 0:
            self.distance = 0
            self.overT = float('inf')
            return

        self.distance = self.model.distance_matrix[0][self.route[0] + 1]
        for i in range(len(self.route)-1):
            self.distance += self.model.distance_matrix[self.route[i] + 1][self.route[i + 1] + 1]
        self.distance += self.model.distance_matrix[self.route[-1] + 1][0]

        prev = 0
        prev_time = 0
        self.overT = 0
        for c in self.route:
            travel_time = self.model.distance_matrix[prev][c + 1]
            arrival = max(prev_time + travel_time, self.model.customer[c].et)
            if arrival > self.model.customer[c].lt:
                self.overT += arrival - self.model.customer[c].lt
                return
            prev_time = arrival + self.model.customer[c].st
            prev = c + 1
        travel_time_back = self.model.distance_matrix[prev][0]
        return_time = prev_time + travel_time_back
        if return_time > self.model.depot.lt:
            self.overT += return_time - self.model.depot.lt

# 计算目标值（引导式目标函数）
def calculate_obj(sol, model):
    route_infos = [RouteInfo(route, model) for route in sol.routes]
    sol.distances = [info.distance for info in route_infos]
    sol.overD = [info.overD for info in route_infos]
    sol.overT = [info.overT for info in route_infos]
    sol.obj = sum(sol.distances) + model.p_d * sum(sol.overD) + model.p_t * sum(sol.overT)
    return sol.obj

# 2-opt 优化
def two_opt(route, model):
    if len(route) < 4:
        return route
    best_route = route[:]
    best_info = RouteInfo(best_route, model)
    best_obj = best_info.distance + model.p_d * best_info.overD + model.p_t * best_info.overT
    for i in range(1, len(route)-2):
        for j in range(i+1, len(route)):
            new_route = best_route[:i] + best_route[i:j][::-1] + best_route[j:]
            new_info = RouteInfo(new_route, model)
            new_obj = new_info.distance + model.p_d * new_info.overD + model.p_t * new_info.overT
            if new_obj < best_obj:
                best_route = new_route
                best_obj = new_obj
    return best_route

# Insert 算子
def insert_operator(route1, route2, model):
    if not route1 or not route2:
        return route1, route2
    best_route1, best_route2 = route1[:], route2[:]
    best_info1, best_info2 = RouteInfo(route1, model), RouteInfo(route2, model)
    best_obj = (best_info1.distance + best_info2.distance +
                model.p_d * (best_info1.overD + best_info2.overD) +
                model.p_t * (best_info1.overT + best_info2.overT))

    for i in range(len(route1)):
        customer = route1[i]
        temp_route1 = route1[:i] + route1[i+1:]
        for j in range(len(route2) + 1):
            temp_route2 = route2[:j] + [customer] + route2[j:]
            info1_new, info2_new = RouteInfo(temp_route1, model), RouteInfo(temp_route2, model)
            new_obj = (info1_new.distance + info2_new.distance +
                       model.p_d * (info1_new.overD + info2_new.overD) +
                       model.p_t * (info1_new.overT + info2_new.overT))
            if new_obj < best_obj:
                best_obj = new_obj
                best_route1, best_route2 = temp_route1[:], temp_route2[:]
    return best_route1, best_route2

# Swap 算子
def swap_operator(route1, route2, model):
    if not route1 or not route2:
        return route1, route2
    best_route1, best_route2 = route1[:], route2[:]
    best_info1, best_info2 = RouteInfo(route1, model), RouteInfo(route2, model)
    best_obj = (best_info1.distance + best_info2.distance +
                model.p_d * (best_info1.overD + best_info2.overD) +
                model.p_t * (best_info1.overT + best_info2.overT))

    for i in range(len(route1)):
        for j in range(len(route2)):
            temp_route1 = route1[:i] + [route2[j]] + route1[i+1:]
            temp_route2 = route2[:j] + [route1[i]] + route2[j+1:]
            info1_new, info2_new = RouteInfo(temp_route1, model), RouteInfo(temp_route2, model)
            new_obj = (info1_new.distance + info2_new.distance +
                       model.p_d * (info1_new.overD + info2_new.overD) +
                       model.p_t * (info1_new.overT + info2_new.overT))
            if new_obj < best_obj:
                best_obj = new_obj
                best_route1, best_route2 = temp_route1[:], temp_route2[:]
    return best_route1, best_route2

# Relocate 算子
def relocate_operator(route1, route2, model):
    if not route1 or not route2:
        return route1, route2
    best_route1, best_route2 = route1[:], route2[:]
    best_info1, best_info2 = RouteInfo(route1, model), RouteInfo(route2, model)
    best_obj = (best_info1.distance + best_info2.distance +
                model.p_d * (best_info1.overD + best_info2.overD) +
                model.p_t * (best_info1.overT + best_info2.overT))

    for i in range(len(route1)):
        customer = route1[i]
        temp_route1 = route1[:i] + route1[i+1:]
        for j in range(len(route2) + 1):
            temp_route2 = route2[:j] + [customer] + route2[j:]
            info1_new, info2_new = RouteInfo(temp_route1, model), RouteInfo(temp_route2, model)
            new_obj = (info1_new.distance + info2_new.distance +
                       model.p_d * (info1_new.overD + info2_new.overD) +
                       model.p_t * (info1_new.overT + info2_new.overT))
            if new_obj < best_obj:
                best_obj = new_obj
                best_route1, best_route2 = temp_route1[:], temp_route2[:]
    return best_route1, best_route2

# Cross 算子
def cross_operator(route1, route2, model):
    if len(route1) < 4 or len(route2) < 4:
        return route1, route2
    best_route1, best_route2 = route1[:], route2[:]
    best_info1, best_info2 = RouteInfo(route1, model), RouteInfo(route2, model)
    best_obj = (best_info1.distance + best_info2.distance +
                model.p_d * (best_info1.overD + best_info2.overD) +
                model.p_t * (best_info1.overT + best_info2.overT))

    seg_len = random.randint(4, min(6, len(route1), len(route2)))
    pos1 = random.randint(0, len(route1)-seg_len)
    pos2 = random.randint(0, len(route2)-seg_len)
    seg1 = route1[pos1:pos1+seg_len]
    seg2 = route2[pos2:pos2+seg_len]
    temp_route1 = route1[:pos1] + seg2 + route1[pos1+seg_len:]
    temp_route2 = route2[:pos2] + seg1 + route2[pos2+seg_len:]
    info1_new, info2_new = RouteInfo(temp_route1, model), RouteInfo(temp_route2, model)
    new_obj = (info1_new.distance + info2_new.distance +
               model.p_d * (info1_new.overD + info2_new.overD) +
               model.p_t * (info1_new.overT + info2_new.overT))
    if new_obj < best_obj:
        best_route1, best_route2 = temp_route1[:], temp_route2[:]
    return best_route1, best_route2

# IntraRouteReordering 算子
def intra_route_reordering(route, model):
    if len(route) < 3:
        return route
    best_route = route[:]
    best_info = RouteInfo(best_route, model)
    best_obj = best_info.distance + model.p_d * best_info.overD + model.p_t * best_info.overT

    sorted_route = sorted(route, key=lambda c: (model.customer[c].et, model.distance_matrix[0][c + 1]))
    info = RouteInfo(sorted_route, model)
    new_obj = info.distance + model.p_d * info.overD + model.p_t * info.overT
    if new_obj < best_obj:
        best_route = sorted_route
        best_obj = new_obj

    for i in range(len(route)):
        for j in range(i + 1, len(route)):
            temp_route = route[:i] + [route[j]] + route[i+1:j] + [route[i]] + route[j+1:]
            info = RouteInfo(temp_route, model)
            new_obj = info.distance + model.p_d * info.overD + model.p_t * info.overT
            if new_obj < best_obj:
                best_route = temp_route
                best_obj = new_obj
    return best_route

# Or-opt 算子
def or_opt_operator(route1, route2, model):
    if len(route1) < 3 or not route2:
        return route1, route2
    best_route1, best_route2 = route1[:], route2[:]
    best_info1, best_info2 = RouteInfo(route1, model), RouteInfo(route2, model)
    best_obj = (best_info1.distance + best_info2.distance +
                model.p_d * (best_info1.overD + best_info2.overD) +
                model.p_t * (best_info1.overT + best_info2.overT))

    for start in range(len(route1) - 2):
        for length in range(2, min(4, len(route1) - start + 1)):
            segment = route1[start:start + length]
            temp_route1 = route1[:start] + route1[start + length:]
            for pos in range(len(route2) + 1):
                temp_route2 = route2[:pos] + segment + route2[pos:]
                info1_new, info2_new = RouteInfo(temp_route1, model), RouteInfo(temp_route2, model)
                new_obj = (info1_new.distance + info2_new.distance +
                           model.p_d * (info1_new.overD + info2_new.overD) +
                           model.p_t * (info1_new.overT + info2_new.overT))
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_route1, best_route2 = temp_route1[:], temp_route2[:]
    return best_route1, best_route2

# Reverse 算子
def reverse_operator(route, model):
    if len(route) < 3:
        return route
    best_route = route[:]
    best_info = RouteInfo(best_route, model)
    best_obj = best_info.distance + model.p_d * best_info.overD + model.p_t * best_info.overT

    start = random.randint(0, len(route) - 3)
    end = random.randint(start + 2, len(route))
    new_route = route[:start] + route[start:end][::-1] + route[end:]
    info = RouteInfo(new_route, model)
    new_obj = info.distance + model.p_d * info.overD + model.p_t * info.overT
    if new_obj < best_obj:
        best_route = new_route
    return best_route

# 2-opt* 算子（优化版）
def two_opt_star(route1, route2, model):
    if len(route1) < 2 or len(route2) < 2:
        return route1, route2
    best_route1, best_route2 = route1[:], route2[:]
    best_info1, best_info2 = RouteInfo(route1, model), RouteInfo(route2, model)
    best_obj = (best_info1.distance + best_info2.distance +
                model.p_d * (best_info1.overD + best_info2.overD) +
                model.p_t * (best_info1.overT + best_info2.overT))

    dist1_to_depot = model.distance_matrix[route1[-1] + 1][0]
    dist2_to_depot = model.distance_matrix[route2[-1] + 1][0]
    for i in range(1, len(route1)):
        for j in range(1, len(route2)):
            new_dist1_to_depot = model.distance_matrix[route1[i-1] + 1][route2[j] + 1]
            new_dist2_to_depot = model.distance_matrix[route2[j-1] + 1][route1[i] + 1]
            delta_dist = (new_dist1_to_depot + new_dist2_to_depot) - (dist1_to_depot + dist2_to_depot)
            if delta_dist > 0:
                continue
            new_route1 = route1[:i] + route2[j:]
            new_route2 = route2[:j] + route1[i:]
            info1_new, info2_new = RouteInfo(new_route1, model), RouteInfo(new_route2, model)
            new_obj = (info1_new.distance + info2_new.distance +
                       model.p_d * (info1_new.overD + info2_new.overD) +
                       model.p_t * (info1_new.overT + info2_new.overT))
            if new_obj < best_obj:
                best_obj = new_obj
                best_route1, best_route2 = new_route1[:], new_route2[:]
    return best_route1, best_route2

# 合并路径（约束驱动）
def merge_routes(routes, model):
    if not routes or len(routes) <= 7:
        return routes
    route_info = [(i, sum(model.customer[c].lt - model.customer[c].et for c in r),
                   len(r), RouteInfo(r, model).distance,
                   model.distance_matrix[r[0] + 1][r[-1] + 1]) for i, r in enumerate(routes)]
    route_info.sort(key=lambda x: (x[4], x[2], -x[1], x[3]))
    attempts = 0
    max_attempts = min(25, len(route_info))
    while len(routes) > 7 and attempts < max_attempts:
        i = route_info[attempts][0]
        for j in range(len(routes)):
            if i != j:
                new_route = routes[i] + routes[j]
                info = RouteInfo(new_route, model)
                if info.overD == 0 and info.overT == 0:
                    routes[i] = new_route
                    routes.pop(j)
                    return routes
        attempts += 1
    return routes

# 并行评估邻域解（约束优先，修复 TypeError）
def evaluate_neighbor(args):
    routes, r, operation, model = args
    temp_routes = [r[:] for r in routes]

    # 调试：检查 r 的类型和值
    if not isinstance(r, (int, tuple)):
        print(f"错误: r 的类型为 {type(r)}, 值: {r}, 操作: {operation}")
        return None, float('inf'), None

    if operation in ["insert", "swap", "relocate", "cross", "or_opt", "two_opt_star"]:
        if not isinstance(r, tuple) or len(r) != 2:
            print(f"错误: 操作 {operation} 需要元组 r, 得到: {r}")
            return None, float('inf'), None
        i, j = r
        route1, route2 = temp_routes[i], temp_routes[j]
        if not route1 or not route2:
            return None, float('inf'), None
        load1 = sum(model.customer[c].demand for c in route1)
        load2 = sum(model.customer[c].demand for c in route2)
        if load1 > model.capacity or load2 > model.capacity:
            return None, float('inf'), None
        time1, time2 = 0, 0
        prev1, prev2 = 0, 0
        for c in route1:
            travel_time = model.distance_matrix[prev1][c + 1]
            arrival = max(time1 + travel_time, model.customer[c].et)
            if arrival > model.customer[c].lt:
                return None, float('inf'), None
            time1 = arrival + model.customer[c].st
            prev1 = c + 1
        for c in route2:
            travel_time = model.distance_matrix[prev2][c + 1]
            arrival = max(time2 + travel_time, model.customer[c].et)
            if arrival > model.customer[c].lt:
                return None, float('inf'), None
            time2 = arrival + model.customer[c].st
            prev2 = c + 1

    if operation == "2opt":
        if not isinstance(r, int):
            print(f"错误: 2opt 需要整数 r, 得到: {r}")
            return None, float('inf'), None
        new_route = two_opt(temp_routes[r], model)
        temp_routes[r] = new_route
        move = ("2opt", r, tuple(new_route))
    elif operation == "insert":
        i, j = r
        new_route_i, new_route_j = insert_operator(temp_routes[i], temp_routes[j], model)
        temp_routes[i], temp_routes[j] = new_route_i, new_route_j
        move = ("insert", i, j, routes[i][0] if routes[i] else -1, routes[j][0] if routes[j] else -1)
    elif operation == "swap":
        i, j = r
        new_route_i, new_route_j = swap_operator(temp_routes[i], temp_routes[j], model)
        temp_routes[i], temp_routes[j] = new_route_i, new_route_j
        move = ("swap", i, j, routes[i][0] if routes[i] else -1, routes[j][0] if routes[j] else -1)
    elif operation == "relocate":
        i, j = r
        new_route_i, new_route_j = relocate_operator(temp_routes[i], temp_routes[j], model)
        temp_routes[i], temp_routes[j] = new_route_i, new_route_j
        move = ("relocate", i, j, routes[i][0] if routes[i] else -1)
    elif operation == "cross":
        i, j = r
        new_route_i, new_route_j = cross_operator(temp_routes[i], temp_routes[j], model)
        temp_routes[i], temp_routes[j] = new_route_i, new_route_j
        move = ("cross", i, j, routes[i][0] if routes[i] else -1, routes[j][0] if routes[j] else -1)
    elif operation == "or_opt":
        i, j = r
        new_route_i, new_route_j = or_opt_operator(temp_routes[i], temp_routes[j], model)
        temp_routes[i], temp_routes[j] = new_route_i, new_route_j
        move = ("or_opt", i, j, routes[i][0] if routes[i] else -1, routes[j][0] if routes[j] else -1)
    elif operation == "two_opt_star":
        i, j = r
        new_route_i, new_route_j = two_opt_star(temp_routes[i], temp_routes[j], model)
        temp_routes[i], temp_routes[j] = new_route_i, new_route_j
        move = ("two_opt_star", i, j, routes[i][0] if routes[i] else -1, routes[j][0] if routes[j] else -1)
    elif operation == "reverse":
        if not isinstance(r, int):
            print(f"错误: reverse 需要整数 r, 得到: {r}")
            return None, float('inf'), None
        new_route = reverse_operator(temp_routes[r], model)
        temp_routes[r] = new_route
        move = ("reverse", r, tuple(new_route))
    else:  # intra_route_reordering
        if not isinstance(r, int):
            print(f"错误: intra 需要整数 r, 得到: {r}")
            return None, float('inf'), None
        new_route = intra_route_reordering(temp_routes[r], model)
        temp_routes[r] = new_route
        move = ("intra", r, tuple(new_route))

    temp_routes = [route for route in temp_routes if route]
    temp_sol = Sol()
    temp_sol.routes = temp_routes
    obj = calculate_obj(temp_sol, model)
    return temp_sol, obj, move

# 自适应算子选择（仿OR-Tools动态调整）
def update_operator_weights(model, operation, improvement):
    if improvement > 0:
        model.operator_weights[operation] = min(model.operator_weights[operation] * 1.15, 0.5)
    else:
        model.operator_weights[operation] = max(model.operator_weights[operation] * 0.85, 0.05)
    total = sum(model.operator_weights.values())
    for op in model.operator_weights:
        model.operator_weights[op] /= total

# 约束驱动的扰动策略（仿OR-Tools引导式扰动）
def smart_perturb(routes, model, stagnation):
    if not routes:
        return routes
    disturb_type = random.choices([1, 2, 3, 4], weights=[0.35, 0.2, 0.2, 0.25])[0]
    
    if disturb_type == 1:  # 移除违反约束的客户并最佳插入
        violation_routes = [(i, RouteInfo(r, model).overD + RouteInfo(r, model).overT) for i, r in enumerate(routes)]
        violation_routes.sort(key=lambda x: x[1], reverse=True)
        remove_count = min(15 + stagnation * 2, 40)
        removed_customers = []
        for i, _ in violation_routes[:min(len(violation_routes), remove_count // 2)]:
            if routes[i]:
                customer = random.choice(routes[i])
                routes[i].remove(customer)
                removed_customers.append(customer)
        for _ in range(remove_count - len(removed_customers)):
            r = random.choice([i for i, r in enumerate(routes) if r])
            if routes[r]:
                customer = random.choice(routes[r])
                routes[r].remove(customer)
                removed_customers.append(customer)
        routes = [r for r in routes if r]
        for customer in removed_customers:
            best_insertion = None
            best_obj = float('inf')
            for i in range(len(routes)):
                for pos in range(len(routes[i]) + 1):
                    temp_route = routes[i][:pos] + [customer] + routes[i][pos:]
                    info = RouteInfo(temp_route, model)
                    obj = info.distance + model.p_d * info.overD + model.p_t * info.overT
                    if obj < best_obj:
                        best_obj = obj
                        best_insertion = (i, pos)
            if best_insertion:
                i, pos = best_insertion
                routes[i] = routes[i][:pos] + [customer] + routes[i][pos:]
            else:
                routes.append([customer])

    elif disturb_type == 2:  # 路径分割
        r = random.choice([i for i, r in enumerate(routes) if len(r) >= 4])
        split_point = len(routes[r]) // 2
        new_route1 = routes[r][:split_point]
        new_route2 = routes[r][split_point:]
        routes[r] = new_route1
        routes.append(new_route2)
        routes = merge_routes(routes, model)

    elif disturb_type == 3:  # 路径合并
        if len(routes) > 1:
            r1, r2 = random.sample(range(len(routes)), 2)
            merged_route = routes[r1] + routes[r2]
            routes.pop(max(r1, r2))
            routes.pop(min(r1, r2))
            split_point = len(merged_route) // 2
            routes.append(merged_route[:split_point])
            routes.append(merged_route[split_point:])
            routes = merge_routes(routes, model)

    else:  # 全局重置（约束驱动）
        all_customers = []
        for r in routes:
            all_customers.extend(r)
        random.shuffle(all_customers)
        routes = []
        current_route = []
        current_load = 0
        current_time = 0
        prev = 0
        for c in sorted(all_customers, key=lambda c: model.customer[c].et):
            demand = model.customer[c].demand
            travel_time = model.distance_matrix[prev][c + 1]
            arrival = max(current_time + travel_time, model.customer[c].et)
            if (current_load + demand <= model.capacity and
                arrival <= model.customer[c].lt):
                current_route.append(c)
                current_load += demand
                current_time = arrival + model.customer[c].st
                prev = c + 1
            else:
                if current_route:
                    routes.append(current_route)
                current_route = [c]
                current_load = demand
                current_time = max(travel_time, model.customer[c].et) + model.customer[c].st
                prev = c + 1
        if current_route:
            routes.append(current_route)

    return [r for r in routes if r]

# Tabu Search 主函数（仿OR-Tools约束驱动搜索）
def TabuSearch(filepath, time_limit, capacity):
    start_time = time.time()
    model = Model()
    readExcel(filepath, model, capacity)

    print(f"客户数量: {model.number}")

    current_sol = clarke_wright_init(model)
    current_sol.routes = split_routes(current_sol.chrom, model)
    calculate_obj(current_sol, model)

    model.best_sol = copy.deepcopy(current_sol)

    tabu_list = deque(maxlen=50)
    tabu_dict = {}
    stagnation = 0
    pool = Pool(processes=cpu_count())

    temp = 10000
    cooling_rate = 0.98
    last_best_obj = current_sol.obj
    it = 0

    while time.time() - start_time < time_limit:
        routes = current_sol.routes
        operations = []

        # 单路径操作：确保 r 是整数
        for r in range(len(routes)):
            if routes[r]:
                operations.append((routes, r, "2opt", model))
                operations.append((routes, r, "intra", model))
                operations.append((routes, r, "reverse", model))

        # 双路径操作：确保 r 是元组 (i, j)
        num_pairs = min(60, len(routes) * (len(routes) - 1) // 2)
        pairs = random.sample([(i, j) for i in range(len(routes)) for j in range(i+1, len(routes))], num_pairs)
        for i, j in pairs:
            if routes[i] and routes[j]:
                op = random.choices(list(model.operator_weights.keys()), 
                                   weights=list(model.operator_weights.values()))[0]
                if op in ["insert", "swap", "relocate", "cross", "or_opt", "two_opt_star"]:
                    operations.append((routes, (i, j), op, model))
                else:
                    # 如果随机选到单路径操作，分配到随机单路径
                    r = random.choice([i, j])
                    operations.append((routes, r, op, model))

        results = pool.map(evaluate_neighbor, operations)
        valid_results = [(sol, obj, move) for sol, obj, move in results if sol is not None and obj != float('inf')]
        if not valid_results:
            continue
        best_neighbor, best_neighbor_obj, best_move = min(valid_results, key=lambda x: x[1])

        if best_neighbor:
            delta = best_neighbor_obj - current_sol.obj
            improvement = last_best_obj - best_neighbor_obj
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_sol = best_neighbor
                current_sol.routes = [route for route in current_sol.routes if route]
                tabu_tenure = max(5, min(12, 5 + int(stagnation / 4)))
                tabu_dict[best_move] = tabu_tenure
                tabu_list.append(best_move)

                if current_sol.obj < model.best_sol.obj:
                    model.best_sol = copy.deepcopy(current_sol)
                    last_best_obj = current_sol.obj
                    stagnation = 0
                    update_operator_weights(model, best_move[0], improvement)
                else:
                    stagnation += 1
                    update_operator_weights(model, best_move[0], 0)

        if stagnation > 6:
            current_sol.routes = smart_perturb(current_sol.routes, model, stagnation)
            calculate_obj(current_sol, model)
            stagnation = 0

        current_sol.routes = merge_routes(current_sol.routes, model)

        total_overD = sum(current_sol.overD)
        total_overT = sum(current_sol.overT)
        if total_overD > 0:
            model.p_d = min(model.p_d * 1.4, 2000)
        else:
            model.p_d = max(50, model.p_d * 0.9)
        if total_overT > 0:
            model.p_t = min(model.p_t * 1.4, 10000)
        else:
            model.p_t = max(500, model.p_t * 0.9)

        for move in list(tabu_dict.keys()):
            tabu_dict[move] -= 1
            if tabu_dict[move] <= 0:
                del tabu_dict[move]

        temp *= cooling_rate

        it += 1
        if it % 10 == 0:
            print(f"第{it}次迭代：当前最优目标值 = {model.best_sol.obj}")

    pool.close()
    pool.join()

    invalid_routes = []
    for route in model.best_sol.routes:
        for customer in route:
            if customer < 0 or customer >= model.number:
                invalid_routes.append(route)
                print(f"无效客户ID在路径中: {route}, 客户ID: {customer}")
    if invalid_routes:
        print(f"发现无效路径: {invalid_routes}")
        model.best_sol.routes = [
            [c for c in route if 0 <= c < model.number]
            for route in model.best_sol.routes
        ]
        model.best_sol.routes = [route for route in model.best_sol.routes if route]
        calculate_obj(model.best_sol, model)

    visited = set()
    for route in model.best_sol.routes:
        visited.update(route)
    missing = set(range(model.number)) - visited
    if missing:
        print(f"缺失客户: {missing}")
        for c in missing:
            model.best_sol.routes.append([c])
        calculate_obj(model.best_sol, model)

    model.best_sol.routes = [route for route in model.best_sol.routes if route]
    return model

# 输出结果
def output(all_best_obj, all_best_Route, all_Route_dis):
    excel = xlsxwriter.Workbook('result.xlsx')
    excelsheet = excel.add_worksheet('sheet0')
    excelsheet.write(0, 0, 'best_cost')
    excelsheet.write(1, 0, min(all_best_obj))
    excelsheet.write(0, 1, 'worst_cost')
    excelsheet.write(1, 1, max(all_best_obj))
    excelsheet.write(0, 2, 'aver_cost')
    excelsheet.write(1, 2, sum(all_best_obj)/len(all_best_obj))
    for i in range(len(all_best_obj)):
        excelsheet.write(0, i+3, f'cost{i+1}')
        excelsheet.write(1, i+3, all_best_obj[i])
    for i in range(len(all_best_obj)):
        excelsheet = excel.add_worksheet(f'sheet{i+1}')
        Route = all_best_Route[i]
        excelsheet.write(0, 0, f'cost{i+1}')
        excelsheet.write(0, 1, all_best_obj[i])
        for r in range(len(Route)):
            excelsheet.write(r+1, 0, f'v{r+1}')
            excelsheet.write(r+1, 1, str(Route[r]))
            excelsheet.write(r+1, 2, str(all_Route_dis[i][r]))
    excel.close()

# 主程序
if __name__ == '__main__':
    filepath = r'/Users/keep-rational/Desktop/算法排名作业(VRPTW)-200客户/代码/Instances/RC2_2_10.xlsx'
    run_time = 5
    time_limit = 180
    capacity = 1000
    all_best_obj = []
    all_best_Route = []
    all_Route_dis = []
    i = 0
    while i < run_time:
        print(f"*====================第{i+1}次运行====================*")
        run_start = time.perf_counter()
        print("开始时间：", run_start)
        model = TabuSearch(filepath, time_limit, capacity)
        all_best_obj.append(model.best_sol.obj)
        all_best_Route.append(model.best_sol.routes)
        all_Route_dis.append(model.best_sol.distances)
        print("最短里程：", model.best_sol.obj)
        print("最优路径方案：", model.best_sol.routes)
        run_end = time.perf_counter()
        print("结束时间：", run_end)
        print(f'运行时间={run_end-run_start}秒')
        i += 1
    output(all_best_obj, all_best_Route, all_Route_dis)