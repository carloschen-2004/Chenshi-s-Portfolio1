#禁忌搜索，最优解2366

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
        self.p_d = 50  # 容量违反惩罚系数
        self.p_t = 100  # 时间窗违反惩罚系数

# 函数：读取数据文件（保留原函数）
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

# 计算两点间的欧几里得距离
def calculate_distance(node1, node2, v=1):
    return math.sqrt((node1.x_coord - node2.x_coord)**2 + (node1.y_coord - node2.y_coord)**2) / v

# Clarke-Wright 节约算法生成初始解
def clarke_wright_init(model):
    sol = Sol()
    customers = list(range(model.number))
    routes = [[c] for c in customers]  # 初始时每个客户一条路径
    savings = []

    # 计算节约值
    depot = model.depot
    for i in range(model.number):
        for j in range(i + 1, model.number):
            ci, cj = model.customer[i], model.customer[j]
            dist_di = calculate_distance(depot, ci)
            dist_dj = calculate_distance(depot, cj)
            dist_ij = calculate_distance(ci, cj)
            saving = dist_di + dist_dj - dist_ij
            savings.append((saving, i, j))
    savings.sort(reverse=True)  # 按节约值降序排序

    # 合并路径
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

        # 检查合并后是否满足约束
        new_route = route_i + route_j
        load = sum(model.customer[c].demand for c in new_route)
        if load > model.capacity:
            continue

        # 检查时间窗约束
        current_time = 0
        prev = depot
        feasible = True
        for c in new_route:
            travel_time = calculate_distance(prev, model.customer[c], model.v)
            arrival = max(current_time + travel_time, model.customer[c].et)
            if arrival > model.customer[c].lt:
                feasible = False
                break
            current_time = arrival + model.customer[c].st
            prev = model.customer[c]
        if not feasible:
            continue

        # 合并路径
        routes.remove(route_i)
        routes.remove(route_j)
        routes.append(new_route)

    # 移除空路径
    routes = [route for route in routes if route]

    # 构造染色体
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
    # 移除空路径
    routes = [route for route in routes if route]
    return routes

# 计算单条路径的距离和违反值（增量计算）
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
        # 计算距离
        depot = self.model.depot
        distance = 0
        for i in range(len(self.route)-1):
            distance += calculate_distance(self.model.customer[self.route[i]], self.model.customer[self.route[i+1]])
        if self.route:
            distance += calculate_distance(depot, self.model.customer[self.route[0]])
            distance += calculate_distance(depot, self.model.customer[self.route[-1]])
        self.distance = distance

        # 计算违反值
        self.overD = max(sum(self.model.customer[c].demand for c in self.route) - self.model.capacity, 0)
        prev = depot
        prev_time = 0
        self.overT = 0
        for c in self.route:
            travel_time = calculate_distance(prev, self.model.customer[c], self.model.v)
            arrival = max(prev_time + travel_time, self.model.customer[c].et)
            if arrival > self.model.customer[c].lt:
                self.overT += arrival - self.model.customer[c].lt
            prev_time = arrival + self.model.customer[c].st
            prev = self.model.customer[c]

# 计算目标值
def calculate_obj(sol, model):
    route_infos = [RouteInfo(route, model) for route in sol.routes]
    distances = [info.distance for info in route_infos]
    overD = [info.overD for info in route_infos]
    overT = [info.overT for info in route_infos]
    sol.distances = distances
    sol.overD = overD
    sol.overT = overT
    sol.obj = sum(distances) + model.p_d * sum(overD) + model.p_t * sum(overT)
    return sol.obj

# 2-opt 优化（路径内）
def two_opt(route, model):
    if len(route) < 4:
        return route
    best_route = route[:]
    route_info = RouteInfo(best_route, model)
    best_obj = route_info.distance + model.p_d * route_info.overD + model.p_t * route_info.overT
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best_route)-2):
            for j in range(i+1, len(best_route)):
                new_route = best_route[:i] + best_route[i:j][::-1] + best_route[j:]
                new_info = RouteInfo(new_route, model)
                new_obj = new_info.distance + model.p_d * new_info.overD + model.p_t * new_info.overT
                if new_obj < best_obj:
                    best_route = new_route
                    best_obj = new_obj
                    improved = True
    return best_route

# 插入算子（路径间）
def insert_operator(route1, route2, model):
    if not route1 or not route2:
        return route1, route2
    best_route1, best_route2 = route1[:], route2[:]
    info1, info2 = RouteInfo(route1, model), RouteInfo(route2, model)
    best_obj = (info1.distance + info2.distance + 
                model.p_d * (info1.overD + info2.overD) + 
                model.p_t * (info1.overT + info2.overT))

    for i in range(len(route1)):
        customer = route1[i]
        if customer < 0 or customer >= model.number:  # 验证客户ID
            continue
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

# 交换算子（路径间）
def swap_operator(route1, route2, model):
    if not route1 or not route2:
        return route1, route2
    best_route1, best_route2 = route1[:], route2[:]
    info1, info2 = RouteInfo(route1, model), RouteInfo(route2, model)
    best_obj = (info1.distance + info2.distance + 
                model.p_d * (info1.overD + info2.overD) + 
                model.p_t * (info1.overT + info2.overT))

    for i in range(len(route1)):
        for j in range(len(route2)):
            if (route1[i] < 0 or route1[i] >= model.number or 
                route2[j] < 0 or route2[j] >= model.number):  # 验证客户ID
                continue
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

# 并行评估邻域解
def evaluate_neighbor(args):
    routes, r, operation, model = args
    temp_routes = routes[:]
    if operation == "2opt":
        new_route = two_opt(temp_routes[r], model)
        temp_routes[r] = new_route
        move = ("2opt", r, tuple(new_route))
    elif operation == "insert":
        i, j = r
        new_route_i, new_route_j = insert_operator(temp_routes[i], temp_routes[j], model)
        temp_routes[i], temp_routes[j] = new_route_i, new_route_j
        move = ("insert", i, j, routes[i][0] if routes[i] else -1, routes[j][0] if routes[j] else -1)
    else:  # swap
        i, j = r
        new_route_i, new_route_j = swap_operator(temp_routes[i], temp_routes[j], model)
        temp_routes[i], temp_routes[j] = new_route_i, new_route_j
        move = ("swap", i, j, routes[i][0] if routes[i] else -1, routes[j][0] if routes[j] else -1)
    
    # 移除空路径
    temp_routes = [route for route in temp_routes if route]
    
    temp_sol = Sol()
    temp_sol.routes = temp_routes
    obj = calculate_obj(temp_sol, model)
    return temp_sol, obj, move

# 禁忌搜索主函数
def TabuSearch(filepath, time_limit, capacity):
    start_time = time.time()
    model = Model()
    readExcel(filepath, model, capacity)

    # 调试：打印客户数量
    print(f"Number of customers: {model.number}")

    # 初始化解
    current_sol = clarke_wright_init(model)
    current_sol.routes = split_routes(current_sol.chrom, model)
    calculate_obj(current_sol, model)

    # 初始化最优解
    model.best_sol = copy.deepcopy(current_sol)

    # 初始化禁忌表
    tabu_list = deque(maxlen=30)
    tabu_tenure = 15
    tabu_dict = {}
    stagnation = 0  # 记录未改进的迭代次数
    history_best_obj = [model.best_sol.obj]

    # 并行处理
    pool = Pool(processes=cpu_count())

    it = 0
    while time.time() - start_time < time_limit:
        routes = current_sol.routes
        operations = []

        # 路径内优化（2-opt）
        for r in range(len(routes)):
            if routes[r]:
                operations.append((routes, r, "2opt", model))

        # 路径间优化（插入和交换）
        num_pairs = min(10, len(routes) * (len(routes) - 1) // 2)
        pairs = random.sample([(i, j) for i in range(len(routes)) for j in range(i+1, len(routes))], num_pairs)
        for i, j in pairs:
            if routes[i] and routes[j]:
                operations.append((routes, (i, j), "insert", model))
                operations.append((routes, (i, j), "swap", model))

        # 并行评估邻域解
        results = pool.map(evaluate_neighbor, operations)
        best_neighbor, best_neighbor_obj, best_move = min(results, key=lambda x: x[1])

        # 更新当前解
        if best_neighbor:
            current_sol = best_neighbor

            # 移除空路径
            current_sol.routes = [route for route in current_sol.routes if route]

            # 更新禁忌表
            if best_move:
                tabu_dict[best_move] = tabu_tenure
                tabu_list.append(best_move)

            # 更新最优解
            if current_sol.obj < model.best_sol.obj:
                model.best_sol = copy.deepcopy(current_sol)
                stagnation = 0
            else:
                stagnation += 1

        # 多样化策略（确保不生成空路径）
        if stagnation > 50:
            routes = current_sol.routes
            for _ in range(5):
                i, j = random.sample(range(len(routes)), 2)
                if routes[i] and routes[j]:  # 确保两条路径都不为空
                    ci = random.randint(0, len(routes[i])-1)
                    cj = random.randint(0, len(routes[j])-1)
                    routes[i][ci], routes[j][cj] = routes[j][cj], routes[i][ci]
            # 移除空路径
            routes = [route for route in routes if route]
            current_sol.routes = routes
            calculate_obj(current_sol, model)
            stagnation = 0

        # 动态调整惩罚系数
        total_overD = sum(current_sol.overD)
        total_overT = sum(current_sol.overT)
        if total_overD > 0:
            model.p_d *= 1.1
        else:
            model.p_d = max(50, model.p_d * 0.9)
        if total_overT > 0:
            model.p_t *= 1.1
        else:
            model.p_t = max(100, model.p_t * 0.9)

        # 更新禁忌表期限
        for move in list(tabu_dict.keys()):
            tabu_dict[move] -= 1
            if tabu_dict[move] <= 0:
                del tabu_dict[move]

        history_best_obj.append(model.best_sol.obj)
        it += 1

        if it % 10 == 0:
            print(f"第{it}次迭代：当前最优目标值 = {model.best_sol.obj}")

    pool.close()
    pool.join()

    # 验证路径有效性
    invalid_routes = []
    for route in model.best_sol.routes:
        for customer in route:
            if customer < 0 or customer >= model.number:
                invalid_routes.append(route)
                print(f"Invalid customer ID in route: {route}, customer ID: {customer}")
    if invalid_routes:
        print(f"Found invalid routes: {invalid_routes}")
        # 修复：移除无效客户
        model.best_sol.routes = [
            [c for c in route if 0 <= c < model.number]
            for route in model.best_sol.routes
        ]
        # 移除空路径
        model.best_sol.routes = [route for route in model.best_sol.routes if route]
        calculate_obj(model.best_sol, model)

    # 确保所有客户都被访问
    visited = set()
    for route in model.best_sol.routes:
        visited.update(route)
    missing = set(range(model.number)) - visited
    if missing:
        print(f"Missing customers: {missing}")
        # 将未访问的客户分配到新路径
        for c in missing:
            model.best_sol.routes.append([c])
        calculate_obj(model.best_sol, model)

    # 最终移除空路径
    model.best_sol.routes = [route for route in model.best_sol.routes if route]

    return model

# 求解结果输出（保留原函数）
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
        print("start time：", run_start)
        model = TabuSearch(filepath, time_limit, capacity)
        all_best_obj.append(model.best_sol.obj)
        all_best_Route.append(model.best_sol.routes)
        all_Route_dis.append(model.best_sol.distances)
        print("最短里程：", model.best_sol.obj)
        print("最短车辆路径方案：", model.best_sol.routes)
        run_end = time.perf_counter()
        print("end time：", run_end)
        print(f'run time={run_end-run_start}秒')
        i += 1
    output(all_best_obj, all_best_Route, all_Route_dis)