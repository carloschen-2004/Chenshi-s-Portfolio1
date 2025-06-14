#这个版本 在4100左右

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlsxwriter
import math
import copy
import random
import time

# Sol类，表示一个可行解，等于一条染色体,存储在model.pop中
class Sol():
    def __init__(self):
        self.chrom = None
        self.Route = None
        self.Route_dis = None
        self.Route_overD = None
        self.Route_overT = None
        self.obj = None
        self.fit = None

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
        self.pop = []
        self.depot = None
        self.customer = []
        self.number = 0
        self.SeqID = []
        self.opt_type = 1
        self.capacity = 0
        self.v = 1
        self.pc = 0.9
        self.pm = 0.1
        self.popsize = 100
        self.vehicle_num = 150
        self.fleet = None
        self.p_d = 10
        self.p_t = 500

# 函数：读取数据文件
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

# 函数：最近邻启发式初始化
def nearest_neighbor_init(model):
    sol = Sol()
    unvisited = list(range(model.number))
    chrom = []
    vehicle_count = 0
    while unvisited:
        current = model.depot
        route = []
        load = 0
        current_time = 0
        while unvisited:
            nearest = min(unvisited, key=lambda x: math.sqrt(
                (current.x_coord - model.customer[x].x_coord)**2 +
                (current.y_coord - model.customer[x].y_coord)**2))
            travel_time = math.sqrt((current.x_coord - model.customer[nearest].x_coord)**2 +
                                    (current.y_coord - model.customer[nearest].y_coord)**2) / model.v
            arrival = max(current_time + travel_time, model.customer[nearest].et)
            if (load + model.customer[nearest].demand <= model.capacity and arrival <= model.customer[nearest].lt):
                route.append(nearest)
                load += model.customer[nearest].demand
                current_time = arrival + model.customer[nearest].st
                unvisited.remove(nearest)
                current = model.customer[nearest]
            else:
                break
        if route:
            chrom.extend(route)
            if unvisited:
                chrom.append(model.number + vehicle_count)
                vehicle_count += 1
    missing = set(range(model.number)) - set(chrom)
    if missing:
        chrom.extend(list(missing))
        chrom.append(model.number + vehicle_count)
        vehicle_count += 1
    while len(chrom) < model.number + model.vehicle_num:
        chrom.append(model.number + vehicle_count)
        vehicle_count += 1
    sol.chrom = chrom[:model.number + model.vehicle_num]
    return sol

# 函数：初始种群生成
def initialpop(model):
    temp_chrom = list(range(0, model.number + model.vehicle_num))
    for i in range(model.popsize):
        sol = Sol()
        if i < model.popsize // 2:
            sol = nearest_neighbor_init(model)
        else:
            random.shuffle(temp_chrom)
            sol.chrom = copy.deepcopy(temp_chrom)
        model.pop.append(sol)

# 函数：解码，按车辆切分染色体，得到车辆路径方案解
def split_routes(chrom, model):
    routes = []
    current_route = []
    current_load = 0
    current_time = 0
    prev = model.depot
    for i in chrom:
        if i < model.number:
            travel_time = math.sqrt((prev.x_coord - model.customer[i].x_coord)**2 +
                                    (prev.y_coord - model.customer[i].y_coord)**2) / model.v
            arrival = max(current_time + travel_time, model.customer[i].et)
            if (current_load + model.customer[i].demand <= model.capacity and
                arrival <= model.customer[i].lt):
                current_route.append(i)
                current_load += model.customer[i].demand
                current_time = arrival + model.customer[i].st
                prev = model.customer[i]
            else:
                if current_route and current_load >= model.capacity * 0.5:
                    routes.append(current_route)
                    current_route = [i]
                    current_load = model.customer[i].demand
                    current_time = model.customer[i].et + model.customer[i].st
                    prev = model.customer[i]
                else:
                    current_route.append(i)
                    current_load += model.customer[i].demand
                    current_time = arrival + model.customer[i].st
                    prev = model.customer[i]
        elif i >= model.number:
            if current_route:
                routes.append(current_route)
                current_route = []
                current_load = 0
                current_time = 0
                prev = model.depot
    if current_route:
        routes.append(current_route)
    return routes

# 函数：单条车辆路径里程计算
def caldistance(route, model):
    if not route:
        return 0
    distance = 0
    depot = model.depot
    for i in range(len(route)-1):
        distance += math.sqrt((model.customer[route[i]].x_coord - model.customer[route[i+1]].x_coord)**2 +
                              (model.customer[route[i]].y_coord - model.customer[route[i+1]].y_coord)**2)
    F_customer = model.customer[route[0]]
    L_customer = model.customer[route[-1]]
    distance += math.sqrt((depot.x_coord - F_customer.x_coord)**2 + (depot.y_coord - F_customer.y_coord)**2)
    distance += math.sqrt((depot.x_coord - L_customer.x_coord)**2 + (depot.y_coord - L_customer.y_coord)**2)
    return distance

# 函数：单条车辆路径违反值计算
def calviolations(route, model):
    if not route:
        return 0, 0
    overD = max(sum(model.customer[c].demand for c in route) - model.capacity, 0)
    overT = 0
    prev = model.depot
    prev_time = 0
    for index, c in enumerate(route):
        if index == 0:
            travel_time = math.sqrt((prev.x_coord - model.customer[c].x_coord)**2 +
                                    (prev.y_coord - model.customer[c].y_coord)**2) / model.v
            arrival = max(prev_time + travel_time, model.customer[c].et)
            if arrival > model.customer[c].lt:
                overT += arrival - model.customer[c].lt
            prev_time = arrival + model.customer[c].st
            prev = c
        else:
            travel_time = math.sqrt((model.customer[prev].x_coord - model.customer[c].x_coord)**2 +
                                    (model.customer[prev].y_coord - model.customer[c].y_coord)**2) / model.v
            arrival = max(prev_time + travel_time, model.customer[c].et)
            if arrival > model.customer[c].lt:
                overT += arrival - model.customer[c].lt
            prev_time = arrival + model.customer[c].st
            prev = c
    return overD, overT

# 函数：2-opt局部搜索
def two_opt(route, model):
    if len(route) < 4:
        return route
    best_route = route[:]
    best_dis = caldistance(best_route, model)
    best_overT = calviolations(best_route, model)[1]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best_route)-2):
            for j in range(i+1, len(best_route)):
                new_route = best_route[:i] + best_route[i:j][::-1] + best_route[j:]
                new_dis = caldistance(new_route, model)
                new_overT = calviolations(new_route, model)[1]
                if new_overT < best_overT or (new_overT == best_overT and new_dis < best_dis):
                    best_route = new_route
                    best_dis = new_dis
                    best_overT = new_overT
                    improved = True
    return best_route

# 函数：适应度值计算（修复：正确集成2-opt）
def calFit(model):
    objMAX = -float('inf')
    best_sol = Sol()
    best_sol.obj = float('inf')
    for sol in model.pop:
        Route = split_routes(sol.chrom, model)
        # 应用2-opt优化每条路径
        Route = [two_opt(route, model) for route in Route if route]
        Route_dis = []
        Route_overD = []
        Route_overT = []
        for route in Route:
            dis = caldistance(route, model)
            overD, overT = calviolations(route, model)
            Route_dis.append(dis)
            Route_overD.append(overD)
            Route_overT.append(overT)
        sol.Route = Route
        sol.Route_dis = Route_dis
        sol.Route_overD = Route_overD
        sol.Route_overT = Route_overT
        sol.obj = sum(Route_dis) + model.p_d * sum(Route_overD) + model.p_t * sum(Route_overT)
        if sol.obj > objMAX:
            objMAX = sol.obj
        if sol.obj < best_sol.obj:
            best_sol = copy.deepcopy(sol)
    
    for sol in model.pop:
        sol.fit = objMAX - sol.obj
    if best_sol.obj < model.best_sol.obj:
        model.best_sol = copy.deepcopy(best_sol)

# 函数：选择算子，二元锦标赛方法
def select(model):
    temp_pop = copy.deepcopy(model.pop)
    model.pop = []
    for i in range(model.popsize):
        f1_index = random.randint(0, len(temp_pop)-1)
        f2_index = random.randint(0, len(temp_pop)-1)
        f1_fit = temp_pop[f1_index].fit
        f2_fit = temp_pop[f2_index].fit
        if f1_fit < f2_fit:
            model.pop.append(temp_pop[f2_index])
        else:
            model.pop.append(temp_pop[f1_index])

# 函数：OX交叉算子
def cross(model):
    temp_pop = copy.deepcopy(model.pop)
    model.pop = []
    while True:
        father_index = random.randint(0, model.popsize-1)
        mother_index = random.randint(0, model.popsize-1)
        if father_index != mother_index:
            father = copy.deepcopy(temp_pop[father_index])
            mother = copy.deepcopy(temp_pop[mother_index])
            if random.random() < model.pc:
                cpoint1 = int(random.randint(0, model.number + model.vehicle_num - 1))
                cpoint2 = int(random.randint(cpoint1, model.number + model.vehicle_num - 1))
                new_father_f = []
                new_father_m = father.chrom[cpoint1:cpoint2+1]
                new_father_b = []
                new_mother_f = []
                new_mother_m = mother.chrom[cpoint1:cpoint2+1]
                new_mother_b = []
                for i in range(model.number + model.vehicle_num):
                    if len(new_father_f) < cpoint1:
                        if mother.chrom[i] not in new_father_m:
                            new_father_f.append(mother.chrom[i])
                    else:
                        if mother.chrom[i] not in new_father_m:
                            new_father_b.append(mother.chrom[i])
                for i in range(model.number + model.vehicle_num):
                    if len(new_mother_f) < cpoint1:
                        if father.chrom[i] not in new_mother_m:
                            new_mother_f.append(father.chrom[i])
                    else:
                        if father.chrom[i] not in new_mother_m:
                            new_mother_b.append(father.chrom[i])
                new_father = new_father_f + new_father_m + new_father_b
                father.chrom = copy.deepcopy(new_father)
                new_mother = new_mother_f + new_mother_m + new_mother_b
                mother.chrom = copy.deepcopy(new_mother)
                model.pop.append(copy.deepcopy(father))
                model.pop.append(copy.deepcopy(mother))
            else:
                model.pop.append(copy.deepcopy(father))
                model.pop.append(copy.deepcopy(mother))
            if len(model.pop) == model.popsize:
                break

# 函数：变异算子，二元突变
def mutation(model):
    temp_pop = copy.deepcopy(model.pop)
    model.pop = []
    while True:
        father_index = int(random.randint(0, model.popsize-1))
        father = copy.deepcopy(temp_pop[father_index])
        mpoint1 = random.randint(0, model.number + model.vehicle_num - 1)
        mpoint2 = random.randint(0, model.number + model.vehicle_num - 1)
        if mpoint1 != mpoint2:
            if random.random() < model.pm:
                point1 = father.chrom[mpoint1]
                father.chrom[mpoint1] = father.chrom[mpoint2]
                father.chrom[mpoint2] = point1
                model.pop.append(copy.deepcopy(father))
            else:
                model.pop.append(copy.deepcopy(father))
            if len(model.pop) == model.popsize:
                break

# 绘图函数：绘制收敛曲线图
def plot_obj(objlist):
    plt.figure()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(np.arange(1, len(objlist)+1), objlist)
    plt.xlabel('迭代次数')
    plt.ylabel('最优目标函数值')
    plt.grid()
    plt.xlim(1, len(objlist)+1)

# 绘图函数：绘制车辆行驶路径图
def plot_route(model):
    plt.figure()
    for route in model.best_sol.Route:
        x_coord = [model.depot.x_coord]
        y_coord = [model.depot.y_coord]
        for i in route:
            x_coord.append(model.customer[i].x_coord)
            y_coord.append(model.customer[i].y_coord)
            plt.text(model.customer[i].x_coord, model.customer[i].y_coord, model.customer[i].name, fontsize=5)
        x_coord.append(model.depot.x_coord)
        y_coord.append(model.depot.y_coord)
        plt.grid()
        plt.plot(x_coord, y_coord, 'b:', linewidth=0.5, marker='o', markersize=2)
    plt.plot(model.depot.x_coord, model.depot.y_coord, 'r', marker='*', markersize=10)
    plt.title('vehicle-route')
    plt.xlabel('x_coord')
    plt.ylabel('y_coord')

# 函数：求解结果输出（保留你的原始版本）
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

# 主函数：GA算法框架
def GA(filepath, time_limit, capacity):
    start_time = time.time()
    model = Model()
    readExcel(filepath, model, capacity)
    best_sol = Sol()
    best_sol.obj = float('inf')
    model.best_sol = best_sol
    initialpop(model)
    calFit(model)
    history_best_obj = []
    history_best_obj.append(model.best_sol.obj)
    it = 0
    while time.time() - start_time < time_limit:
        select(model)
        cross(model)
        mutation(model)
        calFit(model)
        history_best_obj.append(model.best_sol.obj)
        it += 1
    # plot_obj(history_best_obj)
    # plot_route(model)
    # plt.show()
    return model

if __name__ == '__main__':
    filepath = r'/Users/keep-rational/Desktop/算法排名作业(VRPTW)-200客户/代码/Instances/RC2_2_10.xlsx'
    run_time = 5
    time_limit = 180
    capacity = 1000
    all_best_obj = []
    all_best_Route = []
    all_Route_dis = []
    all_Route_overD = []
    all_Route_overT = []
    i = 0
    while i < run_time:
        print(f"*====================第{i+1}次运行====================*")
        run_start = time.perf_counter()
        print("start time：", run_start)
        model = GA(filepath, time_limit, capacity)
        all_best_obj.append(model.best_sol.obj)
        all_best_Route.append(model.best_sol.Route)
        all_Route_dis.append(model.best_sol.Route_dis)
        all_Route_overD.append(model.best_sol.Route_overD)
        all_Route_overT.append(model.best_sol.Route_overT)
        print("最短里程：", model.best_sol.obj)
        print("最短车辆路径方案：", model.best_sol.Route)
        run_end = time.perf_counter()
        print("end time：", run_end)
        print(f'run time={run_end-run_start}秒')
        i += 1
    # 修复：只传递 output 需要的参数
    output(all_best_obj, all_best_Route, all_Route_dis)