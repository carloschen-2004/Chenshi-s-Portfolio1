import pandas as pd
import numpy as np
import networkx as nx
import random
import math
import matplotlib.pyplot as plt

# 定义精确坐标
coordinates = {
    "东门": (1464, -1128), "南门": (1082, -2230), "北门": (954, -360),
    "一食堂": (634, -1362), "二食堂": (428, -1652), "三食堂": (864, -1816),
    "梅苑1栋": (606, -1954), "菊苑1栋": (452, -988), "教学2楼": (924, -1276),
    "教学4楼": (1106, -1580), "计算机学院": (970, -794), "工程中心": (1324, -712),
    "网球场": (704, -920), "体育馆": (442, -598), "校医院": (402, -1898),
    "运维处": (1450, -358)
}

# 目标点位列表
target_points = ["东门", "南门", "北门", "一食堂", "二食堂", "三食堂",
                 "梅苑1栋", "菊苑1栋", "教学2楼", "教学4楼", "计算机学院",
                 "工程中心", "网球场", "体育馆", "校医院"]

# 真实 table1 数据
table1 = pd.DataFrame(
    {
        "东门": [17, 71, 32, 56, 36, 103, 14],
        "南门": [14, 130, 59, 26, 110, 41, 46],
        "北门": [2, 66, 70, 66, 72, 27, 28],
        "一食堂": [73, 0, 91, 0, 33, 22, 82],
        "二食堂": [141, 0, 130, 0, 74, 71, 114],
        "三食堂": [109, 0, 147, 0, 61, 54, 123],
        "梅苑1栋": [130, 1, 81, 12, 67, 0, 128],
        "菊苑1栋": [136, 6, 79, 68, 95, 68, 126],
        "教学2楼": [0, 218, 52, 206, 35, 109, 30],
        "教学4楼": [0, 150, 29, 151, 26, 81, 20],
        "计算机学院": [0, 53, 14, 88, 48, 49, 17],
        "工程中心": [0, 59, 12, 78, 42, 76, 49],
        "网球场": [4, 20, 4, 17, 48, 8, 15],
        "体育馆": [1, 2, 2, 5, 34, 5, 12],
        "校医院": [5, 29, 6, 38, 6, 7, 11]
    },
    index=["7:00:00", "9:00:00", "12:00:00", "14:00:00", "18:00:00", "21:00:00", "23:00:00"]
)

time_weight = {"7:00": 0.2, "9:00": 0.4, "12:00": 1.1, "14:00": 0.9,
               "18:00": 0.6, "21:00": 0.5, "23:00": 0.2}
time_weight_full = {f"{k.split(':')[0]}:00:00": v for k, v in time_weight.items()}

G = nx.Graph()
nodes_all = ["运维处"] + target_points
for i in range(len(nodes_all)):
    for j in range(i + 1, len(nodes_all)):
        u, v = nodes_all[i], nodes_all[j]
        x1, y1 = coordinates[u]
        x2, y2 = coordinates[v]
        dist_km = np.hypot(x1 - x2, y1 - y2) / 1000
        G.add_edge(u, v, weight=dist_km)
        G.add_edge(v, u, weight=dist_km)

distance_matrix = np.zeros((len(nodes_all), len(nodes_all)))
for i, u in enumerate(nodes_all):
    for j, v in enumerate(nodes_all):
        if u == v:
            distance_matrix[i][j] = 0
        else:
            distance_matrix[i][j] = nx.dijkstra_path_length(G, u, v, weight="weight")

time_periods = [
    ("7:00:00", "9:00:00"),   
    ("9:00:00", "12:00:00"), 
    ("12:00:00", "14:00:00"), 
    ("14:00:00", "18:00:00"), 
    ("18:00:00", "21:00:00"), 
    ("21:00:00", "23:00:00")  
]

#设置参数
num_vehicles = 3
vehicle_capacity = 20
speed = 25  # km/h
max_time = 3600 * 2  
T0 = 1000
T_min = 1
alpha = 0.99
iterations_per_temp = 100

def compute_route_metrics(trip, distance_matrix, demands, speed):
    if len(trip) <= 1:
        return 0, 0, 0
    distance = 0
    for i in range(len(trip) - 1):
        distance += distance_matrix[trip[i]][trip[i + 1]]
    time_sec = (distance / speed) * 3600
    load = sum(demands[idx] for idx in trip[1:-1])
    return distance, time_sec, load

def is_trip_feasible(trip, distance_matrix, demands, speed, vehicle_capacity, max_time):
    if len(trip) <= 1:
        return True
    distance, time_sec, _ = compute_route_metrics(trip, distance_matrix, demands, speed)
    if time_sec > max_time:
        return False
    load = 0
    for idx in trip[1:-1]:
        new_load = load + demands[idx]
        if new_load < 0 or new_load > vehicle_capacity:
            return False
        load = new_load
    return True

def is_vehicle_feasible(trips, distance_matrix, demands, speed, vehicle_capacity, max_time):
    total_time = 0
    for trip in trips:
        if not is_trip_feasible(trip, distance_matrix, demands, speed, vehicle_capacity, max_time):
            return False
        _, time_sec, _ = compute_route_metrics(trip, distance_matrix, demands, speed)
        total_time += time_sec
    return total_time <= max_time

def compute_total_time(routes, distance_matrix, demands, speed):
    total_time = 0
    for vehicle_trips in routes:
        for trip in vehicle_trips:
            _, time_sec, _ = compute_route_metrics(trip, distance_matrix, demands, speed)
            total_time += time_sec
    return total_time

def is_feasible(routes, distance_matrix, demands, speed, vehicle_capacity, max_time):
    for vehicle_trips in routes:
        if not is_vehicle_feasible(vehicle_trips, distance_matrix, demands, speed, vehicle_capacity, max_time):
            return False
    return True

def split_demand(demand, nodes_all, target_points):
    split_demands = []
    split_nodes = ["运维处"]
    split_mapping = {0: ("运维处", 0)}
    split_indices = {0: 0}
    new_idx = 1
    for node in target_points:
        d = demand[node]
        if d == 0:
            continue
        orig_idx = nodes_all.index(node)
        if d > 0:
            while d > 0:
                demand_part = min(d, 20)
                split_nodes.append(f"{node}_{new_idx}")
                split_demands.append(demand_part)
                split_mapping[new_idx] = (node, len([n for n in split_nodes if n.startswith(node)]))
                split_indices[new_idx] = orig_idx
                d -= demand_part
                new_idx += 1
        else:
            while d < 0:
                demand_part = max(d, -20)
                split_nodes.append(f"{node}_{new_idx}")
                split_demands.append(demand_part)
                split_mapping[new_idx] = (node, len([n for n in split_nodes if n.startswith(node)]))
                split_indices[new_idx] = orig_idx
                d -= demand_part
                new_idx += 1
    split_demands.insert(0, 0)
    
    split_distance_matrix = np.zeros((len(split_nodes), len(split_nodes)))
    for i, u in enumerate(split_nodes):
        for j, v in enumerate(split_nodes):
            if i == j:
                split_distance_matrix[i][j] = 0
            else:
                u_idx = split_indices[i]
                v_idx = split_indices[j]
                split_distance_matrix[i][j] = distance_matrix[u_idx][v_idx]
    
    return split_nodes, split_demands, split_mapping, split_indices, split_distance_matrix

def generate_initial_solution(num_vehicles, split_nodes, split_demands, vehicle_capacity, split_distance_matrix, speed, max_time):
    routes = [[] for _ in range(num_vehicles)]
    unassigned = [(p, split_demands[p]) for p in range(1, len(split_nodes)) if split_demands[p] != 0]
    unassigned.sort(key=lambda x: abs(x[1]), reverse=True)
    
    current_vehicle = 0
    while unassigned:
        vehicle_trips = routes[current_vehicle]
        current_trip = [0]
        total_time = sum(compute_route_metrics(trip, split_distance_matrix, split_demands, speed)[1] for trip in vehicle_trips)
        
        i = 0
        while i < len(unassigned):
            point_idx, demand = unassigned[i]
            temp_trip = current_trip[:-1] + [point_idx] + [0]
            if is_trip_feasible(temp_trip, split_distance_matrix, split_demands, speed, vehicle_capacity, max_time):
                if total_time + compute_route_metrics(temp_trip, split_distance_matrix, split_demands, speed)[1] <= max_time:
                    current_trip = temp_trip
                    unassigned.pop(i)
                    continue
            i += 1
        
        if len(current_trip) == 2:
            current_demand = split_demands[current_trip[1]]
            if current_demand > 0:
                for i, (point_idx, demand) in enumerate(unassigned):
                    if demand < 0:
                        temp_trip = current_trip[:-1] + [point_idx] + [0]
                        if is_trip_feasible(temp_trip, split_distance_matrix, split_demands, speed, vehicle_capacity, max_time):
                            if total_time + compute_route_metrics(temp_trip, split_distance_matrix, split_demands, speed)[1] <= max_time:
                                current_trip = temp_trip
                                unassigned.pop(i)
                                break
            else:
                for i, (point_idx, demand) in enumerate(unassigned):
                    if demand > 0:
                        temp_trip = current_trip[:-1] + [point_idx] + [0]
                        if is_trip_feasible(temp_trip, split_distance_matrix, split_demands, speed, vehicle_capacity, max_time):
                            if total_time + compute_route_metrics(temp_trip, split_distance_matrix, split_demands, speed)[1] <= max_time:
                                current_trip = temp_trip
                                unassigned.pop(i)
                                break
        
        if len(current_trip) > 1:
            vehicle_trips.append(current_trip)
        
        current_vehicle = (current_vehicle + 1) % num_vehicles
    
    return routes

def swap_points(routes, split_demands):
    new_routes = [trips[:] for trips in routes]
    while True:
        v1 = random.randint(0, len(new_routes) - 1)
        v2 = random.randint(0, len(new_routes) - 1)
        if new_routes[v1] and new_routes[v2]:
            t1 = random.randint(0, len(new_routes[v1]) - 1)
            t2 = random.randint(0, len(new_routes[v2]) - 1)
            trip1, trip2 = new_routes[v1][t1], new_routes[v2][t2]
            if len(trip1) > 2 and len(trip2) > 2:
                p1 = random.randint(1, len(trip1) - 2)
                p2 = random.randint(1, len(trip2) - 2)
                new_routes[v1][t1][p1], new_routes[v2][t2][p2] = new_routes[v2][t2][p2], new_routes[v1][t1][p1]
                break
    return new_routes

def move_point(routes, split_demands):
    new_routes = [trips[:] for trips in routes]
    while True:
        v1 = random.randint(0, len(new_routes) - 1)
        v2 = random.randint(0, len(new_routes) - 1)
        if new_routes[v1] and v1 != v2:
            t1 = random.randint(0, len(new_routes[v1]) - 1)
            trip1 = new_routes[v1][t1]
            if len(trip1) > 2:
                p = random.randint(1, len(trip1) - 2)
                point = new_routes[v1][t1].pop(p)
                if not new_routes[v2]:
                    new_routes[v2].append([0, point, 0])
                else:
                    t2 = random.randint(0, len(new_routes[v2]) - 1)
                    insert_pos = random.randint(1, len(new_routes[v2][t2]) - 1)
                    new_routes[v2][t2].insert(insert_pos, point)
                break
    return new_routes

#模拟退火
def simulated_annealing(num_vehicles, split_nodes, split_demands, split_distance_matrix, speed, vehicle_capacity, max_time):
    current_routes = generate_initial_solution(num_vehicles, split_nodes, split_demands, vehicle_capacity, split_distance_matrix, speed, max_time)
    if not is_feasible(current_routes, split_distance_matrix, split_demands, speed, vehicle_capacity, max_time):
        return None
    current_cost = compute_total_time(current_routes, split_distance_matrix, split_demands, speed)
    best_routes = [[trip[:] for trip in trips] for trips in current_routes]
    best_cost = current_cost

    T = T0
    while T > T_min:
        for _ in range(iterations_per_temp):
            if random.random() < 0.5:
                new_routes = swap_points(current_routes, split_demands)
            else:
                new_routes = move_point(current_routes, split_demands)
            
            if not is_feasible(new_routes, split_distance_matrix, split_demands, speed, vehicle_capacity, max_time):
                continue
            
            new_cost = compute_total_time(new_routes, split_distance_matrix, split_demands, speed)
            cost_diff = new_cost - current_cost
            
            if cost_diff < 0 or random.random() < math.exp(-cost_diff / T):
                current_routes = [[trip[:] for trip in trips] for trips in new_routes]
                current_cost = new_cost
                if new_cost < best_cost:
                    best_routes = [[trip[:] for trip in trips] for trips in new_routes]
                    best_cost = new_cost
        
        T *= alpha
    
    return best_routes

for start_time, end_time in time_periods:
    period_label = f"{start_time[:5]}–{end_time[:5]}"
    print(f"\n=== 时间段 {period_label} ===")
    
    if start_time == "7:00:00":
        changes = table1.loc[end_time] - table1.loc[start_time]
        demand_series = changes * time_weight_full[end_time]
    else:
        changes = table1.loc[end_time] - table1.loc[start_time]
        demand_series = changes * time_weight_full[end_time]
    
    demand = {point: d for point, d in demand_series.items()}
    demand["运维处"] = 0
    
    print("需求：", demand)
    print("总盈余：", sum(d for d in demand.values() if d > 0))
    print("总亏空：", -sum(d for d in demand.values() if d < 0))
    
    split_nodes, split_demands, split_mapping, split_indices, split_distance_matrix = split_demand(demand, nodes_all, target_points)
    
    routes = simulated_annealing(
        num_vehicles, split_nodes, split_demands, split_distance_matrix, speed, vehicle_capacity, max_time
    )
        
    if routes:
        print(f"共享单车调度方案（{period_label}）：")
        total_load = 0
        formatted_routes = []
        for vehicle_id, vehicle_trips in enumerate(routes):
            vehicle_formatted_trips = []
            vehicle_load = 0
            vehicle_time = 0
            for trip_idx, trip in enumerate(vehicle_trips):
                formatted_trip = []
                for idx in trip:
                    orig_node, visit_num = split_mapping[idx]
                    formatted_trip.append(orig_node if idx == 0 else f"{orig_node} (第{visit_num}次)")
                distance, time_sec, load = compute_route_metrics(trip, split_distance_matrix, split_demands, speed)
                vehicle_load += load
                vehicle_time += time_sec
                if len(trip) > 2:
                    print(f"车辆{vehicle_id + 1} 行程{trip_idx + 1}：{' → '.join(formatted_trip)}，"
                          f"距离：{distance:.2f} 千米，时间：{time_sec:.0f} 秒，净搬运：{round(load+0.5,0)} 辆")
                vehicle_formatted_trips.append(formatted_trip)
            total_load += vehicle_load
            formatted_routes.append(vehicle_formatted_trips)
            print(f"车辆{vehicle_id + 1} 总时间：{vehicle_time:.0f} 秒，总净搬运：{round(vehicle_load,0)} 辆")
        print(f"总净搬运量：{round(total_load+0.5,0)} 辆")
    else:
        print("无解！请调整参数。")