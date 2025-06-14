import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 更新坐标，添加新点位
coordinates = {
    "东门": (1464, -1128), "南门": (1082, -2230), "北门": (954, -360),
    "一食堂": (634, -1362), "二食堂": (428, -1652), "三食堂": (864, -1816),
    "梅苑1栋": (606, -1954), "菊苑1栋": (452, -988), "教学2楼": (924, -1276),
    "教学4楼": (1106, -1580), "计算机学院": (970, -794), "工程中心": (1324, -712),
    "网球场": (704, -920), "体育馆": (442, -598), "校医院": (402, -1898),
    "运维处": (1450, -358),
    "教学食堂结合新区": (798.23, -1494.76),  # 新增点位
    "一田右侧": (668.72, -601.53)  # 新增点位
}

# 更新目标点位列表
target_points = ["东门", "南门", "北门", "一食堂", "二食堂", "三食堂",
                 "梅苑1栋", "菊苑1栋", "教学2楼", "教学4楼", "计算机学院",
                 "工程中心", "网球场", "体育馆", "校医院",
                 "教学食堂结合新区", "一田右侧"]  # 添加新点位

# 更新 table1，添加新点位的车辆分布（假设数据）
table1_data = {
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
    "校医院": [5, 29, 6, 38, 6, 7, 11],
    "教学食堂结合新区": [100, 0, 140, 0, 60, 50, 120],  # 假设数据，类似三食堂
    "一田右侧": [3, 5, 5, 10, 15, 5, 10]  # 假设数据，类似体育馆
}

table1 = pd.DataFrame(
    table1_data,
    index=["7:00:00", "9:00:00", "12:00:00", "14:00:00", "18:00:00", "21:00:00", "23:00:00"]
)

# 更新节点列表和距离矩阵
nodes_all = ["运维处"] + target_points
G = nx.Graph()
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

# 参数
speed = 25  # km/h
vehicle_capacity = 20  # 最大运输 20 辆
handling_time_per_bike = 60  # 每辆车搬运时间 1 分钟 = 60 秒
fault_rate = 0.06  # 故障率 6%
target_fault_ratio = 0.02  # 目标故障比例 2%

# 计算故障车辆数量
def compute_faulty_bikes(time):
    total_bikes = table1.loc[time].sum()
    total_faulty = int(total_bikes * fault_rate)
    faulty_bikes = {}
    for point in target_points:
        bikes = table1.loc[time, point]
        faulty = int(bikes / total_bikes * total_faulty) if total_bikes > 0 else 0
        faulty_bikes[point] = faulty
    # 调整总数
    current_total = sum(faulty_bikes.values())
    if current_total < total_faulty:
        remaining = total_faulty - current_total
        sorted_points = sorted(faulty_bikes.items(), key=lambda x: x[1], reverse=True)
        i = 0
        while remaining > 0:
            point, _ = sorted_points[i % len(sorted_points)]
            faulty_bikes[point] += 1
            remaining -= 1
            i += 1
    return faulty_bikes, total_bikes

# 贪心算法生成巡检路线
def generate_inspection_route(faulty_bikes, distance_matrix, speed, vehicle_capacity, handling_time_per_bike, nodes_all):
    routes = []
    total_time = 0
    total_travel_time = 0
    total_handling_time = 0
    total_transported = 0
    remaining_faulty = faulty_bikes.copy()
    
    while sum(remaining_faulty.values()) > 0:
        current_route = ["运维处"]
        current_capacity = vehicle_capacity
        current_time = 0
        current_travel_time = 0
        current_handling_time = 0
        current_pos = 0  # 运维处索引
        route_details = []
        
        while current_capacity > 0 and sum(remaining_faulty.values()) > 0:
            available_points = [(nodes_all.index(point), count) for point, count in remaining_faulty.items() if count > 0]
            if not available_points:
                break
            available_points.sort(key=lambda x: x[1], reverse=True)
            next_point_idx, fault_count = available_points[0]
            next_point = nodes_all[next_point_idx]
            
            distance = distance_matrix[current_pos][next_point_idx]
            travel_time = (distance / speed) * 3600
            current_travel_time += travel_time
            
            bikes_to_transport = min(current_capacity, fault_count)
            handling_time = bikes_to_transport * handling_time_per_bike
            current_handling_time += handling_time
            
            current_time += travel_time + handling_time
            current_capacity -= bikes_to_transport
            total_transported += bikes_to_transport
            remaining_faulty[next_point] -= bikes_to_transport
            
            route_details.append({
                "from": nodes_all[current_pos],
                "to": next_point,
                "travel_time": travel_time,
                "handling_time": handling_time,
                "bikes": bikes_to_transport
            })
            
            current_route.append(next_point)
            current_pos = next_point_idx
        
        distance_back = distance_matrix[current_pos][0]
        travel_time_back = (distance_back / speed) * 3600
        current_travel_time += travel_time_back
        current_time += travel_time_back
        current_route.append("运维处")
        route_details.append({
            "from": nodes_all[current_pos],
            "to": "运维处",
            "travel_time": travel_time_back,
            "handling_time": 0,
            "bikes": 0
        })
        
        routes.append((current_route, current_time, vehicle_capacity - current_capacity, current_travel_time, current_handling_time, route_details))
        total_time += current_time
        total_travel_time += current_travel_time
        total_handling_time += current_handling_time
    
    return routes, total_time, total_travel_time, total_handling_time, total_transported

# 计算 12:00:00 时间段的巡检路线
time = "12:00:00"
faulty_bikes, total_bikes = compute_faulty_bikes(time)
routes, total_time, total_travel_time, total_handling_time, total_transported = generate_inspection_route(
    faulty_bikes, distance_matrix, speed, vehicle_capacity, handling_time_per_bike, nodes_all
)

# 打印巡检路线
print(f"\n=== 巡检时间 {time} ===")
total_faulty = sum(faulty_bikes.values())
fault_ratio = total_faulty / total_bikes if total_bikes > 0 else 0
print(f"当前故障车辆总数：{total_faulty} 辆，故障比例：{fault_ratio:.2%}")

print(f"巡检路线：")
for i, (route, time_sec, bikes, travel_time, handling_time, details) in enumerate(routes):
    print(f"行程 {i+1}：{' → '.join(route)}")
    print(f"  行驶时间：{travel_time:.0f} 秒，搬运时间：{handling_time:.0f} 秒，总时间：{time_sec:.0f} 秒，搬运：{bikes} 辆")
    print("  详细时间：")
    for detail in details:
        if detail["bikes"] > 0:
            print(f"    {detail['from']} → {detail['to']}：行驶 {detail['travel_time']:.0f} 秒，搬运 {detail['bikes']} 辆（{detail['handling_time']:.0f} 秒）")
        else:
            print(f"    {detail['from']} → {detail['to']}：行驶 {detail['travel_time']:.0f} 秒")

print(f"总行驶时间：{total_travel_time:.0f} 秒，总搬运时间：{total_handling_time:.0f} 秒，总时间：{total_time:.0f} 秒，总搬运：{total_transported} 辆")
fault_ratio_after = (total_faulty - total_transported) / total_bikes if total_bikes > 0 else 0
print(f"巡检后故障比例：{fault_ratio_after:.2%}")

# 绘制调度路线图
plt.figure(figsize=(12, 10))
for i, (route, _, _, _, _, _) in enumerate(routes):
    x = [coordinates[point][0] for point in route]
    y = [coordinates[point][1] for point in route]
    plt.plot(x, y, marker='o', linestyle='-', color=['r', 'g', 'b'][i % 3], label=f'行程 {i+1}', linewidth=2, markersize=8)

for node, coord in coordinates.items():
    plt.annotate(node, textcoords="offset points", xy=coord, xytext=(0, 10), ha='center', fontsize=10, color='black')

plt.title(f"鲁迪巡检路线图", fontsize=16, pad=20, color='#333333')
plt.xlabel("X坐标 (米)", fontsize=12)
plt.ylabel("Y坐标 (米)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.tight_layout()
plt.savefig(f"鲁迪巡检路线图.png", dpi=300, bbox_inches="tight")
plt.close()