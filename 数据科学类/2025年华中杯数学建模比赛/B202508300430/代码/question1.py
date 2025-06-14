import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

#提取各个点的像素坐标且模拟路径

# 加载地图图像
map_path = '/Users/keep-rational/Desktop/华中杯/B题：校园共享单车的调度与维护问题/附件/campus_map.jpg'
map_image = cv2.imread(map_path)

# 交互式标注地点
# 每次用户左键点击地图，程序会在点击位置画一个红点，直到标注完所有 15 个地点
points = []
labels = ['东门','南门','北门','一食堂','二食堂','三食堂','梅苑 1 栋','菊苑 1 栋','教学 2 楼','教学 4 楼','计算机学院','工程中心','网球场','体育馆','校医院','共享单车运维处']
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < len(labels):
        points.append((x, y))
        cv2.circle(map_image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Map', map_image)

cv2.imshow('Map', map_image)
cv2.setMouseCallback('Map', click_event) 
cv2.waitKey(0) #无限等待
cv2.destroyAllWindows()

# 保存地点坐标
points_df = pd.DataFrame(points, columns=['x', 'y'], index=labels)
points_df.to_csv('points.csv')
with open('points.json', 'w', encoding='utf-8') as f:
    json.dump({label: list(point) for label, point in zip(labels, points)}, f, ensure_ascii=False)

# 提取黄色真实路径
#  半自动标注黄色路径（使用颜色阈值初步提取）
hsv = cv2.cvtColor(map_image, cv2.COLOR_BGR2HSV)
yellow_lower = np.array([20, 100, 100])
yellow_upper = np.array([30, 255, 255])
initial_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

mask = initial_mask.copy()
def draw_mask(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(mask, (x, y), 10, 255, -1)
        cv2.imshow('Mask', mask)
cv2.imshow('Map', map_image)
cv2.imshow('Mask', mask)
cv2.setMouseCallback('Map', draw_mask)
cv2.waitKey(0)
cv2.imwrite('initial_mask.png', mask)

#计算单位像素距离

map_path = '/Users/keep-rational/Desktop/华中杯/B题：校园共享单车的调度与维护问题/附件/campus_map.jpg'
map_image = cv2.imread(map_path)
points = []

# 鼠标点击事件处理函数
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
        points.append((x, y))
        cv2.circle(map_image, (x, y), 5, (0, 0, 255), -1)  # 画红点
        if len(points) == 2:
            # 绘制比例尺线段
            cv2.line(map_image, points[0], points[1], (255, 0, 0), 2)
            cv2.imshow('Map', map_image)
            # 计算并显示像素距离
            x1, y1 = points[0]
            x2, y2 = points[1]
            pixel_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            print(f"比例尺像素距离为: {pixel_distance:.2f} 像素")
            print(f"单位像素代表的实际距离为: {2000/pixel_distance:.2f} 米/像素") 

# 显示图像并绑定鼠标回调
cv2.imshow('Map', map_image)
cv2.setMouseCallback('Map', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

#单位像素代表的实际距离为: 2.00 米/像素

# 像素坐标
coordinates_pixels = {
    "东门": (732, 564), "南门": (541, 1115), "北门": (477, 180),
    "一食堂": (317, 681), "二食堂": (214, 826), "三食堂": (432, 908),
    "梅苑1栋": (303, 977), "菊苑1栋": (226, 494), "教学2楼": (462, 638),
    "教学4楼": (553, 790), "计算机学院": (485, 397), "工程中心": (662, 356),
    "网球场": (352, 460), "体育馆": (221, 299), "校医院": (201, 949),
    "共享单车运维处": (725,179) 
}

# 比例因子
scale_factor = 2 

# 转换为实际坐标（米），翻转 Y 轴
coordinates_meters = {}
for node, (x, y) in coordinates_pixels.items():
    x_meters = x * scale_factor
    y_meters = -y * scale_factor  # 翻转 Y 轴，Y 值越大越靠下
    coordinates_meters[node] = (x_meters, y_meters)

# 打印实际坐标
print("实际坐标（米）：")
for node, (x, y) in coordinates_meters.items():
    print(f"{node}: ({x:.2f}, {y:.2f})")

# 保存实际坐标为 CSV
coords_list = [(node, x, y) for node, (x, y) in coordinates_meters.items()]
coords_df = pd.DataFrame(coords_list, columns=['地点', 'x', 'y'])
coords_df.set_index('地点', inplace=True)
coords_df.to_csv('actual_points.csv')
print("实际坐标已保存为 actual_points.csv")
