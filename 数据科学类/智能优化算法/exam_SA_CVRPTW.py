import pandas as pd
import math
import random
import numpy as np
import copy
import xlsxwriter
import matplotlib.pyplot as plt
import time

class Sol():
    def __init__(self):
        self.nodes_seq=None
        self.obj=None
        self.routes=None
        self.Route_dis = None   #每条车辆路径对应的行驶里程
class Node():
    def __init__(self):
        self.id=0
        self.name=''
        self.seq_no=0
        self.x_coord=0
        self.y_coord=0
        self.demand=0
        self.et = 0            #节点的最早开始服务时间
        self.lt = 0            #节点的最晚开始服务时间
        self.st = 0            #节点的服务持续时间
class Model():
    def __init__(self):
        self.best_sol=None
        self.customer=[]
        self.node_seq_no_list=[]
        self.depot=None
        self.number_of_nodes=0
        self.vehicle_cap=0
        self.v = 1

#函数：读取数据文件
def readXlsxFile(filepath,model):
    node_seq_no = -1 #the depot node seq_no is -1,and demand node seq_no is 0,1,2,...
    df = pd.read_excel(filepath)
    for i in range(df.shape[0]):
        node=Node()
        node.id=node_seq_no
        node.seq_no=node_seq_no
        node.x_coord= df['x_coord'][i]
        node.y_coord= df['y_coord'][i]
        node.demand=df['demand'][i]
        node.et = df['et'][i]
        node.lt = df['lt'][i]
        node.st = df['st'][i]
        if df['demand'][i] == 0:
            model.depot=node
        else:
            model.customer.append(node)
            model.node_seq_no_list.append(node_seq_no)
        try:
            node.name=df['name'][i]
        except:
            pass
        try:
            node.id=df['id'][i]
        except:
            pass
        node_seq_no=node_seq_no+1
    model.number_of_nodes=len(model.customer)

#函数：随机产生初始解
def genInitialSol(node_seq):
    node_seq=copy.deepcopy(node_seq)
    #random.seed(0)
    random.shuffle(node_seq)
    return node_seq

#函数：构建3类邻域算子
def createActions(n):
    action_list=[]
    nswap=n//2
    # 邻域算子1：单点交叉
    for i in range(nswap):
        action_list.append([1, i, i + nswap])
    # 邻域算子2：两点交叉
    for i in range(0, nswap, 2):
        action_list.append([2, i, i + nswap])
    # 邻域算子3：逆序翻转(步长为4)
    for i in range(0, n, 4):
        action_list.append([3, i, i + 3])
    return action_list

#函数：执行邻域算子
def doACtion(nodes_seq,action):
    nodes_seq=copy.deepcopy(nodes_seq)
    if action[0]==1:
        #执行邻域算子1
        index_1=action[1]
        index_2=action[2]
        temporary=nodes_seq[index_1]
        nodes_seq[index_1]=nodes_seq[index_2]
        nodes_seq[index_2]=temporary
        return nodes_seq
    elif action[0]==2:
        #执行邻域算子2
        index_1 = action[1]
        index_2 = action[2]
        temporary=[nodes_seq[index_1],nodes_seq[index_1+1]]
        nodes_seq[index_1]=nodes_seq[index_2]
        nodes_seq[index_1+1]=nodes_seq[index_2+1]
        nodes_seq[index_2]=temporary[0]
        nodes_seq[index_2+1]=temporary[1]
        return nodes_seq
    elif action[0]==3:
        #执行邻域算子3
        index_1=action[1]
        index_2=action[2]
        nodes_seq[index_1:index_2+1]=list(reversed(nodes_seq[index_1:index_2+1]))
        return nodes_seq
    
#函数：按容量和时间窗约束切分车辆路径
def splitRoutes(nodes_seq,model):
    routes = []
    current_route = []
    current_load = 0
    current_time = 0  # 离开仓库的时间初始为0
    i = 0
    n = len(nodes_seq)
    
    while i < n:
        c = nodes_seq[i]
        if not current_route:
            # 尝试作为路径的第一个客户加入
            travel_time = math.sqrt((model.depot.x_coord - model.customer[c].x_coord)**2 + (model.depot.y_coord - model.customer[c].y_coord)**2)/model.v
            arrival = max(current_time + travel_time, model.customer[c].et)
            new_load = model.customer[c].demand
            current_route.append(c)
            current_load = new_load
            current_time = arrival
            i += 1
        else:
            # 尝试将客户添加到当前路径的末尾
            prec = current_route[-1]
            travel_time = math.sqrt((model.customer[prec].x_coord - model.customer[c].x_coord)**2 + (model.customer[prec].y_coord - model.customer[c].y_coord)**2)/model.v
            arrival = max(current_time + model.customer[prec].st + travel_time, model.customer[c].et)
            new_load = current_load + model.customer[c].demand          
            if new_load <= model.vehicle_cap and arrival <= model.customer[c].lt:
                current_route.append(c)
                current_load = new_load
                current_time = arrival
                i += 1
            else:
                # 无法加入，保存当前路径，重置状态
                routes.append(current_route)
                current_route = []
                current_load = 0
                current_time = 0
                # i不递增，下一轮继续处理当前客户
    # 添加最后一个路径
    if current_route:
        routes.append(current_route)
    return routes

#函数：计算单条路径距离成本
def calDistance(route,model):
    distance=0
    depot=model.depot
    for i in range(len(route)-1):
        from_node=model.customer[route[i]]
        to_node=model.customer[route[i+1]]
        distance+=math.sqrt((from_node.x_coord-to_node.x_coord)**2+(from_node.y_coord-to_node.y_coord)**2)
    first_node=model.customer[route[0]]
    last_node=model.customer[route[-1]]
    distance+=math.sqrt((depot.x_coord-first_node.x_coord)**2+(depot.y_coord-first_node.y_coord)**2)
    distance+=math.sqrt((depot.x_coord-last_node.x_coord)**2+(depot.y_coord - last_node.y_coord)**2)
    return distance

#函数：计算整个车辆路径方案成本
def calObj(nodes_seq,model):
    vehicle_routes = splitRoutes(nodes_seq, model)
    Route_dis = []
    distance=0
    for route in vehicle_routes:
        dis = calDistance(route,model)
        distance+=dis
        Route_dis.append(dis)
    return distance,vehicle_routes,Route_dis
    
#函数：绘制目标函数值迭代图像
def plotObj(obj_list):
    plt.rcParams['font.sans-serif'] = ['SimHei'] #show chinese
    plt.rcParams['axes.unicode_minus'] = False   # Show minus sign
    plt.plot(np.arange(1,len(obj_list)+1),obj_list)
    plt.xlabel('Iterations')
    plt.ylabel('Obj Value')
    plt.grid()
    plt.xlim(1,len(obj_list)+1)
    plt.show()
    
#函数：输出求解结果
def outPut(model):
    work = xlsxwriter.Workbook('result.xlsx')
    worksheet = work.add_worksheet()
    worksheet.write(0, 0, 'opt_type')
    worksheet.write(1, 0, 'obj')
    if model.opt_type == 0:
        worksheet.write(0, 1, 'number of vehicles')
    else:
        worksheet.write(0, 1, 'drive distance of vehicles')
    worksheet.write(1, 1, model.best_sol.obj)
    for row, route in enumerate(model.best_sol.routes):
        worksheet.write(row + 2, 0, 'v' + str(row + 1))
        r = [str(i) for i in route]
        worksheet.write(row + 2, 1, '-'.join(r))
    work.close()

#函数：求解结果输出
def output(all_best_obj,all_best_Route,all_Route_dis):
    excel = xlsxwriter.Workbook('result.xlsx') #生成一个新的Excel文件名称叫result，存放在Python代码同路径
    excelsheet = excel.add_worksheet('sheet0') #为excel创建sheet0
    excelsheet.write(0,0,'best_cost')
    excelsheet.write(1,0,min(all_best_obj))             #表格第2行第1列写入最优结果
    excelsheet.write(0,1,'worst_cost')
    excelsheet.write(1,1,max(all_best_obj))            #表格第2行第2列写入最差结果
    excelsheet.write(0,2,'aver_cost')
    excelsheet.write(1,2,sum(all_best_obj)/len(all_best_obj))   #表格第2行第3列写入平均结果
    #写入每一次运行的求解结果
    for i in range(len(all_best_obj)):
        excelsheet.write(0,i+3,f'cost{i+1}')
        excelsheet.write(1,i+3,all_best_obj[i])
    #写入每一次运行的求解方案
    for i in range(len(all_best_obj)):
        excelsheet = excel.add_worksheet(f'sheet{i+1}') #为excel创建sheet1
        Route = all_best_Route[i]
        excelsheet.write(0,0,f'cost{i+1}')
        excelsheet.write(0,1,all_best_obj[i])
        for r in range(len(Route)):
            excelsheet.write(r+1,0,f'v{r+1}')      #从第2行开始，依次在第1列中写入车辆编号
            excelsheet.write(r+1,1,str(Route[r]))      #从第2行开始，依次在第2列中写入车辆路径
            excelsheet.write(r+1,2,str(all_Route_dis[i][r]))   #从第2行开始，依次在第3列中写入单条里程
    excel.close()
    
#函数：运行SA
def run(filepath,T0,Tf,detaT,v_cap,time_limit):
    """
    T0: 初始温度
    Tf: 终止温度
    deltaT: 降温率
    v_cap:车辆载重
    """
    #实例化算例
    model=Model()
    model.vehicle_cap=v_cap
    readXlsxFile(filepath,model)
    #构建邻域算子列表
    action_list=createActions(model.number_of_nodes)
    #生成初始解
    history_best_obj=[]
    sol=Sol()
    sol.nodes_seq=genInitialSol(model.node_seq_no_list)
    sol.obj,sol.routes,sol.Route_dis=calObj(sol.nodes_seq,model)
    model.best_sol=copy.deepcopy(sol)
    history_best_obj.append(sol.obj)
    #设定当前温度为初始温度
    Tk=T0
    nTk=len(action_list)
    start_time = time.time()   #定义算法起始时间
    #开始迭代求解
    while Tk>=Tf and time.time() - start_time < time_limit:
        for i in range(nTk):
            new_sol = Sol()
            new_sol.nodes_seq = doACtion(sol.nodes_seq, action_list[i])
            new_sol.obj, new_sol.routes, new_sol.Route_dis = calObj(new_sol.nodes_seq, model)
            deta_f=new_sol.obj-sol.obj
            #根据模拟退火接受准则,确定是否接受新解
            if deta_f<0 or math.exp(-deta_f/Tk)>random.random():
                sol=copy.deepcopy(new_sol)
            if sol.obj<model.best_sol.obj:
                model.best_sol=copy.deepcopy(sol)
        #更新温度
        if detaT<1:
            Tk=Tk*detaT
        else:
            Tk = Tk - detaT
        #更新历史最优解
        history_best_obj.append(model.best_sol.obj)
        print("当前温度：%s，局部最优解:%s 全局最优解: %s" % (round(Tk,2),round(sol.obj,2),round(model.best_sol.obj,2)))
    #绘制迭代图
    plotObj(history_best_obj)
    #输出结果
    #outPut(model)
    return model
    
if __name__=='__main__':
    file=r'/Users/keep-rational/Desktop/算法排名作业(VRPTW)-200客户/代码/Instances/RC2_2_10.xlsx'
    run_time = 5       #算法运行次数
    i = 0
    all_best_obj = []
    all_best_Route = []
    all_Route_dis = []
    while i < run_time:
        print(f"*====================第{i+1}次运行====================*")
        run_start = time.perf_counter()
        print("start time：",run_start)
        model = run(filepath=file,T0=8000,Tf=0.001,detaT=0.95,v_cap=1000,time_limit=180)
        all_best_obj.append(model.best_sol.obj)
        all_best_Route.append(model.best_sol.routes)
        all_Route_dis.append(model.best_sol.Route_dis)
        print("最短里程：",model.best_sol.obj)
        print("最短车辆路径方案：",model.best_sol.routes)
        run_end = time.perf_counter()
        print("end time：",run_start)
        print(f'run time={run_end-run_start}秒')
        i += 1 
    output(all_best_obj,all_best_Route,all_Route_dis)