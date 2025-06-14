# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 16:10:14 2025

@author: Administrator
"""
import numpy as np
import pandas as pd                 #提供DataFrame，是一种二维表格型数据
import matplotlib.pyplot as plt     #画图包
import xlsxwriter                   #读取Excel文件包
import math
import copy
import random
import time
##切割时可行//
##加惩罚值

#Sol类，表示一个可行解，等于一条染色体,存储在model.pop中
class Sol():
    def __init__(self):
        self.chrom = None       #染色体，对应于一个完整解方案
        self.Route = None       #车辆路径方案
        self.Route_dis = None   #每条车辆路径对应的行驶里程
        self.obj = None         #目标函数值
        self.fit = None         #适应度值
        
#Node类，表示一个节点
class Node():
    def __init__(self):
        self.id = 0            #节点的序号
        self.name = ''         #节点的名称（客户点:C1，C2，...）
        self.x_coord = 0       #节点的x坐标
        self.y_coord = 0       #节点的y坐标
        self.demand = 0        #节点的载重需求
        self.et = 0            #节点的最早开始服务时间
        self.lt = 0            #节点的最晚开始服务时间
        self.st = 0            #节点的服务持续时间
        
#Model类，存储算法参数
class Model():
    def __init__(self):
        self.best_sol = None   #全局最优解,值类型为Sol()
        self.pop = []          #种群，值类型为Sol()
        self.depot = None      #仓库点，值类型为Node()
        self.customer = []     #客户点集合，值类型为Node(),编号0-99
        self.number = 0        #客户点数量=染色体长度
        self.SeqID = []        #客户点id映射集合，编号0-99
        self.opt_type = 1      #优化目标类型，0：最小车辆数，1：最小行驶距离
        self.capacity = 0      #车辆最大载重，初始化为0
        self.v = 1              #车辆行驶速度，默认为1
        self.pc = 0.8          #交叉率
        self.pm = 0.2          #变异率
        #self.n_select = 50    #采取精英保留策略时，优良个体选择数量
        self.popsize = 100     #种群规模
        self.fleet = None      #车辆集合，编号续客户点
        self.p_d = 10          #载重约束违背惩罚系数  
        self.p_t = 100          #时间约束违背惩罚系数  
#函数：读取数据文件
def readExcel(filepath,model,capacity):
    model.capacity = capacity     #输入车辆载重
    DF = pd.read_excel(filepath)  #读取文件路径里的文件，生成二维矩阵数据
    id_NO = -1                    #客户点id序号，初始赋值为-1（仓库点）
    for i in range(DF.shape[0]):  #shape[0]返回二维数组的行数，shape[1]返回列数
        node = Node()             #每行数据都存储为一个节点
        node.id = id_NO           #读取节点的id序号
        node.name = f'C{i-1}'     #节点名称，仓库为C-1，其余客户点从C0、C1...C99
        node.x_coord = DF['x_coord'][i]    #读取节点的x坐标值
        node.y_coord = DF['y_coord'][i]    #读取节点的y坐标值
        node.demand = DF['demand'][i]      #读取节点的载重需求量
        node.et = DF['et'][i]
        node.lt = DF['lt'][i]
        node.st = DF['st'][i]
        if i == 0:                         #若该节点为读取的第一个节点
            model.depot = node             #那么该节点是仓库
        else:
            model.customer.append(node)    #否则该点是客户点
            model.SeqID.append(id_NO)      #存储客户点id映射集合[0,1,2,..,99]
        id_NO += 1
    model.number = len(model.customer)     #读取总的客户点数量
    
#函数：初始种群生成
def initialpop(model):
    temp_chrom = copy.deepcopy(model.SeqID)     #初始模板染色体为[0,1,2,..,99]
    for i in range(model.popsize):
        random.shuffle(temp_chrom)              #打乱初始染色体顺序，得到一个随机的染色体
        sol = Sol()
        sol.chrom = copy.deepcopy(temp_chrom)   #将该条染色体存储进sol中
        model.pop.append(sol)                   #将该条染色体对应的sol存入种群列表中

#函数：解码，按容量和时间窗约束切分车辆路径
def split_routes(chrom, model):
    routes = []
    current_route = []
    current_load = 0
    current_time = 0  # 离开仓库的时间初始为0
    i = 0
    n = len(chrom)
    
    while i < n:
        c = chrom[i]
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
            if new_load <= model.capacity and arrival <= model.customer[c].lt:
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

#函数：单条车辆路径里程计算
def caldistance(route,model):
    distance = 0                    #初始化单条路径里程为0
    depot = model.depot
    for i in range(len(route)-1):   #计算该路径上所有客户点间的行驶里程
        distance += math.sqrt((model.customer[route[i]].x_coord - model.customer[route[i+1]].x_coord)**2 + 
                              (model.customer[route[i]].y_coord - model.customer[route[i+1]].y_coord)**2)
    F_customer = model.customer[route[0]]    #路径中的第一个客户点
    L_customer = model.customer[route[-1]]   #路径中的最后一个客户点
    distance += math.sqrt((depot.x_coord - F_customer.x_coord)**2 + (depot.y_coord - F_customer.y_coord)**2)
    distance += math.sqrt((depot.x_coord - L_customer.x_coord)**2 + (depot.y_coord - L_customer.y_coord)**2)
    return distance    
    
#函数：适应度值计算：目标函数求行驶里程dis最小，有两种表示适应度值的方法，(1)fit=1/dis;(2)fit=dismax - dis，选(2)
def calFit(model):
    objMAX = -float('inf')
    best_sol = Sol()               #存储当前种群的最优染色体，初始化为空
    best_sol.obj = float('inf')    #初始化当前种群最优染色体目标函数值为无穷大
    for sol in model.pop:          #对种群中的每一条染色体进行操作
        Route = split_routes(sol.chrom, model)  #解码该条染色体，得到车辆路径方案
        Route_dis = []                    #存储该车辆路径方案中每条路径的里程值
        for route in Route:
            dis = caldistance(route,model) #计算得到dis
            Route_dis.append(dis)

        sol.Route = Route         #存储该染色体的车辆路径方案
        sol.Route_dis = Route_dis #存储该染色体每条路径的里程
        sol.obj = sum(Route_dis) #计算目标函数值
        if sol.obj > objMAX:
            objMAX = sol.obj      #更新当前种群最大目标值
        if sol.obj < best_sol.obj:
            best_sol = copy.deepcopy(sol)   #更新当前种群的最优目标值
    
    for sol in model.pop:
        sol.fit = objMAX - sol.obj              #计算当前种群每条染色体的适应度值
    if best_sol.obj < model.best_sol.obj:
        model.best_sol = copy.deepcopy(best_sol) #若当前种群最优目标值优于全局最优目标值，更新全局最优解为当前种群最优染色体           
    
#函数：选择算子，二元锦标赛方法,对比轮盘赌，随机性更大
def select(model):
    temp_pop = copy.deepcopy(model.pop)              #temp_pop充当父代种群
    model.pop = []                                   #初始化子代种群为空
    for i in range(model.popsize):                            #进行一百次选择
        f1_index = random.randint(0,len(temp_pop)-1)          #每次任选2个，选择其中最好的一个
        f2_index = random.randint(0,len(temp_pop)-1)
        f1_fit = temp_pop[f1_index].fit
        f2_fit = temp_pop[f2_index].fit
        if f1_fit < f2_fit:
            model.pop.append(temp_pop[f2_index])
        else:
            model.pop.append(temp_pop[f1_index])
    
#函数：OX交叉算子（example），保留下来的个体随机性更大
def cross(model):
    temp_pop = copy.deepcopy(model.pop)  #进行了选择但还未进行交叉的种群，父代种群
    model.pop = []                       #初始化交叉之后的子代种群为空
    while True:
        father_index = random.randint(0,model.popsize-1)  #随机选出父代1个体索引
        mother_index = random.randint(0,model.popsize-1)  #随机选出父代2个体索引
        if father_index != mother_index:                  #确保两个父代不为同一条染色体
            father = copy.deepcopy(temp_pop[father_index])  #父代1，值类型sol
            mother = copy.deepcopy(temp_pop[mother_index])  #父代2，值类型sol
            if random.random() < model.pc:   #一定概率发生交叉
                cpoint1 = int(random.randint(0,model.number-1))       #交叉点1位置索引
                cpoint2 = int(random.randint(cpoint1,model.number-1)) #交叉点2位置索引
                new_father_f = []                                     #父代1前段基因串
                new_father_m = father.chrom[cpoint1:cpoint2+1]        #父代1交叉段基因串
                new_father_b = []                                     #父代1后段基因串
                new_mother_f = []                                     #父代2前段基因串
                new_mother_m = mother.chrom[cpoint1:cpoint2+1]        #父代2交叉段基因串
                new_mother_b = []                                     #父代2后段基因串
                for i in range(model.number):
                    if len(new_father_f) < cpoint1:   #父代1前串基因还未填充完
                        if mother.chrom[i] not in new_father_m:   #将不在父代1中段的父代2基因
                            new_father_f.append(mother.chrom[i])  #添加至父代1前段
                    else:                             #父代1前串基因填充完，填充父代1后串基因
                        if mother.chrom[i] not in new_father_m:   #将不在父代1中段的父代2基因
                            new_father_b.append(mother.chrom[i])  #添加至父代2后段
                for i in range(model.number):                     #对父代2同样的操作
                    if len(new_mother_f) < cpoint1:
                        if father.chrom[i] not in new_mother_m:
                            new_mother_f.append(father.chrom[i])
                    else:
                        if father.chrom[i] not in new_mother_m:
                            new_mother_b.append(father.chrom[i])
                new_father = new_father_f + new_father_m + new_father_b #得到交叉后新的父代1染色体
                father.chrom = copy.deepcopy(new_father)
                new_mother = new_mother_f + new_mother_m + new_mother_b #得到交叉后新的父代2染色体
                mother.chrom =copy.deepcopy(new_mother)
                model.pop.append(copy.deepcopy(father))  #将交叉后的父代1加入子代种群
                model.pop.append(copy.deepcopy(mother))  #将交叉后的父代2加入子代种群
            else:                                        #若未发生交叉，直接加入子代种群
                model.pop.append(copy.deepcopy(father))
                model.pop.append(copy.deepcopy(mother))
            if len(model.pop) == model.popsize:
                break           
                     
#函数：变异算子，二元突变,随机性更高
def mutation(model):
    temp_pop = copy.deepcopy(model.pop)  #进行了交叉但还未进行变异的种群，父代种群
    model.pop = []                       #初始化变异之后的子代种群为空
    while True:
        father_index = int(random.randint(0,model.popsize-1)) #随机选出一条染色体做父代
        father = copy.deepcopy(temp_pop[father_index])
        mpoint1 = random.randint(0,model.number-1)  #随机选出变异点1
        mpoint2 = random.randint(0,model.number-1)  #随机选出变异点2
        if mpoint1 != mpoint2:   #要求两个点不能相同
            if random.random() < model.pm:   #一定概率产生变异
                point1 = father.chrom[mpoint1]
                father.chrom[mpoint1] = father.chrom[mpoint2] #将变异点1突变为变异点2
                father.chrom[mpoint2] = point1                #将变异点2突变为变异点1
                model.pop.append(copy.deepcopy(father))       #将变异后的染色体添加进子代种群
            else:
                model.pop.append(copy.deepcopy(father))
            if len(model.pop) == model.popsize:
                break   

#绘图函数：绘制收敛曲线图
def plot_obj(objlist):          #传入的是每一代的最优目标函数值
    plt.figure()
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 更改运行配置参数中的字体（font）为黑体（SimHei），用来正常显示中文,这行代码可要可不要
    plt.rcParams['axes.unicode_minus'] = False   #运行配置参数总的轴（axes）正常显示正负号（minus）
    plt.plot(np.arange(1,len(objlist)+1), objlist)  #画图，x坐标为[1,len(objlist)+1],y坐标为objlist
    plt.xlabel('迭代次数')
    plt.ylabel('最优目标函数值')
    plt.grid()                  #显示网格线，1=True=默认=显示，0=False=不显示
    plt.xlim(1,len(objlist)+1)  #显示的是x轴的作图范围

#绘图函数：绘制车辆行驶路径图
def plot_route(model):
    plt.figure()
    for route in model.best_sol.Route:   #对每条车辆路径进行绘制
        x_coord = [model.depot.x_coord]  #生成绘图x坐标列表，先把仓库加进去，作为起点
        y_coord = [model.depot.y_coord]  #生成绘图y坐标列表，先把仓库加进去，作为起点
        for i in route:                  #将路径中每个客户点坐标加进去
            x_coord.append(model.customer[i].x_coord)
            y_coord.append(model.customer[i].y_coord)
            plt.text(model.customer[i].x_coord,model.customer[i].y_coord,model.customer[i].name,fontsize=5) #先在图上画出每个客户点的名称
        x_coord.append(model.depot.x_coord)       #最后再次加入仓库，作为终点
        y_coord.append(model.depot.y_coord) #最后再次加入仓库，作为终点
        plt.grid()              #显示网格线，1=True=默认=显示，0=False=不显示
        plt.plot(x_coord,y_coord,'b:',linewidth=0.5,marker='o',markersize=2)  #设置每条路径的绘图参数，线型及颜色：蓝色、点线(b:)；点型：原点，大小为5
    plt.plot(model.depot.x_coord,model.depot.y_coord,'r',marker='*',markersize=10) #最后再次设置仓库参数：颜色：红色(r)，点型：五角星，大小为10
    plt.title('vehicle-route')       #图片标题名称
    plt.xlabel('x_coord')            #y轴名称
    plt.ylabel('y_coord')            #y轴名称
    #plt.savefig('V-Route.png', dpi=300)  #指定分辨率保存

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

#主函数：GA算法框架
def GA(filepath,time_limit,capacity):
    start_time = time.time()   #定义算法起始时间
    model = Model()            #初始化model
    readExcel(filepath,model,capacity)  #读取数据并构造model
    best_sol=Sol()
    best_sol.obj=float('inf')
    model.best_sol=best_sol    #初始化全局最优解为无穷大
    initialpop(model)          #生成初始种群
    calFit(model)              #计算当初始解每条染色体对应的Route、R_dis、obj、fit
    history_best_obj = []      #历史最优解集合，记录每次迭代得到的最优目标函数值
    history_best_obj.append(model.best_sol.obj) #将初始最优解加入历史最优目标值列表中
    it = 0  #初始化迭代次数
    while time.time() - start_time < time_limit:
        select(model)             #选择操作，更新种群
        cross(model)              #交叉操作，更新种群（注意此时只更新了染色体，没更新Route、R_dis、obj、fit）
        mutation(model)           #变异操作，更新种群（注意此时只更新了染色体，没更新Route、R_dis、obj、fit）
        calFit(model)             #计算当前种群每条染色体对应的Route、R_dis、obj、fit
        history_best_obj.append(model.best_sol.obj)  #更新全局最优解
        #print(f'Iteration {it}，best obj:{model.best_sol.obj}')
        it += 1
    #plot_obj(history_best_obj)
    #plot_route(model)
    #plt.show()
    return model
        
if __name__ == '__main__':
    filepath=r'/Users/keep-rational/Desktop/算法排名作业(VRPTW)-200客户/代码/Instances/RC2_2_10.xlsx'
    run_time = 5      #算法运行次数
    time_limit = 180   #每次运行时间
    capacity = 1000     #参数:车辆最大载重
    all_best_obj = []
    all_best_Route = []
    all_Route_dis = []
    all_node_at = []
    i = 0
    while i < run_time:
        print(f"*====================第{i+1}次运行====================*")
        run_start = time.perf_counter()
        print("start time：",run_start)
        model = GA(filepath,time_limit,capacity)
        #plot_route(model)
        all_best_obj.append(model.best_sol.obj)
        all_best_Route.append(model.best_sol.Route)
        all_Route_dis.append(model.best_sol.Route_dis)
        print("最短里程：",model.best_sol.obj)
        print("最短车辆路径方案：",model.best_sol.Route)
        run_end = time.perf_counter()
        print("end time：",run_start)
        print(f'run time={run_end-run_start}秒')
        i += 1   
    output(all_best_obj,all_best_Route,all_Route_dis)