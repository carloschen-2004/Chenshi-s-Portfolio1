# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math

#函数：读取原始数据文件
def readExcel(filepath):
    #读取文件路径里的文件，生成二维矩阵数据
    DF = pd.read_excel(filepath)
    DF.loc[len(DF)] =  DF.iloc[0]
    #原始数据客户点数量
    num_c = len(DF) - 2
    #构造距离矩阵
    dis_Matrix = np.full((len(DF),len(DF)), float('inf'), dtype=(float))  #创建权值矩阵
    for i in range(len(DF)):
        for j in range(len(DF)):
            if i != j:
                dis_Matrix[i][j] = math.sqrt((DF['x_coord'][i]-DF['x_coord'][j])**2+(DF['y_coord'][i]-DF['y_coord'][j])**2)
    #提取各点需求量，时间窗
    demand = DF['demand'].values
    et = DF['et'].values
    lt = DF['lt'].values
    st = DF['st'].values
    lt = lt[1:201] #只保留客户序号对应的时间窗
    et = et[1:201] #只保留客户序号对应的时间窗
    st = st[1:201] #只保留客户序号对应的时间窗
    return DF,dis_Matrix,demand,num_c,et,lt,st

#函数：读取数据文件并进行验证
def checkRoute(filepath, dis_Matrix,demand,capacity,num_c,et,lt,st):
    #验证里程和需求
    all_sheet = pd.read_excel(filepath, sheet_name=None)
    R = 1
    for sheet_name,Route in all_sheet.items():
        if sheet_name != 'sheet0':
            reslut_R = Route.columns[1] #要验证的路径方案总里程
            count_c = 0 #客户点计数器,初始化为0
            dis_R = 0  #计算路径方案总里程，初始化为0
            for i in range(len(Route)):
                temp_r = eval(Route.iloc[i,1]) #将str格式转换为list格式
                #验证里程,需求
                count_c += len(temp_r) #计算单条路径的客户点数
                dis_r = 0 #单条车辆路径的里程，初始化为0
                demand_r = 0 #单条车辆路径的载重，初始化为0
                for j in range(len(temp_r)-1):
                    dis_r += dis_Matrix[temp_r[j]+1][temp_r[j+1]+1] #计算该路径上所有客户点间的行驶里程
                    demand_r += demand[temp_r[j]+1] #计算该路径上所有客户点需求量
                dis_r = dis_r + dis_Matrix[0][temp_r[0]+1] + dis_Matrix[temp_r[-1]+1][0] #计算起始点里程
                dis_R += dis_r
                if demand_r > capacity:
                    print(f"xxxxxxxxxxxxx第{R}个解验证有误,第{i}条路径载重量为{demand_r}>{capacity}")
                if round(dis_r,2) != round(Route.iloc[i,2],2):
                    print(f"xxxxxxxxxxxxx第{R}个解验证有误,第{i}条路径结果里程为{Route.iloc[i,2]},实际计算里程为{dis_r}")
                #验证时间窗
                prev = 0
                prev_time = 0
                for j in range(len(temp_r)-1):
                    c = temp_r[j] #当前客户点序号
                    #第一个访问点
                    if j == 0:
                        at = max(0+dis_Matrix[0][c+1], et[c])
                        if at > lt[c]:
                            print(f"xxxxxxxxxxxxx第{R}个解有误,客户{j}的访问时间为{at},不满足最晚服务时间窗{lt[c]}")
                        prev = c
                        prev_time = at + st[c]
                    #非第一个节点
                    else:
                        at = max(prev_time+dis_Matrix[prev+1][c+1], et[c])
                        if at > lt[c]:
                            print(f"xxxxxxxxxxxxx第{R}个解有误,客户{j}的访问时间为{at},不满足最晚服务时间窗{lt[c]}")
                        prev = c
                        prev_time = at + st[c]
                        
            if count_c != num_c:
                print(f"xxxxxxxxxxxxx第{R}个解验证有误,原始算例客户点为{num_c}个,解中客户点为{count_c}个")
            elif round(dis_R,2) == round(reslut_R,2):
                print(f"第{R}个解验证正确,dis_R = {dis_R}")
            else:
                print(f"xxxxxxxxxxxxx第{R}个解验证有误,dis_R = {dis_R}, reslut_R = {reslut_R}")
            R += 1       
    return all_sheet

if __name__ == '__main__':
    capacity = 1000  
    Instance_file=r'/Users/keep-rational/Desktop/算法排名作业(VRPTW)-200客户/代码/Instances/RC2_2_10.xlsx'
    DF,dis_Matrix,demand,num_c,et,lt,st = readExcel(Instance_file)
    result_file='result.xlsx'
    all_sheet = checkRoute(result_file, dis_Matrix, demand, capacity, num_c, et,lt,st)
