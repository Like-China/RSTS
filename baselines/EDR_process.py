# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 13:32:19 2021

@author: likem
"""
import numpy as np
from multiprocessing import Manager, Process
import time
import parameters as p
from baseline_dataGetter import DPGenerator

class EDR:
    # threshold of equivalence
    def __init__(self):
        self.min_long = p.longs[0]
        self.max_long = p.longs[1]
        self.min_lat = p.lats[0]
        self.max_lat = p.lats[1]
        
    # two tuples
    def eq(self, c1, c2):
        # return (abs(c1[0] - c2[0])/(self.max_long-self.min_long) <= self.xyDIFF) and (abs(c1[1] - c2[1])/(self.max_lat-self.min_lat) <= self.xyDIFF) \
        #     and (abs(c1[2]-c2[2])/86400 <= self.timeDiff)
        return abs(c1[0] - c2[0]) <= p.xyDiff and abs(c1[1] - c2[1]) <= p.xyDiff \
            # and abs(c1[2]-c2[2]) <= p.timeDiff
    
    ''' 递归求解EDR'''
    def EDR(self,S, R):
        if len(S) == 0:
            return len(R)
        if len(R) == 0:
            return len(S)
        subcost = 0 if self.eq(S[0], R[0]) else 1;
        a = self.EDR(R[1:], S[1:]) + subcost
        b = self.EDR(R[1:], S    ) + 1
        c = self.EDR(R    , S[1:]) + 1
        return min(a, b, c)
    
    ''' 动态规划求解 EDR'''
    def DP_EDR(self, S, R):
        ''' 创建动态规划矩阵'''
        row_nums = len(R)+1
        col_nums = len(S)+1
        # M = np.zeros((row_nums,col_nums))
        M = np.zeros((2,col_nums))
        M[1] = col_nums
        M[0,col_nums-1] = row_nums
        
        
        for row in range(row_nums-2,-1,-1):
            for col in range(col_nums-2,-1,-1):
                subcost = 0 if self.eq(R[row],S[col]) else 1
                # M[row,col] = min(M[row+1,col+1]+subcost, M[row,col+1]+1, M[row+1,col]+1)
                M[0,col] = min(M[1,col+1]+subcost, M[0,col+1]+1, M[1,col]+1)
            ''' 完成一行，置换M中的第二行为第一行，第一行为全0'''
            if row != 0:
                M[1,:] = M[0,:]
                M[0,:] = 0
                # print("{}-{}".format(row,col))
                # print(self.eq(R[row],S[col]))
                # print(M)
                # print("\n")
        return M[0,0]
    
 
def func(T1, T2, costs, index, num_T1, num_T2, Processes_num):
    gap = num_T1 // Processes_num
    begin = index * gap
    end = (index + 1) * gap
    if index == Processes_num-1:
        end = num_T1
    b1 = EDR()
    
    for i in range(begin, end):
        cost = [None] * num_T2
        for j in range(0, num_T2):
            cost[j] = b1.DP_EDR(T1[i], T2[j])
        costs[i] = cost
        # costs.append(cost)
        # print(cost)
        
def EDR_multi_process(T1, T2, Processes_num):
     # b2 = EDwP()
     manger = Manager()
     costs = manger.dict()  # 使用字典
     time1 = time.time()
    
     Processes = []
     for i in range(Processes_num):
        # func(T1,T2,costs,i,num,Processes_num)
        p = Process(target=func, args=(T1, T2, costs, i, len(T1),len(T2), Processes_num))
        Processes.append(p)
        Processes[i].start()

     for i in range(Processes_num):
        Processes[i].join()

     # print(round(time.time() - time1))
     
     ''' 按照键值进行排序 后输出'''
     res = []
     for ii in range(len(T1)):
         res.append(costs.get(ii))
     return res

''' 只计算对应行间的损失值'''
def func1(T1, T2, costs, index, num_T1, num_T2, Processes_num):
    gap = num_T1 // Processes_num
    begin = index * gap
    end = (index + 1) * gap
    if index == Processes_num-1:
        end = num_T1
    xyDiff = 0.1
    timeDiff = 0.1
    b1 = EDR()
    for i in range(begin, end):
        costs[i] = b1.DP_EDR(T1[i], T2[i])
    
def EDR_multi_process1(T1, T2, Processes_num):
     # b2 = EDwP()
     manger = Manager()
     costs = manger.dict()  # 使用字典
     # time1 = time.time()
    
     Processes = []
     for i in range(Processes_num):
        # func(T1,T2,costs,i,num,Processes_num)
        p = Process(target=func1, args=(T1, T2, costs, i, len(T1),len(T2), Processes_num))
        Processes.append(p)
        Processes[i].start()

     for i in range(Processes_num):
        Processes[i].join()

     # print(round(time.time() - time1))
     
     ''' 按照键值进行排序 后输出'''
     res = []
     for ii in range(len(T1)):
         res.append(costs.get(ii))
     return res


'''test'''
if __name__ == "__main__":
    # xyDiff = 10
    # timeDiff = 1000
    # b1 = EDR(xyDiff,timeDiff)
    # print("S1-S2: ", b1.EDR(S1, S2))
    # print("S1-S3: ", b1.EDR(S1, S3))
    # print("S2-S3: ", b1.EDR(S2, S3))
    # print("S3-S4: ", b1.EDR(S3, S4)) 
    # num_T1 = 200
    # num_T2 = 200
    
    # t1 = [[116.42008999999999, 39.96032, 441.0], [116.41283, 39.95265, 1410.0], [116.47835, 39.913959999999996, 4019.0], [116.47185, 39.90733, 4320.0], [116.45806999999999, 39.94815, 5701.0], [116.47018, 39.95265, 6225.0], [116.48226000000001, 39.97217, 6828.0], [116.46824, 39.94185, 7432.0], [116.46875, 39.93646, 8337.0], [116.46021, 39.95841, 8941.0], [116.48308999999999, 39.98877, 9544.0], [116.50894, 40.012640000000005, 35279.0], [116.4866, 40.00656, 35743.0], [116.4653, 40.00062, 37695.0], [116.45433, 39.98806, 38020.0], [116.43418999999999, 39.926559999999995, 40969.0], [116.43675, 39.93508, 41573.0], [116.43845, 39.93934, 42180.0], [116.42055, 39.904509999999995, 43377.0], [116.47273, 39.92695, 45164.0], [116.44698000000001, 39.922109999999996, 45767.0], [116.41618999999999, 39.92305, 46371.0], [116.39283999999999, 39.922, 46673.0], [116.37024, 39.908609999999996, 47276.0], [116.30365, 39.9046, 47880.0]]
    # t2 = [[116.4226, 39.96092, 743.0], [116.4292, 39.88944, 2958.0], [116.47185, 39.90733, 4320.0], [116.49298999999999, 39.95841, 5144.0], [116.46586, 39.948479999999996, 5923.0], [116.48383999999999, 39.96839, 6526.0], [116.47873999999999, 39.95558, 7130.0], [116.46866000000001, 39.9364, 8035.0], [116.46841, 39.94123, 8639.0], [116.4777, 39.97484, 9242.0], [116.5157, 40.021679999999996, 9846.0], [116.48811, 40.00525, 35581.0], [116.486, 40.0067, 37393.0], [116.45575, 39.98833, 37997.0], [116.43233000000001, 39.98082, 38322.0], [116.43055, 39.93002, 41271.0], [116.44253, 39.9369, 41869.0], [116.42823999999999, 39.94085, 42482.0], [116.47623999999999, 39.926809999999996, 44890.0], [116.45629, 39.92198, 45466.0], [116.43348999999999, 39.922959999999996, 46069.0], [116.41618999999999, 39.92305, 46371.0], [116.37558, 39.91963, 46975.0], [116.33209, 39.90625, 47578.0], [116.30791, 39.91992, 48483.0]]
    # print("S3-S4: ", b1.EDR(T1, T2))
    # T1 = [t1]*num_T1
    
    # T2 = [t2]*num_T2
    
    
    datapath = 'C:/Users/likem/Desktop/ijcai2021/test/toydata/Beijing.h5'
    dp = DPGenerator(datapath)
    ''' 读取固定数目的测试轨迹 '''
    dp.read_trjs(1000)
    # ''' 划分Tb, Ta  用于计算 cross similarity'''
    dp.splitTbTa(100, 0.6, 0)
    Processes_num = 10
    
    print(time.ctime())
    costs = EDR_multi_process(dp.Ta1, dp.Ta2, Processes_num)
    print(time.ctime())
    # t1 = time.time()
    # for i in range(num_T1):
    #     for j in range(num_T2):
    #         b1.DP_EDR(T1[i], T2[j])
    # print(time.time()-t1)