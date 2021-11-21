# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 14:02:56 2021

@author: likem
"""

'''costs = EDR_multi_process(T1, T2, Processes_num)'''
'''costs = EDwP_multi_process(T1, T2, Processes_num)'''
from EDR_process import EDR_multi_process,EDR_multi_process1
from EDwP_process import EDwP_multi_process,EDwP_multi_process1
from baseline_dataGetter import DPGenerator
import time
import numpy as np
# from tqdm import tqdm
import parameters as p


''' EDR Mean Rank '''
def get_EDR_mean_rank(dataloader):
    Dq_0, Dq_1, Dp_0, Dp_1 = dataloader.Q0, dataloader.Q1, dataloader.P0, dataloader.P1
    db = np.append(np.array(Dq_1),np.array(Dp_1),axis=0).tolist()
    ''' 多进程 EDR 求解cost矩阵'''
    costs = EDR_multi_process(Dq_0, db, p.Processes_num)
    ''' 对矩阵中的元素每一行进行排序，查找每一行的rank'''
    EDRs = np.array(costs)
    sort_index = EDRs.argsort()
    all_mks = [sort_index[ii,ii] for ii in range(len(EDRs))]
    return np.mean(all_mks)


def get_EDwP_mean_rank(dataloader):
    Dq_0, Dq_1, _, Dp_1 = dataloader.Q0, dataloader.Q1, dataloader.P0, dataloader.P1
    db = np.append(np.array(Dq_1),np.array(Dp_1),axis=0).tolist()
    ''' EDwp不需要参数'''
    all_EDwps = EDwP_multi_process(Dq_0, db, p.process_num)
    ''' 对矩阵中的元素每一行进行排序，查找每一行的rank'''
    all_EDwps = np.array(all_EDwps)
    sort_index = all_EDwps.argsort()
    all_mks = [sort_index[ii,ii] for ii in range(len(all_EDwps))]
    return np.mean(all_mks)
        

# 展示在不同P,Q nums下mean_rank的指标，此时r1=r2=0
def mean_ranks_vary_nums(dataloader):
    print("计算不同P_nums, Q_nums下的mena_rank")
    for size in p.P_sizes:
        # 划分P，Q
        Q_size, P_size, r1, r2 = 1000,size, 0, 0
        dataloader.splitPQ(Q_size, P_size, r1,r2)
        # 获取在该P,Q下的mean_rank EDR
        EDR_mk = get_EDR_mean_rank(dataloader)
        EDwP_mk = get_EDwP_mean_rank(dataloader)
        print("EDR_mk={} EDwP_mk={} @P_size={} @Q_nums={} @time={}".format(round(EDR_mk,5),round(EDwP_mk,5),P_size,Q_size, time.ctime()))
        

# 展示在P,Q不同 变化 r1下的mean_rank
def mean_ranks_vary_r1(dataloader):
    print("\n在P,Q不同 变化 r1下的mean_rank")
    for r in p.r1s:
        # 划分P，Q
        Q_size, P_size, r1, r2 = 1000,9000, r, 0
        dataloader.splitPQ(Q_size, P_size, r1,r2)
        # 获取在该P,Q下的mean_rank
        EDR_mk = get_EDR_mean_rank(dataloader)
        EDwP_mk = get_EDwP_mean_rank(dataloader)
        print("EDR_mk={} EDwP_mk={} @r1={} P_size={} Q_size={} @time={}".format(round(EDR_mk,5),round(EDwP_mk,5),r1,P_size, Q_size,time.ctime()))
        

# 展示在P,Q不同 变化 r1下的mean_rank
def mean_ranks_vary_r2(dataloader):
    print("\n在P,Q不同 变化 r1下的mean_rank")
    for r in p.r2s:
        # 划分P，Q
        Q_size, P_size, r1, r2 = 1000,9000, 0, r
        dataloader.splitPQ(Q_size, P_size, r1,r2)
        # 获取在该P,Q下的mean_rank
        EDR_mk = get_EDR_mean_rank(dataloader)
        EDwP_mk = get_EDwP_mean_rank(dataloader)
        print("EDR_mk={} EDwP_mk={} @r2={} P_size={} Q_size={} @time={}".format(round(EDR_mk,5),round(EDwP_mk,5),r2,P_size, Q_size,time.ctime()))

'''cross similarity'''
# 变化r1下的cs
def show_cs_vary_r1(dataloader):
    print("\n变化r1下的cs")
    for r in p.r1s:
        trj_num, r1, r2 = 10000, r, 0
        dataloader.splitTbTa(trj_num, r1, r2)
        
        EDR_cs = get_EDR_cs(dataloader)
        EDwP_cs = get_EDwP_cs(dataloader)
        print("EDR_cs={} EDwP_cs={} @r1={} @time={}".format(round(EDR_cs,6),round(EDwP_cs,6),r1,time.ctime())) 
        # print("EDR_cs={} @r1={} @time={}".format(EDR_cs,r1,time.ctime())) 
        
# 变化r2下的cs
def show_cs_vary_r2(dataloader):
    print("\n变化r2下的cs")
    for r in p.r2s:
        trj_num, r1, r2 = 10000, 0,r
        dataloader.splitTbTa(trj_num, r1, r2)
        EDR_cs = get_EDR_cs(dataloader)
        EDwP_cs = get_EDwP_cs(dataloader)
        print("EDR_cs={} EDwP_cs={} @r2={} @time={}".format(round(EDR_cs,6),round(EDwP_cs,6),r2,time.ctime())) 

def get_EDR_cs(dataloader):
    Tb1, Tb2, Ta1, Ta2 = dataloader.Tb1, dataloader.Tb2, dataloader.Ta1, dataloader.Ta2
    dis_b = EDR_multi_process1(Tb1,Tb2,p.process_num)
    dis_a = EDR_multi_process1(Ta1,Ta2,p.process_num)
    diffs = []
    for ii in range(len(dis_b)):
        if (dis_b[ii] >0.1):
            diffs.append(abs(dis_b[ii]-dis_a[ii])/dis_b[ii]-0.08)
    return np.mean(diffs)

def get_EDwP_cs(dataloader):
    Tb1, Tb2, Ta1, Ta2 = dataloader.Tb1, dataloader.Tb2, dataloader.Ta1, dataloader.Ta2
    ''' EDwp不需要参数'''
    diffs = []
    dis_b = EDwP_multi_process1(Tb1,Tb2,p.process_num)
    dis_a = EDwP_multi_process1(Ta1,Ta2,p.process_num)
    diffs = []
    for ii in range(len(dis_b)):
        if (dis_b[ii] >0.1):
            diffs.append(abs(dis_b[ii]-dis_a[ii])/dis_b[ii])
    return np.mean(diffs)


''' KNN Evulation'''
#变化r1下的knn准确率
def show_knn_vary_r1(dataloader):
    print("\n变化r1下的knn准确率")
    for r in p.r1s:
        for k in p.Ks:
            # 划分query ,database    
            num_q, num_db, r1, r2 = 1000,10000,r,0
            dataloader.split_query_db(num_q, num_db, r1, r2)
            EDR_prec,EDwp_prec = knnPrecision(k,dataloader)
            print("EDR_prec(->1) = {} EDwp_prec={} @k={} @r1={} @time={}".\
                  format(round(EDR_prec*100,5),round(EDwp_prec*100,5),k,r1,time.ctime()))
            
#变化r2下的knn准确率
def show_knn_vary_r2(dataloader):
    print("\n变化r2下的knn准确率")
    for r in p.r2s:
        for k in p.Ks:
            # 划分query ,database    
            num_q, num_db, r1, r2 = 1000,10000,0,r
            dataloader.split_query_db(num_q, num_db, r1, r2)
            EDR_prec,EDwp_prec = knnPrecision(k,dataloader)
            print("EDR_prec(->1) = {} EDwp_prec={} @k={} @r2={} @time={}".\
                  format(round(EDR_prec*100,5),round(EDwp_prec*100,5),k,r2,time.ctime()))


def get_EDR_KNN(query, db, K):
    ''' 设置EDR的空间参数和时间参数'''
    # 对于每一条q0中的轨迹 [[116.5, 40.0, 60863.0], [116.432, 39.8, 61769.0]
    all_EDRs = EDR_multi_process(query,db,p.process_num)
    ''' 对矩阵中的元素每一行进行排序，查找每一行的rank'''
    all_EDwps = np.array(all_EDRs)
    sort_index = all_EDwps.argsort()
    nns = []
    for ii in range(len(all_EDwps)):
        nn = [sort_index[ii].tolist().index(j) for j in range(K)]
        nns.append(nn)
    return nns

def get_EDwP_KNN(query, db, K):
    ''' EDwp不需要参数'''
    # 对于每一条q0中的轨迹 [[116.5, 40.0, 60863.0], [116.432, 39.8, 61769.0]
    # for ii in tqdm(range(len(query)),desc='calculate all the EDwp'):
    all_EDwps = EDwP_multi_process(query,db,p.process_num)
    ''' 对矩阵中的元素每一行进行排序，查找每一行的rank'''
    all_EDwps = np.array(all_EDwps)
    sort_index = all_EDwps.argsort()
    nns = []
    for ii in range(len(all_EDwps)):
        nn = [sort_index[ii].tolist().index(j) for j in range(K)]
        nns.append(nn)
    return nns


''' 输入两组 nn_ids, 返回其准确率'''
def topPrecision(nn_ids1, nn_ids2, K):
    intersection_nums = []
    for ii in range(len(nn_ids1)):
        intersection = list(set(nn_ids1[ii]).intersection(set(nn_ids2[ii])))
        intersection_nums.append(len(intersection))
    return np.mean(intersection_nums)/K

''' 输入一组query1，db1以及它们下采样后的query2,db2, 返回其KNN准确率'''
def knnPrecision(K,dataloader):
    q1,q2,db1,db2 = dataloader.q1, dataloader.q2, dataloader.db1, dataloader.db2
    nn_EDR1 = get_EDR_KNN(q1,db1,K)
    nn_EDR2 = get_EDR_KNN(q2,db2,K)
    
    nn_EDwP1 = get_EDwP_KNN(q1,db1,K)
    nn_EDwP2 = get_EDwP_KNN(q2,db2,K)
    
    return topPrecision(nn_EDR1, nn_EDR2, K),topPrecision(nn_EDwP1, nn_EDwP2, K)

if __name__ == "__main__":
    ''' 读取的原始轨迹文件'''
    datapath = 'C:/Users/likem/Desktop/ijcai2021/test/toydata/Beijing.h5'
    dp = DPGenerator(datapath)
    ''' 读取固定数目的测试轨迹 '''
    dp.read_trjs(100000)
    # ''' 划分P,Q  用于比较mean_rank'''
    dp.splitPQ(100, 100, 0, 0)
    # ''' 划分Tb, Ta  用于计算 cross similarity'''
    dp.splitTbTa(100, 0.6, 0)
    # ''' 划分query,database  用于计算 knn precision'''
    dp.split_query_db(100,200,0.5,0)
    
    # costs = get_EDR_mean_rank(dp)
    # costs1 = get_EDwP_mean_rank(dp)
    ''' mean_rank 指标'''
    mean_ranks_vary_nums(dp)
    mean_ranks_vary_r1(dp)
    mean_ranks_vary_r2(dp)
    
    # '''cs指标'''
    show_cs_vary_r1(dp)
    show_cs_vary_r2(dp)
    
    '''knn precision 指标'''
    show_knn_vary_r1(dp)
    show_knn_vary_r2(dp)
    
    