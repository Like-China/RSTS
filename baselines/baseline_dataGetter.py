# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 22:27:01 2021

@author: likem
"""

import numpy as np
import random
import h5py
import warnings
warnings.filterwarnings("ignore")


##   D,P 测试数据集生成器
##  从验证集中划分得到P0,P1, Q0,Q1用于计算mean_rank
##  从验证集合中得到 Tb,Tb',Ta,Ta'用于计算cross-similarity
##  从验证集中获取query,db用于计算KNN准确率
class DPGenerator:
    
    def __init__(self, datapath):
       
        
        self.datapath = datapath
        
        
    '''读取所有测试集轨迹 为trj = [[x,y,t],[x,y,t]'''
    def read_trjs(self,nums):
        # 读取所有供测试使用的轨迹数据成为trj = [[x,y,t],[x,y,t]形式]
        self.all_trj = []
        with h5py.File(self.datapath,'r') as f:
            for ii in range(nums):
                trip = np.array(f.get('trips/'+str(ii+1)))
                ts = np.array(f.get('timestamps/'+str(ii+1)))
                ts = [[a] for a in ts]
                self.all_trj.append(np.hstack((trip,ts)).tolist())
        
                 
    def splitPQ(self, num_Q, num_P, r1, r2):
        
        if(len(self.all_trj)>(num_Q+num_P)): 
            # 设置随机数任选P,Q
            random_selected = random.sample(self.all_trj, num_Q+num_P)
            self.Q = random_selected[0:num_Q]
            self.P = random_selected[num_Q:num_Q+num_P]
            ''' 保证Q0-Q1,P0-P1个数相同'''
            self.P0,self.P1,self.Q0,self.Q1 = [],[],[],[]
            for taj in self.Q:
                t1 = [taj[ii] for ii in range(len(taj)) if ii%2==0]
                t2 = [taj[ii] for ii in range(len(taj)) if ii%2==1]
                if len(t1)>10 and len(t2)>10:
                    if r1>0:
                        t1 = self.downsampling(t1,r1)
                        t2 = self.downsampling(t2,r1)
                    if r2>0:
                        t1 = self.distort(t1,r2)
                        t2 = self.distort(t2,r2)
                    self.Q0.append(t1)
                    self.Q1.append(t2)
            for taj in self.P:
                t1 = [taj[ii] for ii in range(len(taj)) if ii%2==0]
                t2 = [taj[ii] for ii in range(len(taj)) if ii%2==1]
                if len(t1)>10 and len(t2)>10:
                    if r1>0:
                        t1 = self.downsampling(t1,r1)
                        t2 = self.downsampling(t2,r1)
                    if r2>0:
                        t1 = self.distort(t1,r2)
                        t2 = self.distort(t2,r2)
                    self.P0.append(t1)
                    self.P1.append(t2)
        else:
            print("Error:Data lacked")
        return self.Q0, self.Q1, self.P0,self.P1
            
    def splitTbTa(self,trj_num, r1, r2):
        if(len(self.all_trj)>2*trj_num): 
            # 设置随机数任选Tb,Ta
            random_selected = random.sample(self.all_trj, 2*trj_num)
            b = random_selected[0:trj_num]
            bb = random_selected[trj_num:trj_num*2]
            self.Tb1 = []
            self.Tb2 = []
            self.Ta1 = []
            self.Ta2 = []
            ''' 保证Ta1=Ta2=Tb1=Tb2'''
            for ii in range(trj_num):
                each1 = b[ii]
                each2 = bb[ii]
                if len(each1)>10 and len(each2)>10:
                    ''' 添加有区别的 Tb1 Tb2两个轨迹集合'''
                    self.Tb1.append(each1)
                    self.Tb2.append(each2)
                    if r1>0:
                        each1 = self.downsampling(each1,r1)
                        each2 = self.downsampling(each2,r1)
                    if r2>0:
                        each1 = self.distort(each1,r2)
                        each2 = self.distort(each2,r2)
                    ''' 添加有区别的 Tb1 Tb2两个轨迹集合 下采样和噪声后的轨迹集合 Ta1,Ta2'''
                    self.Ta1.append(each1)
                    self.Ta2.append(each2)
        return self.Tb1, self.Tb2, self.Ta1, self.Ta2 
                
    def split_query_db(self, num_q, num_db, r1, r2):
        if(len(self.all_trj)>num_q+num_db): 
            # 设置随机数任选Tb,Ta
            random_selected = random.sample(self.all_trj, num_q+num_db)
            q = random_selected[0:num_q]
            db = random_selected[num_q:num_q+num_db]
            self.q1 = []
            self.db1 = []
            self.q2 = []
            self.db2 = []
            ''' 保证q1=q2,db1=db2'''
            for each in q:
                if len(each)>10:
                    self.q1.append(each)
                    if r1>0:
                        each = self.downsampling(each,r1)
                    if r2>0:
                        each = self.distort(each,r2)
                    self.q2.append(each)
            for each in db:
                if len(each)>10:
                    self.db1.append(each)
                    if r1>0:
                        each = self.downsampling(each,r1)
                    if r2>0:
                        each = self.distort(each,r2)
                    self.db2.append(each)
        return self.q1, self.q2, self.db1, self.db2
    
    ''' 对一条轨迹进行下采样 输入输出均为列表'''
    def downsampling(self,vocals, rate):
        randx = np.random.rand(len(vocals))>rate
        sample_vocals = np.array(vocals)[randx]
        return sample_vocals.tolist() if len(sample_vocals)>5 else vocals

    
    
    ''' 对一条轨迹进行distort'''
    def distort(self,vocals, rate):
        mu = 0
        sigma = 0.0001
        noise_words = []
        for word in vocals:
            if(random.random()<rate):
                ''' distort 经度'''
                word[0] = word[0]+random.gauss(mu,sigma)
                ''' distort 维度'''
                word[1] = word[1]+ random.gauss(mu,sigma)
                word[2] = word[2]+ random.gauss(mu,10)
            else:
                noise_words.append(word)
        return noise_words
        
    
'''test'''
if __name__ == "__main__":
    ''' 读取的原始轨迹文件'''
    datapath = 'Beijing.h5'
    dp = DPGenerator(datapath)
    ''' 读取固定数目的测试轨迹 '''
    dp.read_trjs(1000)
    # ''' 划分P,Q  用于比较mean_rank'''
    dp.splitPQ(100, 200, 0.5, 0.5)
    # ''' 划分Tb, Ta  用于计算 cross similarity'''
    dp.splitTbTa(100, 0.6, 0)
    # ''' 划分query,database  用于计算 knn precision'''
    dp.split_query_db(100,200,0.5,0)
    



        
            
        
            
       
        
        
            