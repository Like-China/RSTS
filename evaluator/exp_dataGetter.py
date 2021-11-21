# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 22:27:01 2021

@author: likem
"""

import os
import random
import warnings
warnings.filterwarnings("ignore")
from preprocessor.Region import Region
## D,P 测试数据集生成器
##  从验证集中划分得到P0,P1, Q0,Q1用于计算mean_rank
##  从验证集合中得到 Tb,Tb',Ta,Ta'用于计算cross-similarity
##  从验证集中获取query,db用于计算KNN准确率
class DPGenerator:
    
    def __init__(self, args):
       
        self.args = args
        self.datapath = os.path.join(args.data,'val.trg')
        '''读取所有测试机轨迹'''
        self.read_trjs()
        ''' 建立一个region对象用于产生噪声和下采样'''
        print("创建数据生成对象")
        print("创建Region对象，生成词汇表")
        self.R = Region(args)
        
        
        ''' test'''
        # self.splitPQ(100,200,0.5,0)
        # self.splitTbTa(200,0.5,0)
        # self.split_query_db(1000, 10000, 0.5, 0)
        
        
    '''读取所有测试机轨迹'''
    def read_trjs(self):
        # 读取所有供测试使用的轨迹数据
        self.all_trj = []
        with open(self.datapath,'r') as f:
            read_nums = 0
            for line in f:
                if (read_nums>self.args.read_val_nums):break
                mm = [int(x) for x in line.split()]
                self.all_trj.append(mm)
                
                 
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
                        t1 = self.R.downsampling(t1,r1)
                        t2 = self.R.downsampling(t2,r1)
                    if r2>0:
                        t1 = self.R.distort(t1,r2)
                        t2 = self.R.distort(t2,r2)
                    self.Q0.append(t1)
                    self.Q1.append(t2)
            for taj in self.P:
                t1 = [taj[ii] for ii in range(len(taj)) if ii%2==0]
                t2 = [taj[ii] for ii in range(len(taj)) if ii%2==1]
                if len(t1)>10 and len(t2)>10:
                    if r1>0:
                        t1 = self.R.downsampling(t1,r1)
                        t2 = self.R.downsampling(t2,r1)
                    if r2>0:
                        t1 = self.R.distort(t1,r2)
                        t2 = self.R.distort(t2,r2)
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
                        each1 = self.R.downsampling(each1,r1)
                        each2 = self.R.downsampling(each2,r1)
                    if r2>0:
                        each1 = self.R.distort(each1,r2)
                        each2 = self.R.distort(each2,r2)
                    ''' 添加有区别的 Tb1 Tb2两个轨迹集合 下采样和噪声后的轨迹集合 Ta1,Ta2'''
                    self.Ta1.append(each1)
                    self.Ta2.append(each2)
        return self.Tb1, self.Tb2, self.Ta1, self.Ta2, 
                
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
                        each = self.R.downsampling(each,r1)
                    if r2>0:
                        each = self.R.distort(each,r2)
                    self.q2.append(each)
            for each in db:
                if len(each)>10:
                    self.db1.append(each)
                    if r1>0:
                        each = self.R.downsampling(each,r1)
                    if r2>0:
                        each = self.R.distort(each,r2)
                    self.db2.append(each)
        return self.q1, self.q2, self.db1, self.db2
    
    
    

        
        
        
        
            
        
            
       
        
        
            