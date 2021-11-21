"""
Created on Mon Dec 28 10:54:27 2020

@author: likem
"""
# -*- coding: utf-8 -*-
import numpy as  np
import os
from tqdm import tqdm
from scipy import spatial
from evaluator.exp_dataGetter import DPGenerator
from loader.data_utils import DataOrderScaner,pad_arrays_keep_invp
from trainer.train import EncoderDecoder
import torch
import time
from trainer.t2vec import setArgs



class Trj2vec:
    def __init__(self,args, m0):
        self.args = args
        self.m0 = m0
    
    '''读取trj.t中的轨迹，返回最后一层输出作为向量表示
      输入： trj_file_path 需要转换为向量表示的轨迹文件
      输出 vecs[m0.num_layers-1] decoder的最后一层输出，
          为该组轨迹的向量表示 batch*向量维度 格式：列表
          函数内部自己从文件中加载模型'''
    def t2vec_noModel(self, trj_file_path):
        "read source sequences from trj.t and write the tensor into file trj.h5"
        args = self.args
        m0 = EncoderDecoder(args.vocab_size, args.embedding_size,
                            args.hidden_size, args.num_layers,
                            args.dropout, args.bidirectional)
        # 加载训练模型
        if os.path.isfile(args.checkpoint):
            # print("=> loading checkpoint '{}'".format(args.checkpoint))
            checkpoint = torch.load(args.checkpoint)
            m0.load_state_dict(checkpoint["m0"])
            if torch.cuda.is_available():
                m0.cuda()
            # 不启用dropout和BN
            m0.eval()
            vecs = []
            scaner = DataOrderScaner(os.path.join(args.data, trj_file_path), args.t2vec_batch)
            scaner.load()
            i = 0
            while True:
                i = i + 1
                # src 该组最大轨迹长度*num_seqs(该组轨迹个数) 
                src, lengths, invp = scaner.getbatch()
                if src is None: break
                if torch.cuda.is_available():
                    src, lengths, invp = src.cuda(), lengths.cuda(), invp.cuda()
                h, _ = m0.encoder(src, lengths) # 【层数*双向2，该组轨迹个数，隐藏层数】【6，10，128】
                ## (num_layers, batch, hidden_size * num_directions) 【3，10，256】
                h = m0.encoder_hn2decoder_h0(h)
                ## (batch, num_layers, hidden_size * num_directions) 【10，3，256】
                h = h.transpose(0, 1).contiguous()
                ## (batch, *)
                #h = h.view(h.size(0), -1)
                vecs.append(h[invp].cpu().data)
            ## (num_seqs, num_layers, hidden_size * num_directions)
            
            vecs = torch.cat(vecs) # [10,3,256]
            ## (num_layers, num_seqs, hidden_size * num_directions)
            vecs = vecs.transpose(0, 1).contiguous()  ## [3,10,256]
        else:
            print("=> no checkpoint found at '{}'".format(args.checkpoint))
        # 返回最后一层作为该批次 轨迹的代表
        return vecs[m0.num_layers-1].squeeze(0).numpy().tolist()
    
    '''读取trj.t中的轨迹，返回最后一层输出作为向量表示
      输入： trj_file_path 需要转换为向量表示的轨迹文件
      输出 vecs[m0.num_layers-1] decoder的最后一层输出，
          为该组轨迹的向量表示 batch*向量维度 格式：列表
          模型作为参数进行代入计算'''
    def t2vec(self, m0, trj_file_path):
        "read source sequences from trj.t and write the tensor into file trj.h5"
        args = self.args
        
        # print("=> loading checkpoint '{}'".format(args.checkpoint))
        if torch.cuda.is_available():
            m0.cuda()
        # 不启用dropout和BN
        m0.eval()
        vecs = []
        scaner = DataOrderScaner(os.path.join(args.data, trj_file_path), args.t2vec_batch)
        scaner.load()
        i = 0
        while True:
            i = i + 1
            # src 该组最大轨迹长度*num_seqs(该组轨迹个数) 
            src, lengths, invp = scaner.getbatch()
            if src is None: break
            if torch.cuda.is_available():
                src, lengths, invp = src.cuda(), lengths.cuda(), invp.cuda()
            h, _ = m0.encoder(src, lengths) # 【层数*双向2，该组轨迹个数，隐藏层数】【6，10，128】
            ## (num_layers, batch, hidden_size * num_directions) 【3，10，256】
            h = m0.encoder_hn2decoder_h0(h)
            ## (batch, num_layers, hidden_size * num_directions) 【10，3，256】
            h = h.transpose(0, 1).contiguous()
            ## (batch, *)
            #h = h.view(h.size(0), -1)
            vecs.append(h[invp].cpu().data)
        ## (num_seqs, num_layers, hidden_size * num_directions)
        
        vecs = torch.cat(vecs) # [10,3,256]
        ## (num_layers, num_seqs, hidden_size * num_directions)
        vecs = vecs.transpose(0, 1).contiguous()  ## [3,10,256]
        # 返回最后一层作为该批次 轨迹的代表
        return vecs[m0.num_layers-1].squeeze(0).numpy().tolist()

    
    '''输入：轨迹集合T
    输出：对应的向量表示
    注意T中的轨迹数需要大于 args.t2vec_batch '''
    def t2vec_input(self,T, m0):
        args = self.args
        "read source sequences from trj.t and write the tensor into file trj.h5"
        if torch.cuda.is_available():
            m0.cuda()
        m0.eval()
        
        vecs = []
        
        for ii in range(len(T)//args.t2vec_batch+1):
            # src 该组最大轨迹长度*num_seqs(该组轨迹个数) 
            src, lengths, invp = pad_arrays_keep_invp(T[ii*args.t2vec_batch:(ii+1)*args.t2vec_batch])
            if src is None: break
            if torch.cuda.is_available():
                src, lengths, invp = src.cuda(), lengths.cuda(), invp.cuda()
            h, _ = m0.encoder(src, lengths) # 【层数*双向2，该组轨迹个数，隐藏层数】【6，10，128】
            ## (num_layers, batch, hidden_size * num_directions) 【3，10，256】
            h = m0.encoder_hn2decoder_h0(h)
            ## (batch, num_layers, hidden_size * num_directions) 【10，3，256】
            h = h.transpose(0, 1).contiguous()
            ## (batch, *)
            #h = h.view(h.size(0), -1)
            vecs.append(h[invp].cpu().data)
        ## (num_seqs, num_layers, hidden_size * num_directions)
        
        vecs = torch.cat(vecs) # [10,3,256]
        # ## 存储三层 输出的隐藏层结构，每一层是 batch个256维的向量
        ## (num_layers, num_seqs, hidden_size * num_directions)
        vecs = vecs.transpose(0, 1).contiguous()  ## [3,10,256]
        return vecs[m0.num_layers-1].squeeze(0).numpy().tolist()


    '''输入：轨迹集合T
    输出：对应的向量表示
    注意T中的轨迹数需要大于 args.t2vec_batch '''
    def t2vec_input_noModel(self,T):
        "read source sequences from trj.t and write the tensor into file trj.h5"
        args = self.args
        m0 = EncoderDecoder(args.vocab_size, args.embedding_size,
                            args.hidden_size, args.num_layers,
                            args.dropout, args.bidirectional)
        # 加载训练模型
        if os.path.isfile(args.checkpoint):
            # print("=> loading checkpoint '{}'".format(args.checkpoint))
            checkpoint = torch.load(args.checkpoint)
            m0.load_state_dict(checkpoint["m0"])
            if torch.cuda.is_available():
                m0.cuda()
            # 不启用dropout和BN
            m0.eval()
        else:
            print("不存在模型")
            return
        vecs = []
        
        for ii in range(len(T)//args.t2vec_batch+1):
            # src 该组最大轨迹长度*num_seqs(该组轨迹个数) 
            src, lengths, invp = pad_arrays_keep_invp(T[ii*args.t2vec_batch:(ii+1)*args.t2vec_batch])
            if src is None: break
            if torch.cuda.is_available():
                src, lengths, invp = src.cuda(), lengths.cuda(), invp.cuda()
            h, _ = m0.encoder(src, lengths) # 【层数*双向2，该组轨迹个数，隐藏层数】【6，10，128】
            ## (num_layers, batch, hidden_size * num_directions) 【3，10，256】
            h = m0.encoder_hn2decoder_h0(h)
            ## (batch, num_layers, hidden_size * num_directions) 【10，3，256】
            h = h.transpose(0, 1).contiguous()
            ## (batch, *)
            #h = h.view(h.size(0), -1)
            vecs.append(h[invp].cpu().data)
        ## (num_seqs, num_layers, hidden_size * num_directions)
        
        vecs = torch.cat(vecs) # [10,3,256]
        # ## 存储三层 输出的隐藏层结构，每一层是 batch个256维的向量
        ## (num_layers, num_seqs, hidden_size * num_directions)
        vecs = vecs.transpose(0, 1).contiguous()  ## [3,10,256]
        return vecs[m0.num_layers-1].squeeze(0).numpy().tolist()



class My_criterion:
    
    def __init__(self,args, m0):
        self.args = args
        self.m0 = m0
        self.convert = Trj2vec(args,m0)
        self.dataloader = DPGenerator(args)
    
    '''计算两组 轨迹代表向量集合 中各个向量到各个向量间的距离
    输入： A B 向量列表
    输出： ED=[[0,2,3],[2,0,4]] list'''
    def EuclideanDistances(self,A, B):
        A, B = np.array(A), np.array(B)
        BT = B.transpose()
        # vecProd = A * BT
        vecProd = np.dot(A,BT)
        # print(vecProd)
        SqA =  A**2
        # print(SqA)
        sumSqA = np.matrix(np.sum(SqA, axis=1))
        sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
        # print(sumSqAEx)
     
        SqB = B**2
        sumSqB = np.sum(SqB, axis=1)
        sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))    
        SqED = sumSqBEx + sumSqAEx - 2*vecProd
        SqED[SqED<0]=0.0   
        ED = np.sqrt(SqED)
        # 是个矩阵np.matrix ED=[[0,2,3],[2,0,4]],转为list
        return ED.tolist()

    
    '''计算一条轨迹到 一个轨迹集合中所有轨迹 的距离，并且输出全部排序
    输入： trj list
          trj_list lis[]
    输出： top_k_index  list
          top_k_dist  list'''
    def VD(self, trj, trj_list):
        #[[0.1.1,2.3]]
        dists = self.EuclideanDistances([trj],trj_list)
        # [0.1.1,2.3]
        line = dists[0]
        id_dist_pair = list(enumerate(line))
        # 对第i个点到其他点的距离进行排序
        line.sort()
        id_dist_pair.sort(key=lambda x: x[1])
        # 获取每一个点的top-k索引编号
        top_k_index = [id_dist_pair[i][0] for i in range(len(line))]
        # 获取每一个点的top-k 距离
        top_k_dist = [line[i] for i in range(len(line))]
        return top_k_index, top_k_dist 
    
    '''计算一条轨迹到 一个轨迹集合中所有轨迹 的距离，并且输出top-k排序 
    输入： trj list
          trj_list lis[]
    输出： top_k_index  list
          top_k_dist  list'''
    def top_VD(self, trj, trj_list, K):
        #[0.1.1,2.3]
        dists = self.EuclideanDistances([trj],trj_list)
        # # [0.1.1,2.3]
        line = dists[0]
        id_dist_pair = list(enumerate(line))
        # 对第i个点到其他点的距离进行排序
        line.sort()
        id_dist_pair.sort(key=lambda x: x[1])
        # 获取每一个点的top-k索引编号
        top_k_index = [id_dist_pair[i][0] for i in range(K)]
        # 获取每一个点的top-k 距离
        top_k_dist = [line[i] for i in range(K)]
        return top_k_index, top_k_dist 
    
    '''对于小的向量集合计算距离并获得全部排序，n<10K
    输入： vec_list1 list[]
          vec_list2 lis[]
    输出： rank_V  list
          rank_D  list'''
    def get_rank_vd(self,vec_list1, vec_list2):
        self.Dists = self.EuclideanDistances(vec_list1, vec_list2)
        # 根据Dists，得到rank_V,rank_D
        rank_V = []
        rank_D = []
        for i in range(len(self.Dists)):
            # 读取每一行的距离
            line = self.Dists[i] #.loc[i].tolist()
            id_dist_pair = list(enumerate(line))
            # 对第i个点到其他点的距离进行排序
            line.sort()
            id_dist_pair.sort(key=lambda x: x[1])
            # 获取每一个点的top-k索引编号
            top_k_index = [id_dist_pair[i][0] for i in range(len(line))]
            rank_V.append(top_k_index)
            # 获取每一个点的top-k 距离
            top_k_dist = [line[i] for i in range(len(line))]
            rank_D.append(top_k_dist)
        return rank_V, rank_D
        
    
    '''对于特别大的向量集合，计算距离需要一一计算,只返回每个的top_k排序V,D
    中的top-k
    输入： vec_list1 list[]
          vec_list2 lis[]
          K         int
    输出： top_k_indexs  list
          top_k_dists  list'''
    def get_top_vd(self, vec_list1, vec_list2, K):
        top_k_indexs = []
        top_k_dists = []
        for each_trj in vec_list1:
            top_k_index, top_k_dist = self.top_VD(each_trj, vec_list2, K)
            top_k_indexs.append(top_k_index)
            top_k_dists.append(top_k_dist)
        return top_k_indexs, top_k_dists
    

    ''' 评价方法1： 计算两组 trg, src轨迹内一个点 的top-k, 取并集计算这个并集的rank, 计算两个rank的差异性
    输入：两组轨迹的rank V1 V2 Dataframe结构
          需要比较的前K各邻居 K
    输出：nn_rank1 第一个rank list
          nn_rank2 第二个rank list 
          rank 两个rank的相关系数'''
    def get_nn_rank(self,V1, V2, K):
        knn_v1 = V1[0:K]
        knn_v2 = V2[0:K]
        # K-nn union
        nn_union = list(set(knn_v1).union(set(knn_v2)))
        # 计算union 在V1， V2中的rank
        nn_rank1 = [V1.index(ii) for ii in nn_union]
        nn_rank2 = [V2.index(ii) for ii in nn_union]
        # 计算相关系数
        corr = np.corrcoef(nn_rank1,nn_rank2)[0,1]
        return nn_rank1, nn_rank2, corr

    
    def c1(self,K=10):
        # 1. 获取trg 和src 轨迹向量代表
        if self.m0 != 0:
            trg_vec = self.convert.t2vec(self.m0,'corr.trg')
            src_vec = self.t2vec(self.m0,'corr.src')
        else:
            trg_vec = self.convert.t2vec_noModel('corr.trg')
            src_vec = self.convert.t2vec_noModel('corr.src')
        # 2. 计算trg,src 两两向量间的距离，返回trg_rank_V, src_rank_V
        # trg_rank_V, _ = self.EuclideanDistances(trg_vec[i],trg_vec)
        # src_rank_V, _ = self.EuclideanDistances(src_vec[i],src_vec)
        # print(src_rank_V)
        
        
        ## 计算平均的相关性
        ## 3.每次选定一个向量i, 计算其top_k邻居的在src和target中的rank
        corrs = []
        for i in range(len(trg_vec)):
            trg_rank_V, _ = self.VD(trg_vec[i],trg_vec)
            src_rank_V, _ = self.VD(src_vec[i],src_vec)
            rank_1, rank_2, corr = self.get_nn_rank(src_rank_V, trg_rank_V, K)
            corrs.append(corr)
        return rank_1, rank_2, sum(corrs)/len(corrs)


    '''第二种评价方式-计算平均rank
       输入： Dq_0, Dq_1, Dp_0, Dp_1 对应四个轨迹集的 代表向量集合
             K -匹配近邻数
       输出：平均rank
    '''
    # def c21(self, Dq_0, Dq_1, Dp_0, Dp_1, Q_size, P_size):
    def c21(self, Dq_0, Dq_1, Dp_0, Dp_1):
        # 截取，获取向量表达
        # Dq_0,Dq_1 = Dq_0[0:Q_size], Dq_1[0:Q_size]
        # Dp_0,Dp_1 = Dp_0[0:P_size], Dp_1[0:P_size]
        if self.m0 == 0:
            Dq_0, Dq_1, Dp_0, Dp_1 = self.convert.t2vec_input_noModel(Dq_0),\
                                      self.convert.t2vec_input_noModel(Dq_1),\
                                      self.convert.t2vec_input_noModel(Dp_0),\
                                      self.convert.t2vec_input_noModel(Dp_1)
        else:
            Dq_0, Dq_1, Dp_0, Dp_1 = self.convert.t2vec_input(Dq_0, self.m0),\
                                      self.convert.t2vec_input(Dq_1, self.m0),\
                                      self.convert.t2vec_input(Dp_0, self.m0),\
                                      self.convert.t2vec_input(Dp_1, self.m0)
        all_ranks = []
        DD = np.append(np.array(Dq_1),np.array(Dp_1),axis=0)
        ## 处理较小向量间 mean_rank 直接进行两个矩阵运算，内存消耗大，运算较快 5000-5000 30s
        if (len(Dq_0) <= 3000):
            # 对于Dq_0中的每条轨迹，获取其与Dq_1 U Dp_1 的top_k
            self.rank_V, self.rank_D = self.get_rank_vd(Dq_0,DD.tolist())
            # 对于每一个在 Dq_1中的轨迹，计算其在rank_V中的排名
            for ii in range(len(Dq_0)):
                all_ranks.append(self.rank_V[ii].index(ii))
        ## 处理巨大向量间 mean_rank   mean_rank= 13.98
        ## 用一个向量计算到其他向量时间，内存消耗小，运算较慢
        else:
            # 对于每一个在 Dq_1中的轨迹，计算其在rank_V中的排名
            for n in tqdm(range(len(Dq_0))):
                # 得到Q中第i条轨迹到DD(P+Q)中所有轨迹的排名
                vn, dn = self.VD(Dq_0[n],DD)
                rank_i = vn.index(n)
                all_ranks.append(rank_i)
        return sum(all_ranks)/len(all_ranks)
    
    
    ''' 第二种评价方式 mean_rank'''
    # 展示在不同P,Q nums下mean_rank的指标，此时r1=r2=0
    def mean_ranks_vary_nums(self):
        print("计算不同P_nums, Q_nums下的mena_rank")
        # Q_sizes = [10000,10000,10000,10000,10000]
        P_sizes = [10000,20000,30000,40000,50000]
        for size in P_sizes:
            # 划分P，Q
            # 独立实验次数
            mean_ranks = []
            for ii in range(self.args.exp_nums):
                Q_size, P_size, r1, r2 = 10000,size, 0, 0
                self.dataloader.splitPQ(Q_size, P_size, r1,r2)
                # 获取利用 kdtress在该P,Q下的mean_rank
                mean_rank = self.get_mean_rank()
                mean_ranks.append(mean_rank)
            mean_rank = np.mean(mean_ranks)
            # 获取利用 遍历排序索引 下 在该P,Q下的mean_rank
            # Dq_0, Dq_1, Dp_0, Dp_1 = self.dataloader.Q0, self.dataloader.Q1, self.dataloader.P0,  self.dataloader.P1
            # mean_rank = self.c21(Dq_0, Dq_1, Dp_0, Dp_1)
            print("mean_rank={} @P_size={} @time={}".format(round(mean_rank,5),P_size,time.ctime()))
            
    
    # 展示在P,Q不同 变化 r1下的mean_rank
    def mean_ranks_vary_r1(self):
        print("\n在P,Q不同 变化 r1下的mean_rank")
        r1s = [0.1,0.2,0.3,0.4,0.5]
        for r in r1s:
            # 划分P，Q
            # 独立实验次数
            mean_ranks = []
            for ii in range(self.args.exp_nums):
                Q_size, P_size, r1, r2 = 10000,50000, r, 0
                self.dataloader.splitPQ(Q_size, P_size, r1,r2)
                # 获取在该P,Q下的mean_rank
                mean_rank = self.get_mean_rank()
                mean_ranks.append(mean_rank)
            mean_rank = np.mean(mean_ranks)
            # 获取利用 遍历排序索引 下 在该P,Q下的mean_rank
            # Dq_0, Dq_1, Dp_0, Dp_1 = self.dataloader.Q0, self.dataloader.Q1, self.dataloader.P0,  self.dataloader.P1
            # mean_rank = self.c21(Dq_0, Dq_1, Dp_0, Dp_1)
            print("mean_rank={} @r1={} @time={}".format(round(mean_rank,5),r1,time.ctime()))
            
    
    # 展示在P,Q不同 变化 r1下的mean_rank
    def mean_ranks_vary_r2(self):
        print("\n在P,Q不同 变化 r2下的mean_rank")
        r2s = [0.1,0.2,0.3,0.4,0.5]
        for r in r2s:
            # 划分P，Q
            # 独立实验次数
            mean_ranks = []
            for ii in range(self.args.exp_nums):
                Q_size, P_size, r1, r2 = 10000,50000, 0, r
                self.dataloader.splitPQ(Q_size, P_size, r1,r2)
                # 获取利用 kdtress在该P,Q下的mean_rank
                mean_rank = self.get_mean_rank()
                mean_ranks.append(mean_rank)
            mean_rank = np.mean(mean_ranks)
            # 获取利用 遍历排序索引 下 在该P,Q下的mean_rank
            # Dq_0, Dq_1, Dp_0, Dp_1 = self.dataloader.Q0, self.dataloader.Q1, self.dataloader.P0,  self.dataloader.P1
            # mean_rank = self.c21(Dq_0, Dq_1, Dp_0, Dp_1)
            print("mean_rank={} @r2={} @time={}".format(round(mean_rank,5), r2, time.ctime() ))
        
    
    
    def get_mean_rank(self):
        Dq_0, Dq_1, Dp_0, Dp_1 = self.dataloader.Q0, self.dataloader.Q1, self.dataloader.P0,  self.dataloader.P1
        # 获取向量表达
        if self.m0 == 0:
            Dq_0, Dq_1, Dp_0, Dp_1 = self.convert.t2vec_input_noModel(Dq_0),\
                                      self.convert.t2vec_input_noModel(Dq_1),\
                                      self.convert.t2vec_input_noModel(Dp_0),\
                                      self.convert.t2vec_input_noModel(Dp_1)
        else:
            Dq_0, Dq_1, Dp_0, Dp_1 = self.convert.t2vec_input(Dq_0, self.m0),\
                                      self.convert.t2vec_input(Dq_1, self.m0),\
                                      self.convert.t2vec_input(Dp_0, self.m0),\
                                      self.convert.t2vec_input(Dp_1, self.m0)
        '''create query and the database'''
        query = np.array(Dq_0)
        db = np.append(np.array(Dq_1),np.array(Dp_1),axis=0)
        ''' for each trj in the query, get the rank of the twins in db'''
        tree = spatial.KDTree(data=db)
        all_ranks = []
        max_rank = 3000
        _, ranks = tree.query(query,max_rank)
        for ii in range(len(query)):
            try:
                index = ranks[ii].tolist().index(ii)
            except:
                index = max_rank
            all_ranks.append(index)
        return sum(all_ranks)/len(all_ranks)
    
    '''第三种评价方式 cross-similarity'''
    # 变化r1下的cs
    def show_cs_vary_r1(self):
        print("\n变化r1下的cs")
        # r1s = [0.2,0.4,0.6]
        r1s = [0.1,0.2,0.3,0.4,0.5]
        # css = []
        for r in r1s:
            # 独立实验次数
            css = []
            for ii in range(self.args.exp_nums):
                trj_num, r1, r2 = 10000, r, 0
                self.dataloader.splitTbTa(trj_num, r1, r2)
                cs = self.get_cross_similarity()
                css.append(cs)
            cs = np.mean(css)
            print("cs={} @r1={} @time={}".format(round(cs,5),r1,time.ctime())) 
            # css.append(cs)
        # return css
            
    # 变化r2下的cs
    def show_cs_vary_r2(self):
        print("\n变化r2下的cs")
        # r2s = [0.2,0.4,0.6]
        r2s = [0.1,0.2,0.3,0.4,0.5]
        for r in r2s:
            # 独立实验次数
            css = []
            for ii in range(self.args.exp_nums):
                trj_num, r1, r2 = 10000, 0,r
                self.dataloader.splitTbTa(trj_num, r1, r2)
                cs = self.get_cross_similarity()
                css.append(cs)
            cs = np.mean(css)
            print("cs={} @r2={} @time={}".format(round(cs,5),r2,time.ctime()))
    
    # 获取一次cs
    def get_cross_similarity(self):
        Tb1, Tb2, Ta1, Ta2 = self.dataloader.Tb1, self.dataloader.Tb2, self.dataloader.Ta1, self.dataloader.Ta2
        if self.m0 == 0:
            Tb1, Tb2, Ta1, Ta2 = self.convert.t2vec_input_noModel(Tb1),\
                                self.convert.t2vec_input_noModel(Tb2),\
                                self.convert.t2vec_input_noModel(Ta1),\
                                self.convert.t2vec_input_noModel(Ta2)
        else:
            Tb1, Tb2, Ta1, Ta2 = self.convert.t2vec_input(Tb1, self.m0),\
                                self.convert.t2vec_input(Tb2, self.m0),\
                                self.convert.t2vec_input(Ta1, self.m0),\
                                self.convert.t2vec_input(Ta2, self.m0)
        cs = self.cross_similarity(Tb1, Tb2, Ta1, Ta2)
        return cs

    def cross_similarity(self, vb, vb1, va, va1):
        # 计算欧氏距离
        vb, vb1, va, va1 = np.array(vb),np.array(vb1),np.array(va),np.array(va1)
        diffs = []
        for ii in range(len(vb)):
            dis_b = np.linalg.norm(vb1[ii]-vb[ii])
            dis_a = np.linalg.norm(va1[ii]-va[ii])
            diff = abs(dis_b-dis_a)/dis_b
            if(dis_b>0.1): # 有可能会选到两条相同的轨迹，此时dis_b就会很小，我们 只保留dis_b较大的
                # print(diff)
                # print(dis_b)
                # print(dis_a)
                diffs.append(diff)
        # print(np.mean(diffs))
        # print("\n")
        return np.mean(diffs)
    
    
    
    '''第四种评价方法 KNN'''
    
    #变化r1下的knn准确率
    def show_knn_vary_r1(self):
        print("\n变化r1下的knn准确率")
        r1s = [0.1,0.2,0.3,0.4,0.5]
        # K = [20,40,60,100]
        K = [100,200,300]
        for r in r1s:
            for k in K:
                # 划分query ,database    
                # 独立实验次数
                precisions = []
                for ii in range(self.args.exp_nums):
                    num_q, num_db, r1, r2 = 1000,10000,r,0
                    self.dataloader.split_query_db(num_q, num_db, r1, r2)
                    knn_precision = self.get_knn_precision(k)
                    precisions.append(knn_precision)
                knn_precision = np.mean(precisions)
                print("knn_precision(->1) = {} @k={} @r1={} @time={}".format(round(knn_precision*100,5),k,r1,time.ctime()))
                
    #变化r2下的knn准确率
    def show_knn_vary_r2(self):
        print("\n变化r2下的knn准确率")
        r2s = [0.1,0.2,0.3,0.4,0.5]
        # K = [20,40,60,100]
        K = [100,200,300]
        for r in r2s:
            for k in K:
                # 划分query ,database    
                # 独立实验次数
                precisions = []
                for ii in range(self.args.exp_nums):
                    num_q, num_db, r1, r2 = 1000,10000,0,r
                    self.dataloader.split_query_db(num_q, num_db, r1, r2)
                    knn_precision = self.get_knn_precision(k)
                    precisions.append(knn_precision)
                knn_precision = np.mean(precisions)
                print("knn_precision(->1) = {} @k={} @r2={} @time={}".format(round(knn_precision*100,5),k,r2,time.ctime()))
            
    
    def get_knn_precision(self, K):
        '''
        Parameters
        ----------
        K : TYPE Int
            DESCRIPTION. the number of nn

        Returns
        -------
        precision : TYPE float
            DESCRIPTION. KNN precision

        '''
        q1,q2,db1,db2 = self.dataloader.q1, self.dataloader.q2, self.dataloader.db1, self.dataloader.db2
        # 获得代表表示
        if self.m0 == 0:
            q1,q2,db1,db2 = self.convert.t2vec_input_noModel(q1),\
                            self.convert.t2vec_input_noModel(q2),\
                            self.convert.t2vec_input_noModel(db1),\
                            self.convert.t2vec_input_noModel(db2)
        else:
            q1,q2,db1,db2 = self.convert.t2vec_input(q1, self.m0),\
                            self.convert.t2vec_input(q2, self.m0),\
                            self.convert.t2vec_input(db1, self.m0),\
                            self.convert.t2vec_input(db2, self.m0)
        return self.knnPrecision(q1,db1, q2, db2, K)
                                      
    
    ''' 输入一组query 轨迹向量组和 db轨迹向量组, 返回query在db中的knn编号'''
    def getKnn(self, query, db, K):
        query, db = np.array(query),np.array(db)
        tree = spatial.cKDTree(data=db)
        _, nn_ids = tree.query(query,K)
        return nn_ids
    
    
    ''' 输入两组 nn_ids, 返回其准确率'''
    def topPrecision(self, nn_ids1, nn_ids2, K):
        intersection_nums = []
        for ii in range(len(nn_ids1)):
            intersection = list(set(nn_ids1[ii]).intersection(set(nn_ids2[ii])))
            intersection_nums.append(len(intersection))
        return np.mean(intersection_nums)/K
    
    ''' 输入一组query1，db1以及它们下采样后的query2,db2, 返回其KNN准确率'''
    def knnPrecision(self, q1,db1, q2, db2, K):
        nn_ids1 = self.getKnn(q1,db1,K)
        nn_ids2 = self.getKnn(q2,db2,K)
        return self.topPrecision(nn_ids1, nn_ids2, K)
    
    
'''
测试
'''
if __name__ == "__main__":
    
    
    data = "bj100300"
    print(data)
    scale = 0.001
    time_size = 300
    args = setArgs(data,scale,time_size)
    M = My_criterion(args, 0)
    
    # 划分P，Q
    # Q_size, P_size, r1, r2 = 10000,10000, 0, 0
    # M.dataloader.splitPQ(Q_size, P_size, r1,r2)
    
    # 划分Ta, Tb
    # trj_num, r1, r2 = 10000, 0.1, 0
    # M.dataloader.splitTbTa(trj_num, r1, r2)
    
    # 划分query ,database    
    # num_q, num_db, r1, r2 = 1000,10000,0.1,0
    # M.dataloader.split_query_db(num_q, num_db, r1, r2)
    # print("Test data loaded")
    
    with torch.no_grad():
        # print(time.ctime())
        # ''' test '''
        # mean_rank = M.get_mean_rank()
        # print("mean_rank ={}\n".format(mean_rank))
        
        # cross_similarity = M.get_cross_similarity()
        # print("cross_similarity(->0) = {} \n".format(round(cross_similarity,4)))
        
        # K = 40
        # knn_precision = M.get_knn_precision(K)
        # print("knn_precision(->1) = {} \n".format(knn_precision*100))
        print(time.ctime())
        M.mean_ranks_vary_nums()
        M.mean_ranks_vary_r1()
        
        
        # M.mean_ranks_vary_r2()
        
        M.show_cs_vary_r1()
        # M.show_cs_vary_r2()
        
        M.show_knn_vary_r1()
        # M.show_knn_vary_r2()
    # m = My_criterion(args,0)  
    # #rank_1, rank_2, corr = m.c1()
    # # print("mean_corr: "+str(corr))
    # mean_rank = m.c22(Q0, Q1, P0, P1, 5000,)
    # print("\n mean_rank= "+str(mean_rank))
    
    # a = np.random.randint(100,1000,(20,256))
    # a = a.tolist()
    # # 第一个向量到所有向量的V,D
    # v1, d1 = m.VD(a[0],a)
    # # # 第一个向量到所有向量的top-10 V,D
    # v1_10, d1_10 = m.top_VD(a[0],a,10)
    # ## 得到较小两个向量组 第一个组每个到第二个组的V,D
    # v2, d2 = m.get_rank_vd(a,a)
    # ## 较大向量组 第一个组每个到第二个组的top V,D
    # v2_10, d2_10 = m.get_top_vd(a, a, 5)
    
    