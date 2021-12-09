"""
Created on Mon Dec 28 10:54:27 2020

@author: likem
"""
# -*- coding: utf-8 -*-
import numpy as np
import os
from tqdm import tqdm
from scipy import spatial
from loader.data_scaner import DataOrderScaner
from loader.data_utils import pad_arrays_keep_invp
from trainer.train import EncoderDecoder
import torch
import time
import settings
from settings import set_args


# 输入轨迹向量, 利用训练好的模型得到一组代表向量
class Trj2vec:

    def __init__(self, args, m0):
        self.args = args
        self.m0 = m0
    
    def t2vec0(self, trj_file_path):
        """
        读取trj.t中的轨迹，返回最后一层输出作为向量表示, 函数内部自动加载模型

        :param trj_file_path: 需要转换为向量表示的轨迹文件
        :return: decoder的最后一层输出，
          为该组轨迹的向量表示 batch*向量维度 格式：列表

        """
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
                # (num_layers, batch, hidden_size * num_directions) 【3，10，256】
                h = m0.encoder_hn2decoder_h0(h)
                # (batch, num_layers, hidden_size * num_directions) 【10，3，256】
                h = h.transpose(0, 1).contiguous()
                # (batch, *)
                #h = h.view(h.size(0), -1)
                vecs.append(h[invp].cpu().data)
            # (num_seqs, num_layers, hidden_size * num_directions)
            
            vecs = torch.cat(vecs) # [10,3,256]
            # (num_layers, num_seqs, hidden_size * num_directions)
            vecs = vecs.transpose(0, 1).contiguous()  # [3,10,256]
        else:
            print("=> no checkpoint found at '{}'".format(args.checkpoint))
        # 返回最后一层作为该批次 轨迹的代表
        return vecs[m0.num_layers-1].squeeze(0).numpy().tolist()
    
    def t2vec(self, m0, trj_file_path):
        """
        读取trj.t中的轨迹，需要加载训练模型
        
        :param m0: 需要加载的训练模型
        :param trj_file_path: 需要转换为向量表示的轨迹文件
        :return: 最后一层输出作为向量表示
        """
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
            # (num_layers, batch, hidden_size * num_directions) 【3，10，256】
            h = m0.encoder_hn2decoder_h0(h)
            # (batch, num_layers, hidden_size * num_directions) 【10，3，256】
            h = h.transpose(0, 1).contiguous()
            # (batch, *)
            #h = h.view(h.size(0), -1)
            vecs.append(h[invp].cpu().data)
        # (num_seqs, num_layers, hidden_size * num_directions)
        
        vecs = torch.cat(vecs) # [10,3,256]
        # (num_layers, num_seqs, hidden_size * num_directions)
        vecs = vecs.transpose(0, 1).contiguous()  # [3,10,256]
        # 返回最后一层作为该批次 轨迹的代表
        return vecs[m0.num_layers-1].squeeze(0).numpy().tolist()

    def t2vec_input(self, T, m0):
        """
        给定一组轨迹向量T, 训练模型m0, 返回一批向量表征

        :param T: 一组轨迹向量
        :param m0:训练模型
        :return: 一批向量表征
        """
        args = self.args
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
            # (num_layers, batch, hidden_size * num_directions) 【3，10，256】
            h = m0.encoder_hn2decoder_h0(h)
            # (batch, num_layers, hidden_size * num_directions) 【10，3，256】
            h = h.transpose(0, 1).contiguous()
            # (batch, *)
            # h = h.view(h.size(0), -1)
            vecs.append(h[invp].cpu().data)
        # (num_seqs, num_layers, hidden_size * num_directions)
        
        vecs = torch.cat(vecs)  # [10,3,256]
        # # 存储三层 输出的隐藏层结构，每一层是 batch个256维的向量
        # (num_layers, num_seqs, hidden_size * num_directions)
        vecs = vecs.transpose(0, 1).contiguous()  # [3,10,256]
        return vecs[m0.num_layers-1].squeeze(0).numpy().tolist()

    def t2vec_input0(self, T):
        """
        给定一组轨迹向量T,  返回一批向量表征, 不需要手动加载训练模型

        :param T: 一组轨迹向量
        :return: 一批向量表征
        """
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
            # (num_layers, batch, hidden_size * num_directions) 【3，10，256】
            h = m0.encoder_hn2decoder_h0(h)
            # (batch, num_layers, hidden_size * num_directions) 【10，3，256】
            h = h.transpose(0, 1).contiguous()
            # (batch, *)
            #h = h.view(h.size(0), -1)
            vecs.append(h[invp].cpu().data)
        # (num_seqs, num_layers, hidden_size * num_directions)
        
        vecs = torch.cat(vecs) # [10,3,256]
        # # 存储三层 输出的隐藏层结构，每一层是 batch个256维的向量
        # (num_layers, num_seqs, hidden_size * num_directions)
        vecs = vecs.transpose(0, 1).contiguous()  # [3,10,256]
        return vecs[m0.num_layers-1].squeeze(0).numpy().tolist()


class Criterion:
    
    def __init__(self, args, m0):
        self.args = args
        self.m0 = m0
        self.convert = Trj2vec(args, m0)

    # 1. 比较mean rank 的表现
    def mean_rank(self, Dq_0, Dq_1, Dp_0, Dp_1):
        all_ranks = []
        DD = np.append(np.array(Dq_1), np.array(Dp_1), axis=0)
        # 处理较小向量间 mean_rank 直接进行两个矩阵运算，内存消耗大，运算较快 5000-5000 30s
        # 处理巨大向量间, 用一个向量计算到其他向量时间，内存消耗小，但运算较慢
        if len(Dq_0) <= 3000:
            # 对于Dq_0中的每条轨迹，获取其与Dq_1 U Dp_1 的top_k
            self.rank_V, self.rank_D = self.get_rank_vd(Dq_0, DD.tolist())
            # 对于每一个在 Dq_1中的轨迹，计算其在rank_V中的排名
            for ii in range(len(Dq_0)):
                all_ranks.append(self.rank_V[ii].index(ii))
        else:
            # 对于每一个在 Dq_1中的轨迹，计算其在rank_V中的排名
            for n in tqdm(range(len(Dq_0))):
                # 得到Q中第i条轨迹到DD(P+Q)中所有轨迹的排名
                vn, dn = self.VD(Dq_0[n],DD)
                rank_i = vn.index(n)
                all_ranks.append(rank_i)
        return sum(all_ranks)/len(all_ranks)

    # 在不同P,Q nums下mean_rank的指标，此时r1=r2=0
    def mean_ranks_vary_nums(self):
        print("计算不同P_nums, Q_nums下的mena_rank")
        # Q_sizes = [10000,10000,10000,10000,10000]
        P_sizes = [10000,20000,30000,40000,50000]
        for size in P_sizes:
            mean_ranks = []
            for ii in range(self.args.exp_nums):
                Q_size, P_size, r1, r2 = 10000,size, 0, 0
                self.dataloader.splitPQ(Q_size, P_size, r1,r2)
                # 获取利用 kdtress在该P,Q下的mean_rank
                mean_rank = self.mean_rank_by_tree()
                mean_ranks.append(mean_rank)
            mean_rank = np.mean(mean_ranks)
            # 获取利用 遍历排序索引 下 在该P,Q下的mean_rank
            # Dq_0, Dq_1, Dp_0, Dp_1 = self.dataloader.Q0, self.dataloader.Q1, self.dataloader.P0,  self.dataloader.P1
            # mean_rank = self.mean_rank(Dq_0, Dq_1, Dp_0, Dp_1)
            print("mean_rank={} @P_size={} @time={}".format(round(mean_rank, 5), P_size, time.ctime()))

    def mean_ranks_vary_r1(self):
        print("\n在P,Q不同 变化 r1下的mean_rank")
        r1s = [0.1,0.2,0.3,0.4,0.5]
        for r in r1s:
            mean_ranks = []
            for ii in range(self.args.exp_nums):
                Q_size, P_size, r1, r2 = 10000,50000, r, 0
                self.dataloader.splitPQ(Q_size, P_size, r1,r2)
                # 获取在该P,Q下的mean_rank
                mean_rank = self.mean_rank_by_tree()
                mean_ranks.append(mean_rank)
            mean_rank = np.mean(mean_ranks)
            # 获取利用 遍历排序索引 下 在该P,Q下的mean_rank
            # Dq_0, Dq_1, Dp_0, Dp_1 = self.dataloader.Q0, self.dataloader.Q1, self.dataloader.P0,  self.dataloader.P1
            # mean_rank = self.mean_rank(Dq_0, Dq_1, Dp_0, Dp_1)
            print("mean_rank={} @r1={} @time={}".format(round(mean_rank,5),r1,time.ctime()))

    def mean_ranks_vary_r2(self):
        print("\n在P,Q不同 变化 r2下的mean_rank")
        r2s = [0.1,0.2,0.3,0.4,0.5]
        for r in r2s:
            mean_ranks = []
            for ii in range(self.args.exp_nums):
                Q_size, P_size, r1, r2 = 10000,50000, 0, r
                self.dataloader.splitPQ(Q_size, P_size, r1,r2)
                # 获取利用 kdtress在该P,Q下的mean_rank
                mean_rank = self.mean_rank_by_tree()
                mean_ranks.append(mean_rank)
            mean_rank = np.mean(mean_ranks)
            # 获取利用 遍历排序索引 下 在该P,Q下的mean_rank
            # Dq_0, Dq_1, Dp_0, Dp_1 = self.dataloader.Q0, self.dataloader.Q1, self.dataloader.P0,  self.dataloader.P1
            # mean_rank = self.mean_rank(Dq_0, Dq_1, Dp_0, Dp_1)
            print("mean_rank={} @r2={} @time={}".format(round(mean_rank,5), r2, time.ctime() ))
        
    def mean_rank_by_tree(self, Dq_0, Dq_1, Dp_0, Dp_1):
        # 利用索引来计算mean_rank
        query = np.array(Dq_0)
        db = np.append(np.array(Dq_1), np.array(Dp_1), axis=0)
        ''' for each trj in the query, get the rank of the twins in db'''
        tree = spatial.KDTree(data=db)
        all_ranks = []
        max_rank = 3000
        _, ranks = tree.query(query, max_rank)
        for ii in range(len(query)):
            try:
                index = ranks[ii].tolist().index(ii)
            except:
                index = max_rank
            all_ranks.append(index)
        return sum(all_ranks)/len(all_ranks)
    
    # 2. 比较 cross-similarity 的表现
    def show_cs_vary_r1(self):
        print("\n变化r1下的cs")
        for r in [0.1, 0.2, 0.3, 0.4, 0.5]:
            # 独立实验次数
            css = []
            for ii in range(self.args.exp_nums):
                trj_num, r1, r2 = 10000, r, 0
                self.dataloader.splitTbTa(trj_num, r1, r2)
                cs = self.get_cross_similarity()
                css.append(cs)
            cs = np.mean(css)
            print("cs={} @r1={} @time={}".format(round(cs, 5), r1, time.ctime()))

    def show_cs_vary_r2(self):
        print("\n变化r2下的cs")
        for r in [0.1, 0.2, 0.3, 0.4, 0.5]:
            css = []
            for ii in range(self.args.exp_nums):
                trj_num, r1, r2 = 10000, 0,r
                self.dataloader.splitTbTa(trj_num, r1, r2)
                cs = self.get_cross_similarity()
                css.append(cs)
            cs = np.mean(css)
            print("cs={} @r2={} @time={}".format(round(cs,5),r2,time.ctime()))

    def cross_similarity(self, Tb1, Tb2, Ta1, Ta2):
        vb, vb1, va, va1 = Tb1, Tb2, Ta1, Ta2
        # 计算欧氏距离
        vb, vb1, va, va1 = np.array(vb), np.array(vb1), np.array(va), np.array(va1)
        diffs = []
        for ii in range(len(vb)):
            dis_b = np.linalg.norm(vb1[ii]-vb[ii])
            dis_a = np.linalg.norm(va1[ii]-va[ii])
            diff = abs(dis_b-dis_a)/dis_b
            if dis_b>0.1: # 有可能会选到两条相同的轨迹，此时dis_b就会很小，我们 只保留dis_b较大的
                # print(diff)
                # print(dis_b)
                # print(dis_a)
                diffs.append(diff)
        # print(np.mean(diffs))
        # print("\n")
        return np.mean(diffs)

    # 3. 比较 KNN 的表现
    def show_knn_vary_r1(self):
        print("\n变化r1下的knn准确率")
        for r in [0.1, 0.2, 0.3, 0.4, 0.5]:
            for k in [100,200,300]:
                precisions = []
                for ii in range(self.args.exp_nums):
                    num_q, num_db, r1, r2 = 1000,10000,r,0
                    self.dataloader.split_query_db(num_q, num_db, r1, r2)
                    knn_precision = self.get_knn_precision(k)
                    precisions.append(knn_precision)
                knn_precision = np.mean(precisions)
                print("knn_precision(->1) = {} @k={} @r1={} @time={}".format(round(knn_precision*100, 5), k, r1, time.ctime()))
                
    def show_knn_vary_r2(self):
        print("\n变化r2下的knn准确率")
        for r in [0.1, 0.2, 0.3, 0.4, 0.5]:
            for k in [100, 200, 300]:
                # 划分query ,database    
                # 独立实验次数
                precisions = []
                for ii in range(self.args.exp_nums):
                    num_q, num_db, r1, r2 = 1000, 10000, 0, r
                    self.dataloader.split_query_db(num_q, num_db, r1, r2)
                    knn_precision = self.get_knn_precision(k)
                    precisions.append(knn_precision)
                knn_precision = np.mean(precisions)
                print("knn_precision(->1) = {} @k={} @r2={} @time={}".format(round(knn_precision*100,5),k,r2,time.ctime()))

    def get_knn(self, query, db, K):
        # 输入一组query 轨迹向量组和 db轨迹向量组, 返回query在db中的knn编号
        query, db = np.array(query), np.array(db)
        tree = spatial.cKDTree(data=db)
        _, nn_ids = tree.query(query, K)
        return nn_ids

    def get_knn_precision(self, q1, db1, q2, db2, K):
        # 输入一组query1，db1以及它们下采样后的query2,db2, 获取对应的knn ids
        nn_ids1 = self.get_knn(q1, db1, K)
        nn_ids2 = self.get_knn(q2, db2, K)
        # 获取两组knn ids的交集个数百分比 即 knn 准确率
        intersection_nums = []
        for ii in range(len(nn_ids1)):
            intersection = list(set(nn_ids1[ii]).intersection(set(nn_ids2[ii])))
            intersection_nums.append(len(intersection))
        return np.mean(intersection_nums) / K


if __name__ == "__main__":
    data = "bj100300"
    print(data)
    scale = 0.001
    time_size = 300
    args = setArgs(data,scale,time_size)
    M = Criterion(args, 0)
    
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
        # mean_rank = M.mean_rank_by_tree()
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
    # m = Criterion(args,0)  
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
    # # 得到较小两个向量组 第一个组每个到第二个组的V,D
    # v2, d2 = m.get_rank_vd(a,a)
    # # 较大向量组 第一个组每个到第二个组的top V,D
    # v2_10, d2_10 = m.get_top_vd(a, a, 5)