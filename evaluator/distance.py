"""
定义一系列的向量距离计算函数
"""
import numpy as np


def euclidean_distances(A, B):
    """
    计算两组 轨迹代表向量集合 中各个向量到各个向量间的距离

    :param A: 向量列表
    :param B: 向量列表
    :return: ED=[[0,2,3],[2,0,4]] list
    """
    A, B = np.array(A), np.array(B)
    BT = B.transpose()
    # vecProd = A * BT
    vecProd = np.dot(A, BT)
    # print(vecProd)
    SqA = A ** 2
    # print(SqA)
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
    # print(sumSqAEx)
    SqB = B ** 2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2 * vecProd
    SqED[SqED < 0] = 0.0
    ED = np.sqrt(SqED)
    # 是个矩阵np.matrix ED=[[0,2,3],[2,0,4]],转为list
    return ED.tolist()


def get_all_neighbors(trj, trj_list):
    """
    计算一条轨迹到 一个轨迹集合中所有轨迹 的距离，并且输出全部排序

    :param trj: 一个轨迹表征向量
    :param trj_list: 一组轨迹表征向量
    :return: 一条轨迹到 一个轨迹集合中所有轨迹 的距离 top_k_dist，并且输出全部排序 top_k_index
    """
    # [[0.1.1,2.3]] 两层列表
    dists = euclidean_distances([trj], trj_list)
    # [0.1.1,2.3] 获取第一个元素
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


def get_top_neighbors(trj, trj_list, K):
    """
    计算一条轨迹到 一个轨迹集合中所有轨迹 的距离，并且输出top-k排序

    :param trj: 一个轨迹表征向量
    :param trj_list: 一组轨迹表征向量
    :param K: 取前几个排序
    :return: 一条轨迹到 一个轨迹集合中所有轨迹 的距离 top_k_dist，并且输出全部排序 top_k_index
    """
    # [0.1.1,2.3]
    dists = euclidean_distances([trj], trj_list)
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


def get_rank_vd(vec_list1, vec_list2):
    """
    对于小的向量集合计算距离并获得全部排序，n<10K

    :param vec_list1:
    :param vec_list2:
    :return: rank_V, rank_D
    """
    Dists = euclidean_distances(vec_list1, vec_list2)
    # 根据Dists，得到rank_V,rank_D
    rank_V = []
    rank_D = []
    for i in range(len(Dists)):
        # 读取每一行的距离
        line = Dists[i]  # .loc[i].tolist()
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


def get_top_vd(vec_list1, vec_list2, K):
    """
    对于特别大的向量集合，计算距离需要一一计算,只返回每个的top_k排序V,D

    :param vec_list1:
    :param vec_list2:
    :param K:
    :return: top_k_indexs, top_k_dists
    """
    top_k_indexs = []
    top_k_dists = []
    for each_trj in vec_list1:
        top_k_index, top_k_dist = get_top_neighbors(each_trj, vec_list2, K)
        top_k_indexs.append(top_k_index)
        top_k_dists.append(top_k_dist)
    return top_k_indexs, top_k_dists


