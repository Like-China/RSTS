# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 21:29:45 2021

@author: likem
"""
import numpy as np
import copy
import time
from tqdm import tqdm

class EDwP:
    ''' 求一个点到一条直线的垂足，三维'''
    def getFootPoint3d(self, point, line_p1, line_p2):
        """
        @point, line_p1, line_p2 : [x, y, z]
        """
        x0 = point[0]
        y0 = point[1]
        z0 = point[2]
    
        x1 = line_p1[0]
        y1 = line_p1[1]
        z1 = line_p1[2]
    
        x2 = line_p2[0]
        y2 = line_p2[1]
        z2 = line_p2[2]
    
        k = -((x1 - x0) * (x2 - x1) + (y1 - y0) * (y2 - y1) + (z1 - z0) * (z2 - z1)) / \
            ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)*1.0
    
        xn = k * (x2 - x1) + x1
        yn = k * (y2 - y1) + y1
        zn = k * (z2 - z1) + z1
    
        return (xn, yn, zn)
    
    ''' 求一个点到一条直线的垂足，二维, 并返回距离'''
    def getFootPoint2d(self, point, line_p1, line_p2):
        """
        @point, line_p1, line_p2 : [x, y]
        """
        flag = 1 # 记录垂足是否在线段上
        x0 = point[0]
        y0 = point[1]
    
        x1 = line_p1[0]
        y1 = line_p1[1]
    
        x2 = line_p2[0]
        y2 = line_p2[1]
        
        ''' 第二个点重合'''
        if (x2 - x1) ** 2 + (y2 - y1) ** 2 == 0:
            flag = 0
            return x2, y2, flag
        k = -((x1 - x0) * (x2 - x1) + (y1 - y0) * (y2 - y1) ) / \
            ((x2 - x1) ** 2 + (y2 - y1) ** 2 )*1.0
        xn = k * (x2 - x1) + x1
        yn = k * (y2 - y1) + y1
        '''若垂足不在该线段上,插入的点就等于第二个点'''
        if (xn>max(x1,x2) or xn<min(x1,x2)):
            xn = x2
            yn = y2
            flag = 0
        # dist = np.linalg.norm(np.array([xn,yn])-np.array([x0,y0]))
        return xn, yn, flag
    
    ''' 插入操作  在T1中插入离 T2第二个点最近的点'''
    def insert(self,T1,T2):
        ''' T1=[a1,a2,a3,...]  T2 = [b1,b2,b3,...] '''
        ''' a1 = (x,y,t) '''
        # if(len(T2)>=2 and len(T1)>=2):
        a1, a2 = T1[0],T1[1]
        b1, b2 = T2[0],T2[1]
        ''' 得到 T2第二个点 在T1 e1上的垂足和距离'''
        x,y, flag = self.getFootPoint2d(b2[0:2],a1[0:2],a2[0:2])
        if flag == 0:
            len_e1 = np.linalg.norm(np.array(a1[0:2])-np.array(a2[0:2]))
            len_e2 = np.linalg.norm(np.array(b1[0:2])-np.array(b2[0:2]))
            converage = len_e1+len_e2
            # rep_cost = np.linalg.norm(np.array(a1)-np.array(b1))+\
            #     np.linalg.norm(np.array(a2)-np.array(b2))
            ''' 将时间维度加上权重'''
            rep_cost = np.linalg.norm(np.array(a1[0:2])-np.array(b1[0:2]))+\
                np.linalg.norm(np.array(a2[0:2])-np.array(b2[0:2])) #+(a1[2]-b1[2]+a2[2]-b2[2])/86400
            T1.remove(T1[0])
        else:
            ''' 计算插入点在 T1 e1上占的长度比例'''
            len_e1 = np.linalg.norm(np.array(a1[0:2])-np.array(a2[0:2]))
            len_e2 = np.linalg.norm(np.array(b1[0:2])-np.array(b2[0:2]))
            len_insert = np.linalg.norm(np.array(a1[0:2])-np.array([x,y]))
            ratio = len_insert / len_e1
            ''' 根据长度比例，得到新插入点的时间'''
            # print(ratio)
            i_t = a1[2]+(a2[2]-a1[2])*ratio
            ''' 得到加上时间后的插入点'''
            p = [x,y,i_t]
            # print(p)
            ''' 计算(a1,b1),(p,b2)这两个点的距离，并在T1中替换第一个点为p, T2中删除第一个点'''
            # rep_cost = np.linalg.norm(np.array(a1)-np.array(b1)) + \
            #     np.linalg.norm(np.array(p)-np.array(b2))
            rep_cost = np.linalg.norm(np.array(a1[0:2])-np.array(b1[0:2]))+\
                np.linalg.norm(np.array(p[0:2])-np.array(b2[0:2]))#+(a1[2]-b1[2]+p[2]-b2[2])/86400
            converage = len_insert+len_e2
            ''' 改变T1的第一个点为插入点'''
            T1[0] = p
        return converage,rep_cost*converage
        # return 0
        
    ''' 递归求解'''
    def edwp(self, T1, T2):
        # print("{}-{}".format(len(T1), len(T2)))
        if(len(T1) == 0 and len(T2) == 0):
            return 0
        elif (len(T1) == 0 or len(T2) == 0):
            return 10000
        elif (len(T1) == 1):
            return 0
            tail_node = T1[0]
            dists = 0
            for ii in range(len(T2)-1):
                dist1 = np.linalg.norm(np.array(tail_node)-np.array(T2[ii])) + \
                    np.linalg.norm(np.array(tail_node)-np.array(T2[ii+1]))
                converage = np.linalg.norm(np.array(T2[ii][0:2])-np.array(T2[ii+1][0:2]))
                dists += dist1*converage
            return dists
        elif (len(T2) == 1):
            return 0
            tail_node = T2[0]
            dists = 0
            for ii in range(len(T1)-1):
                dist1 = np.linalg.norm(np.array(tail_node)-np.array(T1[ii])) + \
                    np.linalg.norm(np.array(tail_node)-np.array(T1[ii+1]))
                converage = np.linalg.norm(np.array(T1[ii][0:2])-np.array(T1[ii+1][0:2]))
                dists += dist1*converage
            return dists
        else:
            return min(self.insert(T1,T2)+self.edwp(T1,T2[1:]),self.insert(T2,T1)+self.edwp(T1[1:],T2))
    
    
    ''' 动态规划求解 EDwP'''
    def dp_EDwP(self, T1, T2):
        ''' 计算T1和T2的长总度'''
        # lenT1 = sum([np.linalg.norm(np.array(T1[ii][0:2])-np.array(T1[ii+1][0:2])) for ii in range(len(T1)-1)])
        # lenT2 = sum([np.linalg.norm(np.array(T2[ii][0:2])-np.array(T2[ii+1][0:2])) for ii in range(len(T2)-1)])
        all_cost = 0
        all_converage = 0
        while(True):
            if(len(T1) <=1):
                a1 = T1[0]
                for ii in range(len(T2)-1):
                    b1, b2 = T2[ii],T2[ii+1]
                    len_e = np.linalg.norm(np.array(b1[0:2])-np.array(b2[0:2]))
                    # rep_cost = np.linalg.norm(np.array(a1)-np.array(b1)) + \
                    # np.linalg.norm(np.array(a1)-np.array(b2))
                    rep_cost = np.linalg.norm(np.array(a1[0:2])-np.array(b1[0:2]))+\
                        np.linalg.norm(np.array(a1[0:2])-np.array(b2[0:2]))
                        #+(a1[2]-b1[2]+a1[2]-b2[2])/86400
                    all_converage += len_e
                    all_cost += rep_cost*len_e
                break
            if(len(T2) <=1):
                a1 = T2[0]
                for ii in range(len(T1)-1):
                    b1, b2 = T1[ii],T1[ii+1]
                    len_e = np.linalg.norm(np.array(b1[0:2])-np.array(b2[0:2]))
                    # rep_cost = np.linalg.norm(np.array(a1)-np.array(b1)) + \
                    # np.linalg.norm(np.array(a1)-np.array(b2))
                    rep_cost = np.linalg.norm(np.array(a1[0:2])-np.array(b1[0:2]))+\
                    np.linalg.norm(np.array(a1[0:2])-np.array(b2[0:2]))
                    #+(a1[2]-b1[2]+a1[2]-b2[2])/86400
                    all_converage += len_e
                    all_cost += rep_cost*len_e
                break
            
            a1, a2 = T1[0],T1[1]
            b1, b2 = T2[0],T2[1]
            ''' 将时间较小的点选出,插入到时间较大的节点中去'''
            if(a2[2] < b2[2]):
                conv,cost = self.insert(T2, T1)
                # T1.remove(T1[0])
                T1 = T1[1:]
            else:
                conv,cost = self.insert(T1, T2)
                # T2.remove(T2[0])
                T2 = T2[1:]
            all_converage += conv
            all_cost += cost
        return all_cost/all_converage #lenT1+lenT2
            
    
if __name__ == "__main__":

    # point = np.array([5,2])
    # line_point1 = np.array([2,2])
    # line_point2 = np.array([3,3])
    # point,dist = getFootPoint2d(point,line_point1,line_point2)
    # T1 = [[2,3,1],[2,5,2],[3,3,3],[4,5,4]]
    # T2 = [[0,5,0],[1,1,1],[2,5,2],[3,3,3]]
    b2 = EDwP()
    # dist = b2.Edwp(T1,T2)
    # insert(T1,T2)
    T1 = [[116.42008999999999, 39.96032, 441.0], [116.41283, 39.95265, 1410.0], [116.47835, 39.913959999999996, 4019.0], [116.47185, 39.90733, 4320.0], [116.45806999999999, 39.94815, 5701.0], [116.47018, 39.95265, 6225.0], [116.48226000000001, 39.97217, 6828.0], [116.46824, 39.94185, 7432.0], [116.46875, 39.93646, 8337.0], [116.46021, 39.95841, 8941.0], [116.48308999999999, 39.98877, 9544.0], [116.50894, 40.012640000000005, 35279.0], [116.4866, 40.00656, 35743.0], [116.4653, 40.00062, 37695.0], [116.45433, 39.98806, 38020.0], [116.43418999999999, 39.926559999999995, 40969.0], [116.43675, 39.93508, 41573.0], [116.43845, 39.93934, 42180.0], [116.42055, 39.904509999999995, 43377.0], [116.47273, 39.92695, 45164.0], [116.44698000000001, 39.922109999999996, 45767.0], [116.41618999999999, 39.92305, 46371.0], [116.39283999999999, 39.922, 46673.0], [116.37024, 39.908609999999996, 47276.0], [116.30365, 39.9046, 47880.0]]
    T2 = [[116.4226, 39.96092, 743.0], [116.4292, 39.88944, 2958.0], [116.47185, 39.90733, 4320.0], [116.49298999999999, 39.95841, 5144.0], [116.46586, 39.948479999999996, 5923.0], [116.48383999999999, 39.96839, 6526.0], [116.47873999999999, 39.95558, 7130.0], [116.46866000000001, 39.9364, 8035.0], [116.46841, 39.94123, 8639.0], [116.4777, 39.97484, 9242.0], [116.5157, 40.021679999999996, 9846.0], [116.48811, 40.00525, 35581.0], [116.486, 40.0067, 37393.0], [116.45575, 39.98833, 37997.0], [116.43233000000001, 39.98082, 38322.0], [116.43055, 39.93002, 41271.0], [116.44253, 39.9369, 41869.0], [116.42823999999999, 39.94085, 42482.0], [116.47623999999999, 39.926809999999996, 44890.0], [116.45629, 39.92198, 45466.0], [116.43348999999999, 39.922959999999996, 46069.0], [116.41618999999999, 39.92305, 46371.0], [116.37558, 39.91963, 46975.0], [116.33209, 39.90625, 47578.0], [116.30791, 39.91992, 48483.0]]
    
    t1 = time.time()
    for ii in range(10000):
        all_cost = b2.dp_EDwP(copy.deepcopy(T1), copy.deepcopy(T2))
    print(round(time.time()-t1))
