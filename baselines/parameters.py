# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 19:28:20 2021

@author: likem
"""

process_num = 40
time_w = 0.5
xyDiff = 0.005
timeDiff = 1500
longs = [116.1,116.7]
lats = [40.1,40.7]

''' mean_rank test'''
# 固定Q的数目下，P数目变化的影响
P_sizes = [1000,2000,3000,4000,5000]
r1s = [0.1,0.2,0.3,0.4,0.5]
r2s = [0.1,0.2,0.3,0.4,0.5]
Ks = [100,200,300,400]
