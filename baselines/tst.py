# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 21:48:34 2021

@author: likem
"""

from multiprocessing import Manager, Process

def fun(k,result_dict):
    """被测试函数"""
    print('-----fun函数内部，参数为{}----'.format(k))
    m = k + 10

    result_dict.append(m) # 方法二：manger

def my_process():
    """多进程"""

    # 方法二：Manger
    manger = Manager()
    result_dict = manger.list()  # 使用字典
    # result_dict = [None]*10
    jobs = []

    for i in range(10):
        p = Process(target=fun, args=(i, result_dict))
        jobs.append(p)
        p.start()

    for pr in jobs:
        pr.join()
    var = result_dict
    print('返回结果', var)

def main():
    """主程序"""


    # 3. 使用多进程
    my_process()


if __name__ == '__main__':
    main()
