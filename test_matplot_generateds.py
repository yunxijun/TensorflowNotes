# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
seed =2

def generateds():
    # 基于seed产生随机数
    rdm = np.random.RandomState(seed)
    X = rdm.randn(300,2)
    # 列表带逗号,标准答案
    Y_ = [int(x0 * x0 + x1 * x1 < 2) for (x0, x1) in X]
    # 遍历Y中的所有元素，1赋值给‘red’，0赋值给‘blue’
    # 对应颜色
    Y_c = [['red' if y else 'blue'] for y in Y_]
    print(Y_)
    print(X)

    # reshape 对数据集X和Y进行处理，第一个元素为-1 表示，
    # 随第二个参数计算得到。第二个元素表示多少列，把X整理成n行2列，Y整理成n行1列
    # 按行堆叠
    X = np.vstack(X).reshape(-1, 2)
    Y_ = np.vstack(Y_).reshape(-1, 1)

    return X,Y_,Y_c

