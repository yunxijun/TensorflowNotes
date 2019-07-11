# coding:utf-8
import tensorflow as tf
import numpy as np

test = np.array([[1, 2, 3], [2, 3, 4], [5, 4, 3], [8, 7, 2]])
# axis：0:按照行比较
# axis：1：按照列比较
axis_0= np.argmax(test,0)
axis_1= np.argmax(test,1)

print(test)
print(axis_0)
print(axis_1)