# coding:utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import vgg_train
import utils
from Nclasses import labels

# img_path = raw_input('Input the path and image name:')
img_path = input('Input the path and image name:')
# 用自己定义的load_image对待识别图片进行预处理
img_ready = utils.load_image(img_path)
print("img_ready shape", tf.Session().run(tf.shape(img_ready)))

fig = plt.figure(u"Top-5 预测结果")

with tf.Session() as sess:
    images = tf.placeholder(tf.float32, [1, 224, 224, 3])
    # 实例化vgg
    vgg = vgg_train.Vgg16()  # 运行vgg类的初始化函数，读出保存在npy文件中的模型参数
    # 运行前向传播函数，复现神经网络结构
    vgg.forward(images)
    # 将待识别图像作为输入，喂入计算softmax的节点vgg.prob
    probability = sess.run(vgg.prob, feed_dict={images: img_ready})
    # 将probability列表中，概率最高的5个，所对应的列表索引值，存入top5
    top5 = np.argsort(probability[0])[-1:-6:-1]
    print("top5:", top5)
    values = []  # 新建列表，用来存probability的值
    bar_label = []  # 新建列表，用来存标签列表中对应的值，即物种名称
    for n, i in enumerate(top5):  # 打印键和值
        print("n:", n)
        print("i:", i)
        values.append(probability[0][i])
        bar_label.append(labels[i])
        print(i, ":", labels[i], "----", utils.percent(probability[0][i]))  # 打印每个物种出现的概率

    # 画布一行一列
    ax = fig.add_subplot(111)
    # 绘制柱状图
    ax.bar(range(len(values)), values, tick_label=bar_label, width=0.5, fc='g')
    # 横轴标签
    ax.set_ylabel(u'probabilityit')
    # 标题
    ax.set_title(u'Top-5')
    # 在每个柱子顶端添加预测概率值
    for a, b in zip(range(len(values)), values):
        ax.text(a, b + 0.0005, utils.percent(b), ha='center', va='bottom', fontsize=7)
    plt.show()  # 弹窗显示图像
