# coding：utf-8
import tensorflow as tf
import numpy as np
import PIL as Image
from pip._vendor.distlib.compat import raw_input

import minist_backward
import minist_forward


def restore_model(testPicArr):
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [None, minist_forward.INPUT_NODE])
        y = minist_forward.forward(x, None)
        preValue = tf.argmax(y, 1)

        variable_averages = tf.train.ExponentialMovingAverage(minist_backward.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(minist_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                preValue = sess.run(preValue,feed_dict={x: testPicArr})
                return preValue
            else:
                print("No checkpoint file found")
                return -1


def pre_pic(pic_name):
    img = Image.open(pic_name)
    reIm = img.resize((28, 28, Image.ANTIALIAS))
    # 转成灰度图像
    im_arr =np.array(reIm.convert('L'))
    threshold = 50
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            if im_arr[i][j] < threshold:
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255
    nm_arr = im_arr.reshape([1, 784])
    nm_arr = nm_arr.astype(np.float32)
    img_ready = np.multiply(nm_arr, 1.0/255.0)

    return img_ready


def application():
    # input 从控制台读入数字
    testNum = input("input the number of test picture: ")
    for i in range(testNum):
        # 用raw_input 从控制台读入字符串
        testPic = raw_input("the path of test picture: ")
        testPicArr = pre_pic(testPic)
        preValue = restore_model(testPicArr)
        print("the prediction number is: ", preValue)


def main():
    application()


if __name__ == "__main__":
    main()