import os
import cv2
import numpy as np

image_dst = '/Users/zouhuanxin/Downloads/UCF-101-Image/'
flow_dst = '/Users/zouhuanxin/Downloads/UCF-101-Flow/'

dictionary = {'ApplyEyeMakeup': 0, 'ApplyLipstick': 1, 'Archery': 2, 'BabyCrawling': 3, 'BalanceBeam': 4,
              'BandMarching': 5, 'BaseballPitch': 6, 'Basketball': 7, 'BasketballDunk': 8, 'BenchPress': 9}


# 读取图片数据
def readImage():
    image_files = os.listdir(image_dst)
    image_files = [item for item in image_files if not item.startswith('.')]

    traindata = []
    trainlabel = []

    # 一个file2（一个动作序列）预测出一个动作，不是一帧图片预测
    # traindata中保存应该是一组img [动作组，每组数据有多少张图片，图片的通道数，图片高度，图片宽度]
    # trainlabel中保存的应该是 [动作组]
    # file1表示动作名称，每一个file1下面有n个相同的数据组
    index = 0
    index1 = 0
    for file1 in image_files:
        index = index + 1
        if index > 1:
            break
        images_child_files1 = os.path.join(image_dst, file1)
        for file2 in os.listdir(images_child_files1):
            index1 = index1 + 1
            if index1 > 5:
                break
            recogition_child = []
            # 得到对应动作的文件夹，并获取对应的所有图片转成数组
            # file2为对应动作名称
            images_child_files2 = os.path.join(images_child_files1, file2)
            for file3 in os.listdir(images_child_files2):
                images_child_files3 = os.path.join(images_child_files2, file3)
                img = cv2.imread(images_child_files3)
                img = np.array(img)
                img = img / 255.0

                recogition_child.append(img)
            traindata.append(recogition_child)
            trainlabel.append(dictionary[file1])

    return traindata, np.array(trainlabel)

# 读取光流数据
