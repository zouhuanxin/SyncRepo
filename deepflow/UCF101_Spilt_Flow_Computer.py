import os
import cv2
import numpy as np
import flow_vis
import sys


# 进度条
# finish_tasks_number: int, 已完成的任务数
# tasks_number: int, 总的任务数
def progress_bar(finish_tasks_number, tasks_number):
    percentage = round(finish_tasks_number / tasks_number * 100)
    print(
        "\r{}/{} 进度:{}%:".format(finish_tasks_number, tasks_number, percentage),
        "▓" * (percentage // 2), end="")
    sys.stdout.flush()


# 计算光流
# src 需要计算的图片集合，应该是一个视频所切割产生的图片集合
# des 计算完的光流信息存储为光流图片保存为新的图片，不同颜色与深度表示当前像素点偏移的情况
def generate_flow(src, des):  # 以个体为单位生成光流图
    """
    :param src: 原图每个个体的文件夹路径
    :param des: 光流图每个个体的文件夹路径
    :return:
    """

    def mkfile(file):
        if not os.path.exists(file):
            os.makedirs(file)

    file_list = os.listdir(src)
    file_list = sorted(file_list)
    # print(file_list)

    # 获取第一帧
    frame1 = cv2.imread(os.path.join(src, file_list[0]))
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)

    # 遍历每一行的第一列
    hsv[..., 1] = 255

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    for file in file_list:
        # for i in range(1440):
        path = os.path.join(src, file)
        frame2 = cv2.imread(path)
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        # 返回一个两通道的光流向量，实际上是每个点的像素位移值
        flow = dis.calc(prvs, next, None)
        # flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 25, 3, 5, 1.2, 1)
        # # print(flow.shape)
        # print(flow)
        # # 笛卡尔坐标转换为极坐标，获得极轴和极角
        # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # hsv[..., 0] = ang * 180 / np.pi / 2
        # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        rgb = flow_vis.flow_to_color(flow, convert_to_bgr=False)
        mkfile(des)
        # 写入新的文件夹
        try:
            cv2.imwrite(os.path.join(des, file), rgb)
        except Exception as e:
            continue

        prvs = next
    # print('img number:%d' % len(os.listdir(des)))


# 把视频解码成一帧帧图片
# src 需要切割的视频，是单独的视频不是集合
# dist 输出文件位置
def spilt_video_to_image(src, dist):
    # 打开视频文件
    video_capture = cv2.VideoCapture(src)  # 替换 'your_video.mp4' 为视频文件的路径

    # 检查视频是否成功打开
    if not video_capture.isOpened():
        print("无法打开视频文件")
        exit()

    frame_number = 0  # 初始化帧计数

    while True:
        # 读取一帧
        ret, frame = video_capture.read()

        if not ret:
            break

        # 为每一帧保存一张图片
        frame_filename = f'frame_{frame_number:04d}.jpg'  # 可以根据需要调整文件名格式
        frame_filename = dist + '/' + frame_filename
        cv2.imwrite(frame_filename, frame)

        frame_number += 1

        # 可以添加其他操作，如显示每一帧
        # cv2.imshow('Frame', frame)

        # 如果要在按下键盘上的 'q' 键时退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放视频捕获对象
    video_capture.release()


# 切割并生成光流
def spilt_flow(video_src, image_dst, flow_dst):
    if not os.path.exists(video_src):
        print('文件夹不存在')
        exit()

    video_file_list = os.listdir(video_src)
    video_file_list = [item for item in video_file_list if not item.startswith('.')]
    video_file_list = sorted(video_file_list)
    index = 0
    sum = len(video_file_list)

    for file1 in video_file_list:
        index = index + 1
        progress_bar(index, sum)
        video_child_file = os.path.join(video_src, file1)
        image_child_dst = os.path.join(image_dst, file1)
        flow_child_dst = os.path.join(flow_dst, file1)
        # 创建文件夹
        if not os.path.exists(image_child_dst):
            os.makedirs(image_child_dst)
        if not os.path.exists(flow_child_dst):
            os.makedirs(flow_child_dst)
        for file2 in os.listdir(video_child_file):  # file2则是最后的文件名
            image_child_dst_folder = os.path.join(image_child_dst, file2.split('.')[0])
            flow_child_dst_folder = os.path.join(flow_child_dst, file2.split('.')[0])
            # 为每一个视频创建一个文件夹保存对应的图片
            if not os.path.exists(image_child_dst_folder):
                os.makedirs(image_child_dst_folder)
            if not os.path.exists(flow_child_dst_folder):
                os.makedirs(flow_child_dst_folder)
            video_file = os.path.join(video_child_file, file2)
            # 切割图片
            spilt_video_to_image(video_file, image_child_dst_folder)
            # 计算光流
            generate_flow(image_child_dst_folder, flow_child_dst_folder)


video_src = '/Users/zouhuanxin/Downloads/UCF-101/'
image_dst = '/Users/zouhuanxin/Downloads/UCF-101-Image/'
flow_dst = '/Users/zouhuanxin/Downloads/UCF-101-Flow/'

spilt_flow(video_src, image_dst, flow_dst)
