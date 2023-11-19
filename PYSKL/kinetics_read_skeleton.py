from PreprocessPoseData import *
import os
from utils_zhx import *
from BeanData import *
import json

def read_skeleton(data_path, label_path='', dataset='kinetics'):
    # 获取到所有到json文件名称，每一个json文件表示一个视频
    global coordinate
    sample_names = os.listdir(data_path)
    sample_name_len = len(sample_names)
    for index in range(sample_name_len):
        progress_bar(index, sample_name_len)
        sample_name = sample_names[index]
        if sample_name != '-a24wxdPP84.json':
            continue
        sample_path = os.path.join(data_path, sample_name)
        with open(sample_path, 'r') as f:
            video_info = json.load(f)
        for frame_info in video_info['data']:
            frame_index = frame_info['frame_index']
            coordinates = []
            for m, skeleton_info in enumerate(frame_info["skeleton"]):
                pose = skeleton_info['pose']
                score = skeleton_info['score']
                persondata = PersonData(m)
                j = 0
                for i in range(0, len(pose), 2):
                    coord = Coordinate(pose[i] * height, pose[i + 1] * width, score[j])
                    persondata.append(coord)
                    j = j + 1
                coordinates.append(persondata)
            if len(coordinates) != 0:
                # 开始绘制热力图
                save_path = './tmp/{}'.format(sample_name.replace('.json', ''))
                if os.path.exists(save_path) is False:
                    os.mkdir(save_path)
                drawHotImage(coordinates, createImage(), video_info['label'], save_path, dataset)


def load_data(data_path, label_path='', dataset='kinetics'):
    read_skeleton(data_path, label_path, dataset)


read_skeleton('/Users/zouhuanxin/Downloads/kinetics-skeleton/kinetics_train',
              '/Users/zouhuanxin/Downloads/kinetics-skeleton/kinetics_train_label.json')
