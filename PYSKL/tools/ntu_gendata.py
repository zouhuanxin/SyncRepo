"""
将图片数据读取出来变成npy数据格式
"""
import os

import numpy as np
from PIL import Image
from numpy.lib.format import open_memmap

def read_skeleton(file):
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []
            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)
    return skeleton_sequence


def read(path, file):
    seq_info = read_skeleton(file)
    sample_names = os.listdir(path)
    # T C H W
    data = np.zeros((seq_info['numFrame'], 3, 200, 200))
    for sample_name in enumerate(sample_names):
        child_path = os.path.join(path, sample_name)
        sample_names2 = os.listdir(child_path)
        index = 0
        for sample_name2 in enumerate(sample_names2):
            image = Image.open(os.path.join(child_path, sample_name2))
            image = np.array(image) / 255
            data[index] = image
            index = index + 1
    return data


def genNpy(path,out_path,part='eval'):
    sample_names = os.listdir(path)

