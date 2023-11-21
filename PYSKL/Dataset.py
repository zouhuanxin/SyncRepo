import os
import numpy as np
import torch
import torch.utils.data as data

from PIL import Image
import torchvision.transforms as transforms


# 创建转换操作
transform = transforms.Compose([
    transforms.Resize((100, 100)),  # 调整图片大小
    transforms.ToTensor()  # 归一化到0-1
])


class Feeder(data.Dataset):

    def __init__(self, data_path, skeleton_path=None, phase='train'):
        self.data_path = data_path
        self.skeleton_path = skeleton_path
        self.skeleton_sample_names = os.listdir(skeleton_path)  # 骨架数据
        self.phase = phase
        if phase == 'train':
            self.sample_names = self.readData()[0]
        else:
            self.sample_names = self.readData()[1]

    # 根据phase阶段来读取一个文件夹中不同的数据
    def readData(self):
        train_names = []
        test_names = []
        data_paths = os.listdir(self.data_path)
        for i,filename in enumerate(data_paths):
            action_class = int(
                filename[filename.find('A') + 1:filename.find('A') + 4])
            subject_id = int(
                filename[filename.find('P') + 1:filename.find('P') + 4]) #人物，总共40个人，35个人训练，5个人测试
            camera_id = int(
                filename[filename.find('C') + 1:filename.find('C') + 4])
            if subject_id <= 35:
                train_names.append(filename)
            else:
                test_names.append(filename)
        return train_names, test_names


    def __len__(self):
        return len(self.sample_names)

    # 根据文件名搜索对应的骨架文件的下标
    def searchSkeletonByFilename(self, filename):
        filename = filename + '.skeleton'
        index = self.skeleton_sample_names.index(filename)
        return index

    # 解析骨架数据为[C,T,V]
    def read_skeleton(self, file):
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

    def read_xyz(self, file, max_body=2, num_joint=25):
        seq_info = self.read_skeleton(file)
        # data = np.zeros((3, seq_info['numFrame'], num_joint, max_body))
        data = np.zeros((3, 150, num_joint, max_body))
        frames = 150
        for n, f in enumerate(seq_info['frameInfo']):
            if n >= frames:
                break
            for m, b in enumerate(f['bodyInfo']):
                for j, v in enumerate(b['jointInfo']):
                    if m < max_body and j < num_joint:
                        data[:, n, j, m] = [v['x'], v['y'], v['z']]
                    else:
                        pass
        return data

    # 将图片序列均匀分割然后采样26次
    def sample_from_groups(self, my_list, num_groups):
        # 确定列表的长度
        list_size = len(my_list)
        # 如果小于num_groups则补帧,默认补最后一帧
        if list_size < num_groups:
            for i in range(list_size, num_groups):
                my_list.append(my_list[len(my_list) - 1])
            list_size = len(my_list)

        # 计算每个组的理论大小和余数
        group_size, remainder = divmod(list_size, num_groups)

        # 初始化结果列表
        sampled_elements = []

        num_groups_integer = num_groups
        if remainder != 0:  # 有余数就减1，前面采样25次
            num_groups_integer = num_groups_integer - 1

        # 循环遍历整除的部分，每个组随机采样一个元素
        for i in range(num_groups_integer):
            start_index = i * group_size
            end_index = start_index + group_size
            sampled_element = my_list[np.random.randint(start_index, end_index)]
            sampled_elements.append(sampled_element)

        if remainder != 0:
            # 处理余数部分，余数自成一组
            sampled_element = my_list[np.random.randint(list_size - remainder - 1, list_size - 1)]
            sampled_elements.append(sampled_element)

        return sampled_elements

    def __getitem__(self, index):
        minFrame = 26
        video_file_name = self.sample_names[index]
        video_file_path = os.path.join(self.data_path, video_file_name)
        video_cilp_paths = os.listdir(video_file_path)  # 这个动作视频有多少帧
        data = np.zeros((minFrame, 3, 100, 100))
        video_cilp_files = self.sample_from_groups(video_cilp_paths, minFrame)  # 均匀采样
        for i, sample_name in enumerate(video_cilp_files):
            img_path = os.path.join(video_file_path, sample_name)
            image = Image.open(img_path)
            image_tensor = transform(image)
            data[i] = image_tensor
        label = video_file_name[video_file_name.find('A') + 1:video_file_name.find('A') + 4]
        data = torch.Tensor(data).requires_grad_(True)

        skeleton_data = []
        if self.skeleton_path is not None:
            skeleton_file_name = self.skeleton_sample_names[self.searchSkeletonByFilename(video_file_name)]
            skeleton_file_path = os.path.join(self.skeleton_path, skeleton_file_name)
            skeleton_data = self.read_xyz(skeleton_file_path)
        skeleton_data = torch.Tensor(skeleton_data).requires_grad_(True)
        return data, int(label) - 1, skeleton_data
