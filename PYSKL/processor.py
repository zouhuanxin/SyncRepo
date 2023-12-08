import numpy as np
import torch.nn as nn

from BaseProcessor import BaseProcessor

from tools.utils_zhx import *

from model.net import PysklModel as Model
from feeder.Dataset import Feeder


batch_size = 4

#test_path_after = 'D:/数据集/'
#test_path2_after = '/23085412008/PYSKL/PYSKL/'

class Processor(BaseProcessor):
    def __init__(self):
        super().__init__(end_epoch=100)
        self.init_environment()
        self.load_model(Model(batch_size=batch_size))
        train_feeder = Feeder('D:/数据集/skeleton_img_small', skeleton_path='D:/数据集/nturgb+d_skeletons', phase='train')
        test_feeder = Feeder('D:/数据集/skeleton_img_small', skeleton_path='D:/数据集/nturgb+d_skeletons', phase='test')
        self.load_data(dataset=train_feeder, batch_size=batch_size, dataType='train')
        self.load_data(dataset=test_feeder, batch_size=batch_size, dataType='test')
        self.loss_func = nn.CrossEntropyLoss()

    def train(self):
        loader = self.data_loader['train']
        i = 0

        for data, label, skeleton_data in loader:
            self.optimizer.zero_grad()
            data = data.reshape(data.shape[0] * data.shape[1], data.shape[2], data.shape[3], data.shape[4])
            data = data.to(self.dev)
            label = label.to(self.dev)
            skeleton_data = skeleton_data.reshape(skeleton_data.shape[0], skeleton_data.shape[1],
                                                  skeleton_data.shape[2] * skeleton_data.shape[4],
                                                  skeleton_data.shape[3])
            skeleton_data = skeleton_data.to(self.dev)
            output = self.model(data, skeleton_data)
            loss = self.loss_func(output, label)
            loss.backward()
            self.current_associated_data['loss'] = self.current_associated_data['loss'] + loss.item()
            self.optimizer.step()

            progress_bar(self.current_associated_data['epoch'], self.current_associated_data['loss'], i, len(loader))
            i = i + 1

    def test(self):
        loader = self.data_loader['test']
        outputs = []
        labels = []
        i = 0
        for data, label, skeleton_data in loader:
            if data.shape[0] == batch_size:  # 过滤非法
                data = data.reshape(data.shape[0] * data.shape[1], data.shape[2], data.shape[3], data.shape[4])
                data = data.to(self.dev)
                label = label.to(self.dev)
                skeleton_data = skeleton_data.reshape(skeleton_data.shape[0], skeleton_data.shape[1],
                                                      skeleton_data.shape[2] * skeleton_data.shape[4],
                                                      skeleton_data.shape[3])
                skeleton_data = skeleton_data.to(self.dev)

                output = self.model(data, skeleton_data)
                for a in range(len(label)):
                    labels.append(label[a])
                for b in range(len(output)):
                    outputs.append(find_max_index(output[b]))

            progress_bar2(i, len(loader))
            i = i + 1

        outputs = np.array(outputs)
        labels = np.array(labels)
        T = 0
        for i in range(len(outputs)):
            if outputs[i] == labels[i]:
                T = T + 1
        print('True Number:{},False Number:{},Correctness:{}'.format(T, len(outputs) - T, T / len(outputs)))


# 4145
Processor().start('train')
