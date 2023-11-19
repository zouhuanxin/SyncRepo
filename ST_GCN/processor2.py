import torch
import torch.nn as nn

from BaseProcessor import BaseProcessor

from feeder import Feeder
from st_gcn import Model
from utils_zhx import *


class Processor2(BaseProcessor):
    def __init__(self):
        super().__init__(start_epoch=0, end_epoch=100)
        self.init_environment()
        graph_args = {
            "layout": "openpose",
            "strategy": "spatial"
        }
        self.load_model(Model(3, 400, graph_args, True))
        self.load_data(dataset=Feeder('/Users/zouhuanxin/Downloads/数据集/kinetics-skeleton/train_data.npy',
                                      '/Users/zouhuanxin/Downloads/数据集/kinetics-skeleton/train_label.pkl'))
        self.load_optimizer()
        self.loss_func = nn.CrossEntropyLoss()

    def train(self):
        loader = self.data_loader['train']
        loss_value = 0
        i = 0

        for data, label in loader:
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            output = self.model(data)
            print(output.shape)
            print(label.shape)
            loss = self.loss_func(output, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_value = loss_value + loss

            progress_bar(self.current_epoch['epoch'], loss_value, i, len(loader))
            i = i + 1
        torch.save(self.model.state_dict(), './model/st_gcn_model.pth')


Processor2().start()
