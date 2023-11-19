import numpy as np
import torch
import torch.nn as nn

from BaseProcessor import BaseProcessor

from utils_zhx import *

from net import DualGraphEncoder as Model

from torch_geometric.data import DataLoader
from GeometricFeeder import SkeletonDataset

sk_adj = torch.tensor(
    [[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
     [1, 1, 2, 3, 1, 5, 6, 2, 8, 9, 5, 11, 12, 0, 0, 14, 15]]
)


class Processor(BaseProcessor):
    def __init__(self):
        super().__init__(start_epoch=0, end_epoch=100)
        self.init_environment()
        self.load_model(Model(in_channels=3,
                              hidden_channels=128,
                              out_channels=128,
                              num_layers=8,
                              num_heads=8,
                              sequential=False))
        dataset = SkeletonDataset('/23085412007/zhx/kinetics_data/', 'kinetics')
        self.load_data(dataloader=DataLoader(dataset.data, batch_size=16))
        self.load_optimizer()
        self.loss_func = nn.CrossEntropyLoss()

    def train(self):
        loader = self.data_loader['train']
        loss_value = 0

        for i, batch in enumerate(loader):
            data = batch.x.float().to(self.dev)
            label = batch.y.long().to(self.dev)

            output = self.model(data, sk_adj, batch.batch)
            loss = self.loss_func(output, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_value = loss_value + loss

            progress_bar(self.current_epoch['epoch'], loss_value, i, len(loader))
        torch.save(self.model.state_dict(), './model/star_model.pth')



Processor().start()
