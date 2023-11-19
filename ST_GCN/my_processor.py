import sys
import argparse
import yaml
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from my_io import IO
from feeder import Feeder


class Processor(IO):
    """
        基础处理器
        所有模型继承此处理器进行计算
        1.加载环境参数
        2.初始化环境
        3.加载模型（这里的模型加载顶级父模型，如果不做其他配置的话）
        4.加载权重
        5.设置gpu
        6.加载数据
    """

    def __init__(self, argv=None):
        self.load_arg(argv)
        self.init_environment()
        self.load_model()
        self.load_weights()
        self.gpu()
        self.load_data()

    def init_environment(self):
        super().init_environment()
        self.result = dict()
        self.iter_info = dict()
        self.epoch_info = dict()
        self.meta_info = dict(epoch=0, iter=0)

    def load_data(self):
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = DataLoader(
                dataset=Feeder('/Users/zouhuanxin/Downloads/kinetics-skeleton/train_data.npy',
                               '/Users/zouhuanxin/Downloads/kinetics-skeleton/train_label.pkl'),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True)
        else:
            self.data_loader['test'] = DataLoader(
                dataset=Feeder('/Users/zouhuanxin/Downloads/kinetics-skeleton/train_data.npy',
                               '/Users/zouhuanxin/Downloads/kinetics-skeleton/train_label.pkl'),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=0)

    # 展示epoch的信息
    def show_epoch_info(self):
        for k, v in self.epoch_info.items():
            print('\t{}: {}'.format(k, v))

    # 展示单次循环信息
    def show_iter_info(self):
        if self.meta_info['iter'] % 100 == 0:
            info = '\tIter {} Done.'.format(self.meta_info['iter'])
            for k, v in self.iter_info.items():
                if isinstance(v, float):
                    info = info + ' | {}: {:.4f}'.format(k, v)
                else:
                    info = info + ' | {}: {}'.format(k, v)

            print(info)

    def train(self):
        for _ in range(100):
            self.iter_info['loss'] = 0
            self.show_iter_info()
            self.meta_info['iter'] += 1
        self.epoch_info['mean loss'] = 0
        self.show_epoch_info()

    def test(self):
        for _ in range(100):
            self.iter_info['loss'] = 1
            self.show_iter_info()
        self.epoch_info['mean loss'] = 1
        self.show_epoch_info()

    def start(self):
        # training phase
        if self.arg.phase == 'train':
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.meta_info['epoch'] = epoch
                # training
                print('Training epoch: {}'.format(epoch))
                self.train()
                print('Done.')

                # save model
                if ((epoch + 1) % 10 == 0) or (
                        epoch + 1 == self.arg.num_epoch):
                    filename = 'epoch{}_model.pt'.format(epoch + 1)
                    torch.save(self.model, filename)

                # evaluation
                if ((epoch + 1) % 5 == 0) or (
                        epoch + 1 == self.arg.num_epoch):
                    print('Eval epoch: {}'.format(epoch))
                    self.test()
                    print('Done.')
        # test phase
        elif self.arg.phase == 'test':
            # the path of weights must be appointed
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            print('Model:   {}.'.format(self.arg.model))
            print('Weights: {}.'.format(self.arg.weights))

            # evaluation
            print('Evaluation Start:')
            self.test()
            print('Done.\n')

    @staticmethod
    def get_parser(add_help=False):

        # region arguments yapf: disable
        # parameter priority: command line > config > default
        parser = argparse.ArgumentParser(add_help=add_help, description='Base Processor')

        parser.add_argument('-c', '--config', default=None, help='path to the configuration file')

        # processor
        parser.add_argument('--phase', default='train', help='must be train or test')
        parser.add_argument('--start_epoch', type=int, default=0, help='start training from which epoch')
        parser.add_argument('--num_epoch', type=int, default=80, help='stop training in which epoch')
        parser.add_argument('--use_gpu', type=bool, default=False, help='use GPUs or not')

        # model
        parser.add_argument('--model', default='TopModel', help='the model will be used')
        parser.add_argument('--weights', default=None, help='the weights for network initialization')

        # feeder
        parser.add_argument('--feeder', default='feeder', help='data loader will be used')
        parser.add_argument('--data_path', default=None, help='data_path')
        parser.add_argument('--label_path', default=None, help='label_path')

        parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
        parser.add_argument('--test_batch_size', type=int, default=256, help='test batch size')

        return parser
