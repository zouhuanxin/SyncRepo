import sys
import argparse
import yaml
import numpy as np

# torch
import torch
import torch.nn as nn

from utils_zhx import *


class IO():
    """
        IO Processor
        这里负责完成
    """

    def __init__(self, argv=None):
        self.load_arg(argv)
        self.init_environment()
        self.load_model()
        self.load_weights()
        self.gpu()

    def load_arg(self, argv=None):
        parser = self.get_parser()

        # load arg form config file
        p = parser.parse_args(argv)
        if p.config is not None:
            # load config file
            with open(p.config, 'r') as f:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)

            # update parser from config file
            key = vars(p).keys()
            for k in default_arg.keys():
                if k not in key:
                    print('Unknown Arguments: {}'.format(k))
                    assert k in key

            parser.set_defaults(**default_arg)

        self.arg = parser.parse_args(argv)

    def init_environment(self):
        # gpu
        if self.arg.use_gpu:
            if torch.cuda.is_available():
                self.dev = "cuda:0"
        else:
            self.dev = "cpu"

    def load_model(self):
        self.model = import_module(self.arg.model).Model()

    def load_weights(self):
        if self.arg.weights:
            pretrained_weights = torch.load(self.arg.weights)
            self.model.load_state_dict(pretrained_weights)

    def gpu(self):
        # move modules to gpu
        self.model = self.model.to(self.dev)


    @staticmethod
    def get_parser(add_help=False):

        # region arguments yapf: disable
        # parameter priority: command line > config > default
        parser = argparse.ArgumentParser(add_help=add_help, description='IO Processor')

        parser.add_argument('-c', '--config', default=None, help='path to the configuration file')

        # processor
        parser.add_argument('--use_gpu', type=bool, default=False, help='use GPUs or not')

        # model
        parser.add_argument('--model', default='TopModel', help='the model will be used')
        parser.add_argument('--weights', default=None, help='the weights for network initialization')

        return parser
