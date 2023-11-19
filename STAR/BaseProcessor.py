import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class BaseProcessor():
    """
    基础处理器
    所有模型继承此处理器进行计算
    1.加载环境变量
    2.加载模型
    3.加载权重
    4.加载数据
    5.加载优化器
    6.训练
    7.测试
    8.启动
    """

    def __init__(self, phase='train', start_epoch=int, end_epoch=int):
        self.phase = phase
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch

    def init_environment(self):
        # gpu
        if torch.cuda.is_available():
            self.dev = "cuda:0"
        else:
            self.dev = "cpu"

        self.current_epoch = dict()

    def load_model(self, model):
        if model is None:
            print('model不能为空')
        self.model = model.to(self.dev)

    # weights权重文件
    def load_weighs(self, weights):
        if self.model is None:
            print('model不能为空')
        if weights is None:
            print('weights文件不能为空')
        pretrained_weights = torch.load(weights)
        self.model.load_state_dict(pretrained_weights)

    def load_data(self, dataset=None, batch_size=64, dataloader=None):
        self.data_loader = dict()
        if dataloader is not None:
            if self.phase == 'train':
                self.data_loader['train'] = dataloader
            else:
                self.data_loader['test'] = dataloader
        else:
            if self.phase == 'train':
                self.data_loader['train'] = DataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=True)
            else:
                self.data_loader['test'] = DataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    shuffle=False)

    def load_optimizer(self, optimizer='SGD', learn_rate=0.001, weight_decay=0.0001):
        if optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learn_rate,
                momentum=0.9,
                nesterov=True,
                weight_decay=weight_decay)
        elif optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learn_rate,
                weight_decay=weight_decay)
        else:
            raise ValueError()

    # train,test方法需要子类自行实现
    def train(self):
        pass

    def test(self):
        pass

    def start(self):
        # training phase
        if self.phase == 'train':
            for epoch in range(self.start_epoch, self.end_epoch):
                self.current_epoch['epoch'] = epoch
                print('Training epoch: {}'.format(epoch))
                self.train()
                print('Done.')

                # save model
                if ((epoch + 1) % 10 == 0) or (epoch + 1 == self.end_epoch):
                    filename = 'epoch{}_model.pt'.format(epoch + 1)
                    torch.save(self.model, filename)

                # evaluation
                if ((epoch + 1) % 5 == 0) or (epoch + 1 == self.end_epoch):
                    print('Eval epoch: {}'.format(epoch))
                    self.test()
                    print('Done.')
        # test phase
        elif self.phase == 'test':
            print('Evaluation Start:')
            self.test()
            print('Done.\n')
