import torch.utils.data as data
import numpy as np
import random


class Feeder(data.Dataset):

    def __init__(self):
        pass

    def __len__(self):
        return 1000

    def __getitem__(self, index):
        random_integer = random.randint(1, 10)
        print(random_integer)
        a = np.zeros((1, random_integer, 3, 100, 100))
        # b = np.zeros((20, 3, 100, 100))
        # c = np.zeros((30, 3, 100, 100))
        res = []
        res.append(a)
        # res.append(b)
        # res.append(c)
        return res
