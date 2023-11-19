import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
#
# a = np.zeros((10, 3, 100, 100))
# b = np.zeros((20, 3, 100, 100))
# c = np.zeros((30, 3, 100, 100))
#
# d = []
# d.append(a)
# d.append(b)
# d.append(c)
#
# print(d)


import numpy as np


def sample_from_groups(my_list, num_groups):
    # 确定列表的长度
    list_size = len(my_list)
    # 如果小于num_groups则补帧,默认补最后一帧
    if list_size < num_groups:
        for i in range(list_size, num_groups):
            my_list.append(my_list[len(my_list) - 1])
        list_size = len(my_list)
    print(my_list)
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

# 示例用法
my_array = ['34-18.jpg', '34-6.jpg', '34-11.jpg', '34-4.jpg', '34-21.jpg', '34-2.jpg', '34-1.jpg', '34-20.jpg', '34-14.jpg', '34-12.jpg', '34-8.jpg', '34-5.jpg', '34-15.jpg', '34-13.jpg', '34-17.jpg', '34-7.jpg', '34-16.jpg', '34-3.jpg', '34-10.jpg', '34-9.jpg', '34-19.jpg', '34-0.jpg']
print(len(my_array))
num_groups = 26

sampled_elements = sample_from_groups(my_array, num_groups)
print(sampled_elements)
