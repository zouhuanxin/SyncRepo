'''
author：zhx
function：封装的通用函数
'''

import sys


# 进度条
# epoch：当前循环下标
# loss：本轮循环损失
# finish_tasks_number: int, 已完成的任务数
# tasks_number: int, 总的任务数
def progress_bar(epoch, loss, finish_tasks_number, tasks_number):
    percentage = round(finish_tasks_number / tasks_number * 100)
    print(
        "\repoch:{} loss:{:.3f} {}/{} 进度:{}%:".format(epoch, loss, finish_tasks_number, tasks_number, percentage),
        "▓" * (percentage // 2), end="")
    sys.stdout.flush()


def show_epoch_info(True_num, Sum):
    print('正确率:{}'.format(True_num / Sum))


# 传入一个x，将x按照size大小分割成块,输入必须是pytorch类型
# 支持4维度
# 第一个维度 batchsize
# 第二个维度 channels
# 第三个维度 horizontal_patches 水平切割出的块数量
# 第四个维度 vertical_patches 垂直切割出的块数量
# 第五个维度 kernelsize
# 第六个维度 kernelsize
def partition_4_torch(x, kernel_size=3, stride=1):
    patchs = x.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)

    return patchs


# 传入一个x，将x按照size大小分割成块,输入必须是pytorch类型
# 支持2维度
# 第一个维度 horizontal_patches 水平切割出的块数量
# 第二个维度 vertical_patches 垂直切割出的块数量
# 第三个维度 kernelsize
# 第四个维度 kernelsize
def partition_2_torch(x, kernel_size=3, stride=1):
    patchs = x.unfold(0, kernel_size, stride).unfold(1, kernel_size, stride)

    return patchs


# 动态导入模块
def import_module(module_name):
    try:
        module = __import__(module_name)
        return module
    except ImportError:
        return None  # 如果模块不存在，返回None

