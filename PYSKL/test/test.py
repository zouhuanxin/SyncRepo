# minFrame = 3000
# index = 0
# data_path = '/Users/zouhuanxin/Downloads/数据集/nturgb+d_skeletons'
# samples = os.listdir(data_path)


# def read_skeleton(file):
#     with open(file, 'r') as f:
#         numFrame = int(f.readline())
#     return numFrame
#
#
# for filename in samples:
#     progress_bar2(index, len(samples))
#     numFrame = read_skeleton(os.path.join(data_path, filename))
#     if numFrame < minFrame:
#         minFrame = numFrame
#     index = index + 1
#
# print(minFrame)

# max = 300
# min = 26


# path = '/Users/zouhuanxin/PycharmProjects/pythonProject1/PYSKL/train_data'
# names = os.listdir(path)
# for i,name in enumerate(names):
#     if name == '.DS_Store':
#         continue
#     p = os.path.join(path, name)
#     if len(os.listdir(p)) < 26:
#         print(len(os.listdir(p)))


group_size, remainder = divmod(45, 26)
print(group_size)
print(remainder)