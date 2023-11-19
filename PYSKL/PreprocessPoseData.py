"""
预处理数据

直接利用现有的骨骼点数据集制作热力图

"""
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os
import time
import numpy as np

# 生成指定大小的图片
width = 100
height = 100


def createImage():
    # 指定图片的大小（宽度和高度）

    # 指定背景颜色（R, G, B），这里使用纯黑色
    background_color = (0, 0, 0)

    # 创建一张纯色背景的图片
    image = Image.new("RGB", (width, height), background_color)
    return image


# 计算坐标点的热力值，这里使用示例距离计算，你可以根据实际需求定义热力值计算方法
def calculate_heatmap_value(coord, coordinates):
    # 例如，计算每个坐标点到给定坐标点的距离，并将距离用作热力值
    values = []
    for person in coordinates:
        for sk_coord in person.get_coordinates():
            values.append(((coord[0] - sk_coord.get_x()) ** 2 + (coord[1] - sk_coord.get_y()) ** 2) * 2)
    return min(values)


# 将关节点都链接起来
def skeleton_parts(dataset='kinetics'):
    if ('ntu' in dataset) or ('NTU' in dataset):
        sk_adj = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24],
                  [1, 20, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10, 0, 12, 13, 14, 0, 16, 17, 18, 22, 7, 24, 11]]
    elif 'kinetics' in dataset:
        sk_adj = [[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                  [1, 1, 2, 3, 1, 5, 6, 2, 8, 9, 5, 11, 12, 0, 0, 14, 15]]
    return sk_adj


# 根据指定的坐标点在图片上绘制坐标点热力图,并且保存起来,保存文件名是：动作名+时间戳.jpg
def drawHotImage(coordinates, image, file_name, save_path, dataset='ntu'):
    if os.path.exists(save_path) is False:
        OSError('save_path为空')
        return
    save_path = '{}/{}.jpg'.format(save_path, file_name)
    if os.path.exists(save_path):
        return

    # 方法1.创建一个与图像大小相同的热力图,速度很慢
    # heatmap = Image.new("L", image.size)
    # for x in range(image.width):
    #     for y in range(image.height):
    #         heatmap_value = calculate_heatmap_value((x, y), coordinates) * 4
    #         heatmap.putpixel((x, y), int(heatmap_value))

    # 方法2.使用矢量化操作计算热力图值，速度较快
    heatmap = np.zeros((height, width), dtype=np.uint8)
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x, y = x.flatten(), y.flatten()
    for person in coordinates:
        for sk_coord in person.get_coordinates():
            dist_sq = (x - sk_coord.get_x()) ** 2 + (y - sk_coord.get_y()) ** 2
            heatmap_values = 4 * np.exp(-dist_sq / (2 * 10 ** 2))  # 这里使用高斯核作为示例
            heatmap += heatmap_values.reshape(height, width).astype(np.uint8)
    heatmap = Image.fromarray(heatmap)

    # 绘制热力图
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.axis('off')
    plt.imsave(save_path, heatmap, cmap='hot')
    drawSkConnect(save_path, coordinates, dataset)


# 根据指定的热力点来绘制骨骼链接
def drawSkConnect(save_path, coordinates, dataset='kinetics'):
    global num_joins
    if ('ntu' in dataset) or ('NTU' in dataset):
        num_joins = 24
    elif 'kinetics' in dataset:
        num_joins = 17
    else:
        ValueError('dataset数据类型错误')
    image = Image.open(save_path)
    draw = ImageDraw.Draw(image)
    for person in coordinates:
        for i in range(num_joins):
            # 过滤过低的的分数的节点
            # if person.get_coordinates()[skeleton_parts('kinetics')[0][i]].get_score() < 0.4 or \
            #         person.get_coordinates()[skeleton_parts('kinetics')[1][i]].get_score() < 0.4:
            #     continue
            draw.line([(person.get_coordinates()[skeleton_parts(dataset)[0][i]].get_x(),
                        person.get_coordinates()[skeleton_parts(dataset)[0][i]].get_y()), (
                           person.get_coordinates()[skeleton_parts(dataset)[1][i]].get_x(),
                           person.get_coordinates()[skeleton_parts(dataset)[1][i]].get_y())],
                      fill=(255, 127, 0), width=1)
    drawSkPosition(draw, coordinates)
    image.save(save_path)


# 根据指定的热力点来编码position到图上
# 绘制数字
# 绘制一个小的颜色点
colors = [(255, 228, 225), (47, 79, 79), (119, 136, 153), (100, 149, 237), (0, 100, 0), (255, 255, 0), (218, 165, 32),
          (188, 143, 143), (210, 105, 30), (174, 238, 238), (150, 205, 205), (122, 197, 205), (0, 229, 238),
          (127, 255, 212),
          (102, 205, 170), (193, 255, 193), (155, 205, 155), (78, 238, 148), (46, 139, 87), (0, 238, 118), (0, 238, 0),
          (127, 255, 0), (102, 205, 0), (192, 255, 62), (179, 238, 58)]


def drawSkPosition(draw, coordinates):
    for person in coordinates:
        index = 0
        for sk_coord in person.get_coordinates():
            # position = (sk_coord.get_x(), sk_coord.get_y())
            # font = ImageFont.truetype("./font/font1.ttf", 8)
            # draw.text(position, str(index), fill=(67, 205, 128), font=font)
            # draw.point(position, fill=colors[index])
            center_x, center_y = sk_coord.get_x(), sk_coord.get_y()
            radius = 2
            fill_color = colors[index]
            draw.ellipse([(center_x - radius, center_y - radius),
                          (center_x + radius, center_y + radius)],
                         fill=fill_color)

            index = index + 1
