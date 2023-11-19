from PIL import Image
import matplotlib.pyplot as plt

# 读取图像
# image = Image.open("/Users/zouhuanxin/Downloads/zhxnote/test03.jpg")
# 指定图片的大小（宽度和高度）
width = 200
height = 200

# 指定背景颜色（R, G, B），这里使用纯白色
background_color = (0, 0, 0)

# 创建一张纯色背景的图片
image = Image.new("RGB", (width, height), background_color)

# 坐标点示例，你需要替换为你的坐标点
# 这里使用(x, y)表示坐标点的位置
coordinates = [(100, 100), (40, 150), (120, 70)]

# 创建一个与图像大小相同的热力图
heatmap = Image.new("L", image.size)


# 计算坐标点的热力值，这里使用示例距离计算，你可以根据实际需求定义热力值计算方法
def calculate_heatmap_value(coord, coordinates):
    # 例如，计算每个坐标点到给定坐标点的距离，并将距离用作热力值
    return min([((coord[0] - x) ** 2 + (coord[1] - y) ** 2) for x, y in coordinates])


for x in range(image.width):
    for y in range(image.height):
        heatmap_value = calculate_heatmap_value((x, y), coordinates)
        heatmap.putpixel((x, y), int(heatmap_value))

# 绘制热力图
plt.imshow(heatmap, cmap='hot', interpolation='nearest')
# plt.colorbar()
# plt.title("Heatmap of Given Coordinates")
plt.axis('off')

# 显示图像和热力图
plt.show()

#plt.imsave('./test01.jpg', heatmap, cmap='hot')
