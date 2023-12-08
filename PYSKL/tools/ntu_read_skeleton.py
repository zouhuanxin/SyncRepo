from utils_zhx import *
from PYSKL.feeder.BeanData import *
from PreprocessPoseData import *


def read_skeleton(file):
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []
            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)
    return skeleton_sequence


def read_xy(data_path, filename, label, max_body=2, num_joint=25):
    minX = 0
    minY = 0
    maxX = 1920
    maxY = 1080
    margin = 50
    seq_info = read_skeleton(os.path.join(data_path, filename))
    for n, f in enumerate(seq_info['frameInfo']):  # n是当前帧数
        # 先计算最小xy 最大xy
        temp_X = []
        temp_Y = []
        for m, b in enumerate(f['bodyInfo']):  # m是当前人体骨架数
            for j, v in enumerate(b['jointInfo']):  # j是当前的一组关节点
                if m < max_body and j < num_joint:
                    temp_X.append(v['colorX'])
                    temp_Y.append(v['colorY'])
        # 如果没有关节点则过滤
        if len(temp_X) == 0:
            continue
        temp_X = np.array(temp_X)
        temp_Y = np.array(temp_Y)
        minX = min(temp_X) - margin
        maxX = max(temp_X) + margin
        minY = min(temp_Y) - margin
        maxY = max(temp_Y) + margin
        coordinates = []
        for m, b in enumerate(f['bodyInfo']):  # m是当前帧的人体骨架数
            persondata = PersonData(m)
            for j, v in enumerate(b['jointInfo']):  # j是当前的一组关节点
                if m < max_body and j < num_joint:
                    coord = Coordinate((v['colorX'] - minX) / (maxX - minX) * width,
                                       (v['colorY'] - minY) / (maxY - minY) * height)
                    persondata.append(coord)
                else:
                    pass
            # 如果当前帧的骨架数量不对则过滤
            if len(persondata.get_coordinates()) == 25:
                coordinates.append(persondata)
        if len(coordinates) != 0:
            # 开始绘制热力图
            save_path = './skeleton_img/{}'.format(filename.split('.')[0])
            if os.path.exists(save_path) is False:
                os.mkdir(save_path)
            drawHotImage(coordinates, createImage(), '{}-{}'.format(label, n), save_path)


training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
training_cameras = [2, 3]


# "X-Sub"表示训练集和验证集来自不同的人物，而"X-View"则表示训练集和测试集来自不同的摄像头视角。
def load_data(data_path, benchmark='xsub', part='train'):
    index = 0
    for filename in os.listdir(data_path):
        progress_bar2(index, len(os.listdir(data_path)))
        action_class = int(
            filename[filename.find('A') + 1:filename.find('A') + 4])
        subject_id = int(
            filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(
            filename[filename.find('C') + 1:filename.find('C') + 4])

        if benchmark == 'xview':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError()

        if issample:
            # sample_name.append(filename)
            # sample_label.append(action_class - 1)
            read_xy(data_path, filename, action_class - 1)
        index = index + 1


#load_data('/Users/zouhuanxin/Downloads/数据集/nturgb+d_skeletons')
#load_data('/23085412007/zhx/PycharmProjects/nturgb+d_skeletons')
load_data('/23085412008/PYSKL/PYSKL/nturgb+d_skeletons')