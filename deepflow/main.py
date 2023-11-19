import numpy as np
import torch
from utils_zhx import *

from Model import *
from LoadData import *
import torch.optim as optim

spatialStreamConvNet = SpatialStreamConvNet(240, 320, 3)

data, label = readImage()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=spatialStreamConvNet.parameters(), lr=0.0001)

print(len(data))
for epoch in range(10):
    outs = []
    for i in data:
        x = torch.Tensor(np.array(i))
        x = x.permute(0, 3, 1, 2)
        output = spatialStreamConvNet(x)
        # 计算出这一个组别最终所代表的动作
        res = []
        for j in output:
            index = find_max_index(j)
            res.append(index)
        # 得到最终的计算结果
        out = find_max_index(res)
        outs.append(out)
    # 计算损失
    outs = torch.Tensor(np.array(outs))
    target = torch.Tensor(label)
    outs = outs.reshape(1, -1)
    target = target.reshape(1, -1)

    optimizer.zero_grad()
    loss = criterion(outs, target)
    loss.requires_grad_(True)
    print(loss)

    loss.backward()
    optimizer.step()

