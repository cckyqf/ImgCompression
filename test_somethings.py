import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn

# class Model(nn.Module):
#     def __init__(self,num_classes):
#         super().__init__()

#         self.m = nn.Conv2d(1,16,3)
    
#     def forward(self, x):
#         return self.m(x)
    

# model = Model(num_classes=2)
# optimizer = optim.SGD(params = model.parameters(), lr=1.0)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90], gamma=0.1)

# plt.figure()
# x = list(range(100))
# y = []

# for epoch in range(100):

#     for idx in range(100):
#         scheduler.step()
#         lr = scheduler.get_lr()
#         print(epoch, scheduler.get_lr()[0])
#         y.append(scheduler.get_lr()[0])
# plt.xlabel("epoch")
# plt.ylabel("learning rate")
# # plt.plot(x,y)
# plt.plot(y)
# plt.show()


# # model = torch.load("runs/train/exp18/model/G.pth")
# from model import net_DWConv
# model = net_DWConv(channels=1, base=64, M=8, mode="train")
# print(model)

import numpy as np

x = np.arange(-10,10,0.1)
x1 = torch.from_numpy(x)
y = torch.sigmoid(x1)

y = y.cpu().numpy()

plt.plot(x,y)

plt.show()
