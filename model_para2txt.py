# 将模型的参数保存为txt文件
import torch
from model import net_DWConv
import os
import shutil
import numpy as np

device = torch.device("cpu")
model_path = "./runs/train/exp6/model/G.pth"
state_dict = torch.load(model_path, map_location=device)

channels = 1
base = 64
M = 8
mode = "test"
model = net_DWConv(channels,base,M,mode)

model.load_state_dict(state_dict)
model.eval()


#只选择编码端网络
print(model.sample1.weight.shape)

# 模型参数保存
save_path = "./model_parameters/"
if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.mkdir(save_path)

data1 = model.sample1.weight.detach().numpy()
# 卷积核个数，卷积核的通道数
num_filter, num_chn = data1.shape[:2]
for i in range(num_filter):
    for j in range(num_chn):

        data = data1[i,j]

        save_name = save_path + f"网络层1_卷积核{i}_通道{j}.txt"
        np.savetxt(save_name, data)


data2 = model.sample2.depth_conv.weight.detach().numpy()
# 卷积核个数，卷积核的通道数
num_filter, num_chn = data2.shape[:2]
for i in range(num_filter):
    for j in range(num_chn):

        data = data2[i,j]

        save_name = save_path + f"网络层2_深度卷积_卷积核{i}_通道{j}.txt"
        np.savetxt(save_name, data)


data2 = model.sample2.point_conv.weight.detach().numpy()
num_filter = data2.shape[0]
for i in range(num_filter):
    data = data2[i].reshape(-1)
    save_name = save_path + f"网络层2_点卷积_卷积核{i}.txt"
    np.savetxt(save_name, data.reshape(-1))


data3 = model.sample3.depth_conv.weight.detach().numpy()
# 卷积核个数，卷积核的通道数
num_filter, num_chn = data3.shape[:2]
for i in range(num_filter):
    for j in range(num_chn):

        data = data3[i,j]

        save_name = save_path + f"网络层3_深度卷积_卷积核{i}_通道{j}.txt"
        np.savetxt(save_name, data)

data3 = model.sample3.point_conv.weight.detach().numpy()
num_filter = data3.shape[0]
for i in range(num_filter):
    data = data3[i].reshape(-1)
    save_name = save_path + f"网络层3_点卷积_卷积核{i}.txt"
    np.savetxt(save_name, data)
