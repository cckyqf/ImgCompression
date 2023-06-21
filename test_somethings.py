# import matplotlib.pyplot as plt
# import numpy as np

# x = np.arange(-10,10,0.1)
# # x1 = torch.from_numpy(x)
# # y = torch.sigmoid(x1)
# # y = y.cpu().numpy()

# y = x
# plt.figure()
# plt.plot(x,y)

# plt.show()

import torch
path = "runs/train/exp58/model/best.pt"

model = torch.load(path, map_location='cpu')["model"]
print(model)

# import numpy as np

# psnr = np.array([25.119, 22.324, 22.807, 23.568, 18.521])
# num = np.array([5, 14, 80, 100, 100])

# mean = (psnr*num).sum() / num.sum()
# print(mean)