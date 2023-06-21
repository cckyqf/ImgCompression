import torch
import torch.nn as nn
import numpy as np
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import torchvision
import os
import argparse
from model import encoder, net_encoder
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
import cv2
import zipfile

from general import select_device, increment_path, norm_256
import random
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='CSNet')

parser.add_argument('--testpath', type=str, default='../test_img/128/', 
                    help='Sun-Hays80,BSD100,urban100' )#测试图像文件夹

# parser.add_argument('--model',default='./runs/train/exp56/model/best.pt')
parser.add_argument('--model',default='./runs/train/exp57/model/best.pt')
parser.add_argument('--save_path',default='./')#重建图像保存的文件夹

parser.add_argument('--project', default='runs/val', help='save to project/name')
parser.add_argument('--name', default='exp', help='save to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')

parser.add_argument('--batch-size',type=int,default=1,metavar='N')
parser.add_argument('--image-size',type=int,default=128,metavar='N')
parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

parser.add_argument('--cuda',action='store_true',default=True)
parser.add_argument('--ngpu',type=int,default=1,metavar='N')
parser.add_argument('--seed',type=int,default=1,metavar='S')
parser.add_argument('--log-interval',type=int,default=100,metavar='N')

parser.add_argument('--base',type=int,default=32)#卷积核的数量
parser.add_argument('--M',type=int,default=8)
parser.add_argument('--mode',type=str,default="test")
parser.add_argument('--cr',type=int,default=8)
opt = parser.parse_args()


# 测试中间结果的保存路径
opt.save_path = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
print(f"results saved to {opt.save_path}")
if not os.path.exists('%s/orig' % (opt.save_path)):
    os.makedirs('%s/orig' % (opt.save_path))
if not os.path.exists('%s/recon' % (opt.save_path)):
    os.makedirs('%s/recon' % (opt.save_path))
if not os.path.exists('%s/measurement' % (opt.save_path)):
    os.makedirs('%s/measurement' % (opt.save_path))
if not os.path.exists('%s/map' % (opt.save_path)):
    os.makedirs('%s/map' % (opt.save_path))


if opt.seed is None:
    opt.seed = np.random.randint(1,10000)
print('Random seed: ',opt.seed)
random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

device = select_device(opt.device, batch_size=opt.batch_size)

criterion_mse = nn.MSELoss().to(device)


def calc_ent(x):
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp
    return ent

def data_loader():
    kwopt = {'num_workers': 8, 'pin_memory': True} if opt.cuda else {}
    transforms = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((opt.image_size, opt.image_size)),
                    torchvision.transforms.Grayscale(num_output_channels=1),
                    torchvision.transforms.ToTensor(),
                    # torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                        ])
    dataset = torchvision.datasets.ImageFolder(opt.testpath,transform=transforms)
    test_loader = torch.utils.data.DataLoader(dataset,batch_size = opt.batch_size,shuffle = False,**kwopt)
    return dataset, test_loader


@torch.no_grad()
def evaluation(test_dataset, testloader):

    channels = 1
    
    G_path = opt.model
    ckpt = torch.load(G_path, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak

    trained_epochs = ckpt["epoch"]
    print(f"训练epoch:{trained_epochs}")
    
    G = encoder(channels,opt.base,opt.M,opt.mode)
    state_dict = {k1:v2 for (k1,v1), (k2,v2) in zip(G.state_dict().items(), ckpt["model"].state_dict().items())}
    G.load_state_dict(state_dict)

    G.mode = "test"
    G.to(device)
    G.eval()

    # 获得
    classnames = test_dataset.classes
    class_to_idx = test_dataset.class_to_idx

    data_range = {
        "input": [],
        "y1_0": [],
        
        "y2_0": [],
        "y2_1": [],
        
        "y3_0": [],
        "y3_1": [],

        "sample1.conv": [],
        "sample2.depth_conv": [],
        "sample2.point_conv": [],
        "sample3.depth_conv": [],
        "sample3.point_conv": [],
    }
    mse_total = 0
    for idx, (input, class_id) in tqdm(enumerate(testloader), total=len(testloader)):
        
        classname = classnames[class_id.item()]


        input = input.to(device)
        input = norm_256(input)

        # 第一层
        y1_0 = G.sample1.conv(input)
        y1_1 = G.sample1.act(y1_0)

        y2_0 = G.sample2.c_shuffle(y1_1)
        y2_0 = G.sample2.depth_conv(y2_0)
        y2_1 = G.sample2.point_conv(y2_0)
        y2_2 = G.sample2.act(y2_1)


        y3_0 = G.sample3.depth_conv(y2_2)
        y3_1 = G.sample3.point_conv(y3_0)
        

        data_range['input'].append(input)
        data_range['y1_0'].append(y1_0)

        data_range['y2_0'].append(y2_0)
        data_range['y2_1'].append(y2_1)


        data_range['y3_0'].append(y3_0)
        data_range['y3_1'].append(y3_1)


        data_range['sample1.conv'].append(G.sample1.conv.weight)
        data_range['sample2.depth_conv'].append(G.sample2.depth_conv.weight)
        data_range['sample2.point_conv'].append(G.sample2.point_conv.weight)
        data_range['sample3.depth_conv'].append(G.sample3.depth_conv.weight)
        data_range['sample3.point_conv'].append(G.sample3.point_conv.weight)

    return data_range

def main():
    test_dataset, test_loader = data_loader()
    data_range = evaluation(test_dataset, test_loader)

    for k,v in data_range.items():
        
        v = torch.stack(v, dim=1)

        plt.figure()
        plt.hist(v.cpu().detach().numpy().reshape(-1))
        plt.savefig(k+".png")
        plt.close()
        
        min = v.min().item()
        max = v.max().item()

        print(f"{k}: max:{max}, min:{min}")


if __name__ == '__main__':
    main()

