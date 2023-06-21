import torch
import torch.nn as nn
import numpy as np
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import torchvision
import os
import argparse
from model import net_DWConv, net_paper, net_SASA2
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
import cv2
import zipfile

from general import select_device, increment_path, norm_256
import random
from pathlib import Path
from tqdm import tqdm
import time

parser = argparse.ArgumentParser(description='CSNet')

parser.add_argument('--testpath', type=str, default='../test_img/128/', 
                    help='Sun-Hays80,BSD100,urban100' )#测试图像文件夹

parser.add_argument('--model',default='./runs/train/exp44/model/best.pt')
parser.add_argument('--image-size',type=int,default=128,metavar='N')

parser.add_argument('--save_path',default='./')#重建图像保存的文件夹

parser.add_argument('--project', default='runs/val', help='save to project/name')
parser.add_argument('--name', default='exp', help='save to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')

parser.add_argument('--batch-size',type=int,default=1,metavar='N')
parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

parser.add_argument('--half',action='store_true',default=False)

parser.add_argument('--cuda',action='store_true',default=True)
parser.add_argument('--ngpu',type=int,default=1,metavar='N')
parser.add_argument('--seed',type=int,default=1,metavar='S')
parser.add_argument('--log-interval',type=int,default=100,metavar='N')

parser.add_argument('--base',type=int,default=32)#卷积核的数量
parser.add_argument('--M',type=int,default=8)
parser.add_argument('--mode',type=str,default="test")
parser.add_argument('--cr',type=int,default=8)
opt = parser.parse_args()



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

from typing import Any, Tuple
class ImageFolderPath(torchvision.datasets.ImageFolder):

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path

def data_loader():
    kwopt = {'num_workers': 8, 'pin_memory': True} if opt.cuda else {}
    transforms = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((opt.image_size, opt.image_size)),
                    torchvision.transforms.Grayscale(num_output_channels=1),
                    torchvision.transforms.ToTensor(),
                    # torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                        ])
    dataset = ImageFolderPath(opt.testpath,transform=transforms)
    test_loader = torch.utils.data.DataLoader(dataset,batch_size = opt.batch_size,shuffle = False,**kwopt)
    return dataset, test_loader

@torch.no_grad()
def evaluation(test_dataset, testloader):

    channels = 1
    
    G_path = opt.model
    ckpt = torch.load(G_path, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak

    trained_epochs = ckpt["epoch"]
    print(f"训练epoch:{trained_epochs}")
    
    G = ckpt["model"]
    # G = net_paper(channels,64,opt.M,opt.mode)
    # state_dict = {k1:v2 for (k1,v1), (k2,v2) in zip(G.state_dict().items(), ckpt["model"].state_dict().items())}
    # G.load_state_dict(state_dict)


    G.mode = "test"
    G.to(device)
    G.eval()

    num_img = 0
    total_time = 0
    num_loop = 10
    for i in range(num_loop):
        for idx, (input, class_id, img_path) in tqdm(enumerate(testloader), total=len(testloader)):
            
            input = input.to(device)
            input = norm_256(input)

            t1 = time.time()
            y = G.sample1(input)
            y = G.sample2(y)
            y = G.sample3(y)
            zeros_tensor = torch.zeros_like(y)
            ones_tensor = torch.ones_like(y)
            y = torch.where(y < 0.5, zeros_tensor, ones_tensor)
            t2 = time.time()

            num_img += 1
            total_time += t2-t1

    print('Test: average time to encode: %.4f ms,' % (total_time*1000 / num_img))

def main():
    test_dataset, test_loader = data_loader()

    
    evaluation(test_dataset, test_loader)


if __name__ == '__main__':
    main()