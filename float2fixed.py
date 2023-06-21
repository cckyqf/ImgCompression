import torch
import torch.nn as nn
import numpy as np
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import torchvision
import os
import argparse
from model import encoder, net_encoder, net_decoder
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
import cv2
import zipfile
from copy import deepcopy
from general import select_device, increment_path, norm_256
import random
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='CSNet')

parser.add_argument('--testpath', type=str, default='../test_img/visual/', 
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
parser.add_argument('--fixed_point',action='store_true',default=True)


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


# 浮点数，转化为定点数
def float2fixed(data, exp_n):

    # print(data)

    scale = 2**exp_n
    # 
    data = data * scale

    # 向下取整
    data = torch.floor(data)
    # data = torch.floor(data+0.5)

    # data = data.type(torch.int16)

    data = data / scale

    # print(data)
    # print(data.dtype)
    # torch.int16

    return data


def model_float2fixed(model, model_exp_n):

    model.sample1.conv.weight.data = float2fixed(model.sample1.conv.weight.data, model_exp_n[0])

    model.sample2.depth_conv.weight.data = float2fixed(model.sample2.depth_conv.weight.data, model_exp_n[1])
    
    model.sample2.point_conv.weight.data = float2fixed(model.sample2.point_conv.weight.data, model_exp_n[2])

    model.sample3.depth_conv.weight.data = float2fixed(model.sample3.depth_conv.weight.data, model_exp_n[3])

    model.sample3.point_conv.weight.data = float2fixed(model.sample3.point_conv.weight.data, model_exp_n[4])


    return model



@torch.no_grad()
def evaluation(test_dataset, testloader):

    channels = 1
    
    G_path = opt.model
    ckpt = torch.load(G_path, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak

    trained_epochs = ckpt["epoch"]
    print(f"训练epoch:{trained_epochs}")
    
    G = net_decoder(channels,opt.base,opt.M,opt.mode)

    state_dict = {k1:v2 for (k1,v1), (k2,v2) in zip(G.state_dict().items(), ckpt["model"].state_dict().items())}
    G.load_state_dict(state_dict)

    G.mode = "test"
    G.to(device)
    G.eval()

    # 模型的编码端进行定点化
    # # exp56模型的16位定点化参数
    # model_exp_n = [14, 14, 14, 14, 14]
    # input_output_exp_n = [13, 12, 12, 12, 11]

    # exp40模型的16位定点化参数，即使用几个bits存储小数部分
    #  input_output_exp_n = [13, 12, 12, 12, 11]
    # model_exp_n = [14, 14, 15, 14, 14]

    # exp57模型的定点化参数，即使用几个bits存储小数部分
    input_output_exp_n = [13, 12, 12, 12, 11]
    model_exp_n = [14, 14, 14, 14, 14]
    


    # 如果改成18位定点数，则存储小数部分可以增加bits
    bits = 16
    add_n = bits - 16
    model_exp_n = list(map(lambda x:x+add_n, model_exp_n))
    input_output_exp_n = list(map(lambda x:x+add_n, input_output_exp_n))



    # G = model_float2fixed(G, model_exp_n)
    G_orig = deepcopy(G)
    G_fixed = model_float2fixed(G, model_exp_n)
    for (k1,v1),(k2,v2) in zip(G_orig.named_parameters(), G_fixed.named_parameters()):
        # print(id(v1), id(v2))
        diff = torch.abs(v1-v2)
        num = diff.numel()
        diff = diff.sum().item() / num

        # 只打印编码端的
        if "sample" in k1:
            print(k1, diff)

    # exit()

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
    for idx, (input, class_id, img_path) in tqdm(enumerate(testloader), total=len(testloader)):
        


        input = input.to(device)
        input = norm_256(input)
        target = torch.tensor(input.cpu().numpy(), device=device)


        # 第一层
        y1_0 = G_fixed.sample1.conv(input)
        y1_0 = float2fixed(y1_0, input_output_exp_n[0])
        y1_1 = G_fixed.sample1.act(y1_0)

        y2_0 = G_fixed.sample2.c_shuffle(y1_1)

        y2_0 = G_fixed.sample2.depth_conv(y2_0)
        y2_0 = float2fixed(y2_0, input_output_exp_n[1])
        y2_1 = G_fixed.sample2.point_conv(y2_0)
        y2_1 = float2fixed(y2_1, input_output_exp_n[2])
        y2_2 = G_fixed.sample2.act(y2_1)


        y3_0 = G_fixed.sample3.depth_conv(y2_2)
        y3_0 = float2fixed(y3_0, input_output_exp_n[3])
        y3_1 = G_fixed.sample3.point_conv(y3_0)
        y3_1 = float2fixed(y3_1, input_output_exp_n[4])


        # 量化
        zeros_tensor = torch.zeros_like(y3_1)
        ones_tensor = torch.ones_like(y3_1)
        y = torch.where(y3_1 < 0.5,zeros_tensor,ones_tensor)


        '''解码器'''
        output = G_fixed.recon_feat(y)
        output = G_fixed.layer0(output)

        output = G_fixed.layer1(output)
        output = G_fixed.layer2(output)
        output = G_fixed.layer3(output)

        # # 4x下采样
        # output = G_fixed.layer1(output)+F.interpolate(output, size=None, scale_factor=2, mode='bilinear', align_corners=None)
        # # 2x下采样
        # output = G_fixed.layer2(output)+F.interpolate(output, size=None, scale_factor=2, mode='bilinear', align_corners=None)
        # # 原始图像尺寸
        # output = G_fixed.layer3(output)+F.interpolate(output, size=None, scale_factor=2, mode='bilinear', align_corners=None)

        output = G_fixed.conv_d4(output)
        if hasattr(G_fixed, "output_act"):
            output = G_fixed.output_act(output)


        # output, y = G_fixed(input)


        # 准备计算PSNR/SSIM
        # y = (y > 0.5).float()
        y_save = y.cpu().detach().numpy()

        # save_name = classname + '_' + str(idx)
        img_path=img_path[0]
        save_name = os.path.splitext(os.path.basename(img_path))[0]

        np.savetxt('%s/measurement/%s.txt' %(opt.save_path, save_name),y_save.ravel(),fmt='%d')
        
        mse = criterion_mse(output,target)
        mse_total += mse.item()

        z = zipfile.ZipFile('%s/measurement/%s.zip'%(opt.save_path, save_name),'w', zipfile.ZIP_DEFLATED) #压缩
        z.write('%s/measurement/%s.txt' %(opt.save_path, save_name))

        # print('Test:[%d/%d] mse:%.4f \n' % (idx,len(testloader),mse.item()))

        vutils.save_image(target.data,'%s/orig/%s.bmp'% (opt.save_path, save_name), padding=0)
        vutils.save_image(output.data,'%s/recon/%s.bmp' % (opt.save_path, save_name), padding=0)




        # data_range['input'].append(input)
        # data_range['y1_0'].append(y1_0)

        # data_range['y2_0'].append(y2_0)
        # data_range['y2_1'].append(y2_1)


        # data_range['y3_0'].append(y3_0)
        # data_range['y3_1'].append(y3_1)


        # data_range['sample1.conv'].append(G.sample1.conv.weight)
        # data_range['sample2.depth_conv'].append(G.sample2.depth_conv.weight)
        # data_range['sample2.point_conv'].append(G.sample2.point_conv.weight)
        # data_range['sample3.depth_conv'].append(G.sample3.depth_conv.weight)
        # data_range['sample3.point_conv'].append(G.sample3.point_conv.weight)

    return data_range

def main():
    test_dataset, test_loader = data_loader()
    data_range = evaluation(test_dataset, test_loader)
    files = os.listdir('%s/orig/' % (opt.save_path))

    y_entropy_number = {}
    psnr_number = {}
    ssim_number = {}
    orig_fsize_number = {}
    y_fsize_number = {}

    for file_name in tqdm(files):

        name = os.path.splitext(file_name)[0]

        dataset_name = name.split('_')[0]
        if dataset_name not in y_entropy_number.keys():
            y_entropy_number[dataset_name] = []
            psnr_number[dataset_name] = []
            ssim_number[dataset_name] = []
            orig_fsize_number[dataset_name] = []
            y_fsize_number[dataset_name] = []

        y_entropy = calc_ent(np.loadtxt('%s/measurement/%s.txt' %(opt.save_path, name),dtype='int'))
        orig_img = cv2.imread('%s/orig/%s.bmp'% (opt.save_path, name), flags=cv2.IMREAD_GRAYSCALE)
        recon_img = cv2.imread('%s/recon/%s.bmp' % (opt.save_path, name), flags=cv2.IMREAD_GRAYSCALE)
        psnr = compare_psnr(orig_img,recon_img)
        ssim = compare_ssim(orig_img,recon_img,multichannel=False)

        ls = ["1_17__0__128.bmp", "1_17__0__256.bmp", "1_17__0__384.bmp"]

        if name+".bmp" in ls:
            print(name, psnr, ssim)

        orig_fsize = os.path.getsize('%s/orig/%s.bmp'% (opt.save_path, name))
        y_fsize = os.path.getsize('%s/measurement/%s.zip'%(opt.save_path, name))

        y_entropy_number[dataset_name].append(y_entropy)
        psnr_number[dataset_name].append(psnr)
        ssim_number[dataset_name].append(ssim)
        orig_fsize_number[dataset_name].append(orig_fsize)
        y_fsize_number[dataset_name].append(y_fsize)


    psnr_ls = []
    ssim_ls = []
    for dataset_name in y_entropy_number.keys():

        psnr = psnr_number[dataset_name]
        ssim = ssim_number[dataset_name]

        psnr_ls = psnr_ls + psnr
        ssim_ls = ssim_ls + ssim

        psnr = np.mean(psnr)
        ssim = np.mean(ssim)

        print("%s, psnr : %.5f, ssim : %.5f" % (dataset_name, psnr, ssim))

    # print('avg ideal compress factor : %.5f' %(opt.cr * (1 / y_entropy)))
    # print('avg actal compress factor : %.5f' %((opt.image_size * opt.image_size * 1)/y_fsize))

    psnr = np.mean(psnr_ls)
    ssim = np.mean(ssim_ls)
    print('avg psnr : %.5f' %(psnr))
    print('avg ssim : %.5f' %(ssim))

    # for k,v in data_range.items():
        
    #     v = torch.stack(v, dim=1)

    #     plt.figure()
    #     plt.hist(v.cpu().detach().numpy().reshape(-1))
    #     plt.savefig(k+".png")
    #     plt.close()
        
    #     min = v.min().item()
    #     max = v.max().item()

    #     print(f"{k}: max:{max}, min:{min}")


if __name__ == '__main__':
    main()
    # data = torch.rand((2,4)) * 100

    # data = float2fixed(data, exp_n=12)
