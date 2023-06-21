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

parser = argparse.ArgumentParser(description='CSNet')

parser.add_argument('--testpath', type=str, default='../test_img/128/', 
                    help='Sun-Hays80,BSD100,urban100' )#测试图像文件夹

parser.add_argument('--model',default='./runs/train/exp57/model/best.pt')
parser.add_argument('--save_path',default='./')#重建图像保存的文件夹

parser.add_argument('--project', default='runs/val', help='save to project/name')
parser.add_argument('--name', default='exp', help='save to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')

parser.add_argument('--batch-size',type=int,default=1,metavar='N')
parser.add_argument('--image-size',type=int,default=128,metavar='N')
parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

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
    
    G = ckpt["model"]
    # G = net_paper(channels,64,opt.M,opt.mode)
    # state_dict = {k1:v2 for (k1,v1), (k2,v2) in zip(G.state_dict().items(), ckpt["model"].state_dict().items())}
    # G.load_state_dict(state_dict)


    G.mode = "test"
    G.to(device)
    G.eval()

    # G_fuse = torch.quantization.fuse_modules(G, [['fc', 'relu']])
    # print(G_fuse)

    
    backend = "fbgemm"
    G.qconfig = torch.quantization.get_default_qconfig(backend)  # 不同平台不同配置

    # listmix = [['conv','relu']] # 可以是conv+bn conv+relu conv+bn+relu 
    # G_fuse = torch.quantization.fuse_modules(G, listmix) # 合并某些层，不想合并这句也可以跳过

    G_prepared = torch.quantization.prepare(G)
    G_int8 = torch.quantization.convert(G_prepared)

    G_int8.eval()

    # 获得
    classnames = test_dataset.classes
    class_to_idx = test_dataset.class_to_idx

    half = opt.half
    if half:
        G.half()

    mse_total = 0
    for idx, (input, class_id) in tqdm(enumerate(testloader), total=len(testloader)):
        
        classname = classnames[class_id.item()]


        input = input.to(device)
        input = norm_256(input)

        if half:
            input = input.half()

        target = torch.tensor(input.cpu().numpy(), device=device)

        # output, y = G(input)
        output, y = G_int8(input)

        # y = (y > 0.5).float()
        y_save = y.cpu().detach().numpy()

        save_name = classname + '_' + str(idx)
        np.savetxt('%s/measurement/%s.txt' %(opt.save_path, save_name),y_save.ravel(),fmt='%d')
        
        mse = criterion_mse(output,target)
        mse_total += mse.item()

        z = zipfile.ZipFile('%s/measurement/%s.zip'%(opt.save_path, save_name),'w', zipfile.ZIP_DEFLATED) #压缩
        z.write('%s/measurement/%s.txt' %(opt.save_path, save_name))

        # print('Test:[%d/%d] mse:%.4f \n' % (idx,len(testloader),mse.item()))

        vutils.save_image(target.data,'%s/orig/%s.bmp'% (opt.save_path, save_name), padding=0)
        vutils.save_image(output.data,'%s/recon/%s.bmp' % (opt.save_path, save_name), padding=0)

    print('Test: average mse: %.4f,' % (mse_total / len(testloader)))

def main():
    test_dataset, test_loader = data_loader()
    evaluation(test_dataset, test_loader)

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



if __name__ == '__main__':
    main()