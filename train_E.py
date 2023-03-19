import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import torchvision
import os
import argparse
from model import net_DWConv, net_paper, net_SASA2, net_DW_Trans, net_DW_CBAM
from torch.utils.tensorboard import SummaryWriter

'''在损失函数中加入了对y的熵的约束，使其尽可能服从均匀分布，即01个数接近'''
from tqdm import tqdm
import random
import yaml
from general import increment_path, select_device, norm_256
from pathlib import Path

def parse_opt():

    parser = argparse.ArgumentParser(description='CSNet')
    parser.add_argument('--model_name', type=str, default='net_DWConv', 
                        help='i.e. net_paper, net_SATA2, net_DWConv, net_DW_Trans, net_DW_CBAM')
    parser.add_argument('--trainpath',default='../Image/256_size/train/')
    parser.add_argument('--valpath',default='../Image/128_size/val/')
    parser.add_argument('--batch-size',type=int,default=16,metavar='N')
    parser.add_argument('--image-size',type=int,default=128,metavar='N')
    parser.add_argument('--start_epoch',type=int,default=0,metavar='N')#加载checkpoint即会改变
    # parser.add_argument('--epochs',type=int,default=15000,metavar='N')
    # parser.add_argument('--lr_deacy',type=float,default=5000,metavar='LR') # 第几个epoch开始衰减
    parser.add_argument('--epochs',type=int,default=120,metavar='N')
    parser.add_argument('--lr_deacy',type=float,default=80,metavar='LR') # 第几个epoch开始衰减
    parser.add_argument('--lr',type=float,default=1e-4,metavar='LR')

    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')

    parser.add_argument('--cuda',action='store_true',default=True)
    parser.add_argument('--ngpu',type=int,default=1,metavar='N')
    parser.add_argument('--seed',type=int,default=1,metavar='S')
    parser.add_argument('--log-interval',type=int,default=20,metavar='N')
    parser.add_argument('--save_path',default='./test')
    parser.add_argument('--outf',default='./')
    parser.add_argument('--base',type=int,default=32)#卷积核的数量
    parser.add_argument('--M',type=int,default=8)#y的通道数
    parser.add_argument('--gamma',type=float,default=0.01)#率失真的比例
    parser.add_argument('--mode',type=str,default="train")
    parser.add_argument('--resume',action='store_true',default=False)#是否加载checkpoint，默认True即可
    opt = parser.parse_args()

    return opt


def compute_loss(criterion_mse, output, target, E_y, ngpu, gamma):

    if E_y < 1.0 * ngpu:
        loss_r = gamma * E_y
    else:
        loss_r = torch.tensor([0], device=E_y.device)
    loss_mse = criterion_mse(output,target)
    loss = loss_mse - loss_r

    return loss, loss_mse.item(), loss_r.item()



def data_loader(opt):
    kwopt = {'num_workers': 8, 'pin_memory': True} if opt.cuda else {}
    train_transforms = torchvision.transforms.Compose([
                    torchvision.transforms.RandomCrop(opt.image_size),#随机剪裁
                    torchvision.transforms.Grayscale(num_output_channels=1),#转化为灰度图
                    torchvision.transforms.RandomHorizontalFlip(),#依照概率水平翻转
                    torchvision.transforms.RandomVerticalFlip(),#依照概率垂直翻转
                    torchvision.transforms.ToTensor(),#转化为tensor
                    # torchvision.transforms.Normalize((0.5, ), (0.5, ))#归一化
                    # torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                        ])
    
    val_transforms = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((opt.image_size, opt.image_size)),#随机剪裁
                    torchvision.transforms.Grayscale(num_output_channels=1),#转化为灰度图
                    torchvision.transforms.ToTensor(),#转化为tensor
                                        ])
    
    train_dataset = torchvision.datasets.ImageFolder(opt.trainpath,transform=train_transforms)
    val_dataset = torchvision.datasets.ImageFolder(opt.valpath,transform=val_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = opt.batch_size,shuffle = True,**kwopt)
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size = opt.batch_size,shuffle = False,**kwopt)
    return train_loader, val_loader



def train(writer, device, opt):

    trainloader, valloader = data_loader(opt)

    start_epoch, epochs = opt.start_epoch,opt.epochs

    # 灰度图像，单通道
    channels = 1
    G = eval(opt.model_name)(channels,opt.base,opt.M,opt.mode)
    G.to(device)

    # 网络的前向传播过程写入tensorboard
    # [0,1]之间均匀分布
    example = torch.rand((1, channels, opt.image_size,opt.image_size), device=device)
    writer.add_graph(torch.jit.trace(G, example, strict=False), [])

    for m in G.modules():
        if isinstance(m, (nn.Conv2d)) or isinstance(m, (nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in',nonlinearity='relu')

    optimizer_G = optim.Adam(G.parameters(),lr=opt.lr,betas=(0.9,0.999))
    scheduler_G = optim.lr_scheduler.MultiStepLR(optimizer_G, [opt.lr_deacy], gamma = 0.1, last_epoch=-1)
   
    criterion_mse = nn.MSELoss().to(device)

    if opt.resume:
        if os.path.isfile('%s/checkpoint' % (opt.outf)):
            checkpoint = torch.load('%s/checkpoint' % (opt.outf))
            start_epoch = checkpoint['epoch'] + 1
            G.load_state_dict(checkpoint['model'])
            optimizer_G.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found")


    min_loss = 100000
    num_batch = len(trainloader)
    for epoch in range(start_epoch,epochs):
        loss_total = 0
        mse_total = 0
        E_y_total = 0

        G.train()
        pbar = enumerate(trainloader)
        pbar = tqdm(pbar, total=num_batch, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for idx, (input, _) in pbar:

            input = input.to(device)
            input = norm_256(input)
            target = torch.tensor(input.cpu().numpy(), device=device)
            
            optimizer_G.zero_grad()
    
            output, E_y = G(input)

            loss, loss_mse_item, loss_r_item = compute_loss(criterion_mse, output, target, E_y, opt.ngpu, opt.gamma)

            loss.backward()
            optimizer_G.step()
            
            loss_item = loss.item()
            E_y_item = E_y.item()
            
            loss_total += loss_item
            mse_total += loss_mse_item
            E_y_total += E_y_item
            
            if idx% opt.log_interval == 0:
                # 打印一下损失等
                pbar.set_description('[%d/%d][%d/%d] loss:%.4f, mse:%.4f, E_y:%.4f' % (epoch,epochs,idx,num_batch,loss_item,loss_mse_item,E_y_item))
        
        
        lr = scheduler_G.get_last_lr()
        assert len(lr) == 1
        writer.add_scalar('lr/lr', lr[0], epoch)
        scheduler_G.step()

        writer.add_scalar('train/loss', loss_total/num_batch, epoch)
        writer.add_scalar('train/mse',  mse_total/num_batch, epoch)
        writer.add_scalar('train/E_y',  E_y_total/num_batch, epoch)

        a = vutils.make_grid(input[:2].detach(),normalize=True,scale_each=True)
        b = vutils.make_grid(output[:2].detach(),normalize=True,scale_each=True)

        writer.add_image('orin',a,epoch)
        writer.add_image('recon',b,epoch)

        G.eval()
        average_mse = val(epoch,channels,valloader,G,criterion_mse, writer, opt)

        if average_mse < min_loss:
            min_loss = average_mse
            print(f"epoch:{epoch} save model that has min_mse_loss")
            checkpoint = {
                'epoch': epoch,
                'model': G,
            }    
            torch.save(checkpoint, '%s/model/best.pt' % (opt.outf))

        checkpoint = {
            'epoch': epoch,
            'model': G,
            'optimizer': optimizer_G.state_dict(),
        }
        torch.save(checkpoint, '%s/model/last.pt' % (opt.outf))


@torch.no_grad()
def val(epoch,channels,valloader,G,criterion_mse, writer, opt):
    
    device = next(G.parameters()).device

    G.eval()
    
    loss_total = 0
    mse_total = 0
    E_y_total = 0

    num_batch = len(valloader)
    pbar = enumerate(valloader)
    pbar = tqdm(pbar, total=num_batch, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    for idx, (input, _) in pbar:

        input = input.to(device)
        input = norm_256(input)
        target = torch.tensor(input.cpu().numpy(), device=device)

        output,E_y = G(input)

        loss, loss_mse_item, loss_r_item = compute_loss(criterion_mse, output, target, E_y, opt.ngpu, opt.gamma)

        loss_total += loss.item()
        mse_total += loss_mse_item
        E_y_total += E_y.item()

    loss_average = loss_total / num_batch
    mse_average = mse_total / num_batch
    E_y_average = E_y_total / num_batch

    # print('Test:[%d] loss:%.4f, mse:%.4f, E_y:%.4f' % (epoch,loss_average,mse_average,E_y_average))

    pbar.set_description('Test:[%d] loss:%.4f, mse:%.4f, E_y:%.4f' % (epoch,loss_average,mse_average,E_y_average))

    writer.add_scalar('test/loss',loss_average, epoch)
    writer.add_scalar('test/mse', mse_average, epoch)
    writer.add_scalar('test/E_y', E_y_average, epoch)

    return mse_average


def main():
    opt = parse_opt()

    if opt.seed is None:
        opt.seed = np.random.randint(1,10000)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)
    
    # cudnn.benchmark, cudnn.deterministic = False, True
    cudnn.benchmark, cudnn.deterministic = True, False

    device = select_device(opt.device, batch_size=opt.batch_size)

    # 训练中间结果的保存路径
    opt.outf = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    print(f"results saved to {opt.outf}")
    if not os.path.exists('%s/model' % (opt.outf)):
        os.makedirs('%s/model' % (opt.outf))
    # 参数保存下来
    with open(opt.outf + '/opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)    
    writer = SummaryWriter(log_dir=opt.outf)

    
    train(writer, device, opt)

if __name__ == '__main__':
    
    main()

