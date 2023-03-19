import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import torchvision
import os
import argparse
import net_bilinear
from torchvision import datasets,transforms
from torch.autograd import Variable
from torch.nn import init
from tensorboardX import SummaryWriter
'''在损失函数中加入了对y的熵的约束，使其尽可能服从均匀分布，即01个数接近'''

os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser(description='CSNet')
parser.add_argument('--dataset',default='own_image')
parser.add_argument('--trainpath',default='../Image/train/')
parser.add_argument('--valpath',default='../Image/test/')
parser.add_argument('--batch-size',type=int,default=16,metavar='N')
parser.add_argument('--image-size',type=int,default=128,metavar='N')
parser.add_argument('--start_epoch',type=int,default=0,metavar='N')#加载checkpoint即会改变
parser.add_argument('--epochs',type=int,default=10000,metavar='N')
parser.add_argument('--lr',type=float,default=1e-4,metavar='LR')
parser.add_argument('--cuda',action='store_true',default=True)
parser.add_argument('--ngpu',type=int,default=1,metavar='N')
parser.add_argument('--seed',type=int,default=1,metavar='S')
parser.add_argument('--log-interval',type=int,default=20,metavar='N')
parser.add_argument('--save_path',default='./test')
parser.add_argument('--outf',default='./results_gamma0.01_bilinear')
parser.add_argument('--base',type=int,default=64)#卷积核的数量
parser.add_argument('--M',type=int,default=16)#y的通道数
parser.add_argument('--gamma',type=float,default=0.01)#率失真的比例
parser.add_argument('--mode',type=str,default="train")
parser.add_argument('--resume',action='store_true',default=True)#是否加载checkpoint，默认True即可
opt = parser.parse_args()

if torch.cuda.is_available() and not opt.cuda:
    print("please run with GPU")
if opt.seed is None:
    opt.seed = np.random.randint(1,10000)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)
cudnn.benchmark = True

if not os.path.exists('%s/model' % (opt.outf)):
    os.makedirs('%s/model' % (opt.outf))
if not os.path.exists('%s/log' % (opt.outf)):
    os.makedirs('%s/log' % (opt.outf))
log_dir = '%s/log' % (opt.outf)
writer = SummaryWriter(log_dir=log_dir)

def data_loader():
    kwopt = {'num_workers': 8, 'pin_memory': True} if opt.cuda else {}
    transforms = torchvision.transforms.Compose([
                    torchvision.transforms.RandomCrop(opt.image_size),#随机剪裁
                    torchvision.transforms.Grayscale(num_output_channels=1),#转化为灰度图
                    torchvision.transforms.RandomHorizontalFlip(),#依照概率水平翻转
                    torchvision.transforms.RandomVerticalFlip(),#依照概率垂直翻转
                    torchvision.transforms.ToTensor(),#转化为tensor
                    # torchvision.transforms.Normalize((0.5, ), (0.5, ))#归一化
                    # torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                        ])
    train_dataset = torchvision.datasets.ImageFolder(opt.trainpath,transform=transforms)
    val_dataset = torchvision.datasets.ImageFolder(opt.valpath,transform=transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = opt.batch_size,shuffle = True,**kwopt)
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size = opt.batch_size,shuffle = True,**kwopt)
    return train_loader, val_loader

def train(start_epoch,epochs,trainloader, valloader):
    input, _ = trainloader.__iter__().__next__()
    input = input.numpy()
    sz_input = input.shape#128*1*256*256
    channels = sz_input[1]#通道数（3）
    img_size = sz_input[3]#256

    input = torch.FloatTensor(opt.batch_size,channels,img_size,img_size)
    label = torch.FloatTensor(2)

    G = net_bilinear.Generater(channels,opt.base,opt.M,opt.mode)

    for m in G.modules():
        if isinstance(m, (nn.Conv2d)) or isinstance(m, (nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in',nonlinearity='relu')

    optimizer_G = optim.Adam(G.parameters(),lr=opt.lr,betas=(0.9,0.999))
    scheduler_G = optim.lr_scheduler.MultiStepLR(optimizer_G, [3500], gamma = 0.1, last_epoch=-1)
   
    criterion_mse = nn.MSELoss()
    cudnn.benchmark = True

    if opt.cuda:
        device_id = [0]
        G = nn.DataParallel(G.cuda(),device_ids = device_id)
        criterion_mse.cuda()
        input = input.cuda()

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
    for epoch in range(start_epoch,epochs):
        for idx, (input, _) in enumerate(trainloader, 0):
            if input.size(0) != opt.batch_size:
                continue
            G.train()
            G.zero_grad()
            output,E_y = G(input.cuda(),idx,opt.save_path)
            E_y = torch.sum(E_y)


            if E_y < 1.0 * opt.ngpu:
                loss_r = opt.gamma * E_y
            else:
                loss_r = torch.tensor([0]).cuda()

            loss_mse = criterion_mse(output,input.cuda())
            loss = loss_mse - loss_r

            loss.backward()
            optimizer_G.step()
            scheduler_G.step()

            if idx % opt.log_interval == 0:
                print('[%d/%d][%d/%d] mse:%.4f,loss_r:%.4f,E_y：%.4f' % (epoch,epochs,idx,len(trainloader),loss_mse.item(),loss_r.item(),E_y))

        writer.add_scalar('train/mse',loss, epoch)
        writer.add_scalar('train/E_y',E_y, epoch)
        a = vutils.make_grid(input[:2],normalize=True,scale_each=True)
        b = vutils.make_grid(output[:2],normalize=True,scale_each=True)

        writer.add_image('orin',a,epoch)
        writer.add_image('recon',b,epoch)

        G.eval()
        average_mse = val(epoch,channels,valloader,input,G,criterion_mse)

        if average_mse < min_loss:
            min_loss = average_mse
            print("save model")
            torch.save(G.state_dict(),'%s/model/G.pth' % (opt.outf))

        checkpoint = {
            'epoch': epoch,
            'model': G.state_dict(),
            'optimizer': optimizer_G.state_dict(),
        }
        torch.save(checkpoint,'%s/checkpoint' % (opt.outf))

def val(epoch,channels,valloader,input,G,criterion_mse):
    mse_total = 0
    average_mse = 0
    for idx, (input, _) in enumerate(valloader, 0):
        if input.size(0) != opt.batch_size:
            continue
        with torch.no_grad():
            output,E_y = G(input.cuda(),idx,opt.save_path)

            mse = criterion_mse(output,input.cuda())
            mse_total += mse
            average_mse = mse_total.item() / len(valloader)

    print('Test:[%d] average mse:%.4f' % (epoch,average_mse))
    writer.add_scalar('test/mse', average_mse, epoch)
    writer.add_scalar('test/E_y',E_y, epoch)

    return average_mse

def main():
    train_loader,val_loader = data_loader()
    train(opt.start_epoch,opt.epochs,train_loader,val_loader)

if __name__ == '__main__':
    main()

