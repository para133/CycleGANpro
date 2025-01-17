import os
import itertools
from torch.utils.tensorboard import SummaryWriter
import argparse
from torch.utils.data.dataloader import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn

from util.util import train_one_epoch, val
from model.CycleGAN import Cycle_Gan_G, Cycle_Gan_HD, Cycle_Gan_LD
from data.dataset import CreateDatasets
from util.image_pool import ImagePool

os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

def train(opt):
    # 训练超参数设置
    batch = opt.batch
    data_path = opt.dataPath
    print_every = opt.every
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    epochs = opt.epoch
    img_size = opt.imgsize
    
    # 创建保存权重的文件夹
    if not os.path.exists(opt.savePath):
        os.mkdir(opt.savePath)
 
    # 加载数据集
    train_datasets = CreateDatasets(data_path, img_size, mode='train')
    val_datasets = CreateDatasets(data_path, img_size, mode='test')
 
    train_loader = DataLoader(dataset=train_datasets, batch_size=batch, shuffle=True, num_workers=opt.numworker,
                              drop_last=True)
    val_loader = DataLoader(dataset=val_datasets, batch_size=batch, shuffle=True, num_workers=opt.numworker,
                            drop_last=True)
 
    # 实例化网络
    Cycle_G_A = Cycle_Gan_G().to(device)
    Cycle_HD_A = Cycle_Gan_HD().to(device)
    Cycle_LD_A = Cycle_Gan_LD().to(device)
    Cycle_G_B = Cycle_Gan_G().to(device)
    Cycle_HD_B = Cycle_Gan_HD().to(device)
    Cycle_LD_B = Cycle_Gan_LD().to(device)
 
    # 定义优化器
    # 生成器的优化器
    optim_G = optim.Adam(itertools.chain(Cycle_G_A.parameters(), Cycle_G_B.parameters()), lr=0.0001, betas=(0.5, 0.999), weight_decay=0.0001)
    # 判别器的优化器
    optim_D = optim.Adam(itertools.chain(Cycle_HD_A.parameters(), Cycle_LD_A.parameters(), Cycle_HD_B.parameters(), Cycle_LD_B.parameters()), lr=0.0001, betas=(0.5, 0.999), weight_decay=0.0001)
    
    # 定义损失函数
    # 对于真伪图像的判断需要使用MSELoss
    loss = nn.MSELoss()
    # 对于循环一致性(即重建产生的图像)损失使用L1Loss
    l1_loss = nn.L1Loss()
    
    start_epoch = 0
    
    # 初始化图像池，里面存放了50张模型生成的图像即假图像
    A_fake_pool = ImagePool(50)
    B_fake_pool = ImagePool(50)
 
    # 加载预训练权重
    if opt.weight != '':
        ckpt = torch.load(opt.weight)
        Cycle_G_A.load_state_dict(ckpt['Ga_model'], strict=False)
        Cycle_G_B.load_state_dict(ckpt['Gb_model'], strict=False)
        Cycle_HD_A.load_state_dict(ckpt['HDa_model'], strict=False)
        Cycle_LD_A.load_state_dict(ckpt['LDa_model'], strict=False)
        Cycle_HD_B.load_state_dict(ckpt['HDb_model'], strict=False)
        Cycle_LD_B.load_state_dict(ckpt['LDb_model'], strict=False)

        start_epoch = ckpt['epoch'] + 1
 
    writer = SummaryWriter('train_logs')
    # 开始训练
    for epoch in range(start_epoch, epochs):
        loss_mG, loss_mD = train_one_epoch(Ga=Cycle_G_A, HDa=Cycle_HD_A, LDa=Cycle_LD_A,Gb=Cycle_G_B, HDb=Cycle_HD_B, LDb=Cycle_LD_B,
                                           train_loader=train_loader,
                                           optim_G=optim_G, optim_D=optim_D, writer=writer, loss=loss, device=device,
                                           plot_every=print_every, epoch=epoch, l1_loss=l1_loss,
                                           A_fake_pool=A_fake_pool, B_fake_pool=B_fake_pool)
 
        writer.add_scalars(main_tag='train_loss', tag_scalar_dict={
            'loss_G': loss_mG,
            'loss_D': loss_mD
        }, global_step=epoch)

        if (epoch + 1) % 5 == 0:
            # 保存模型
            torch.save({
                'Ga_model': Cycle_G_A.state_dict(),
                'Gb_model': Cycle_G_B.state_dict(),
                'HDa_model': Cycle_HD_A.state_dict(),
                'LDa_model': Cycle_LD_A.state_dict(),
                'HDb_model': Cycle_HD_B.state_dict(),
                'LDb_model': Cycle_LD_B.state_dict(),
                'epoch': epoch
            }, opt.savePath + f'/{opt.dataPath}_{epoch}epoch.pth')
 
        # 验证集
        val(Ga=Cycle_G_A, HDa=Cycle_HD_A, LDa=Cycle_LD_A,Gb=Cycle_G_B, HDb=Cycle_HD_B, LDb=Cycle_LD_B,val_loader=val_loader, loss=loss, l1_loss=l1_loss,
            device=device, epoch=epoch, val_result_path=opt.val_result_path)
 
 
 
def cfg():
    parse = argparse.ArgumentParser()
    parse.add_argument('--batch', type=int, default=1)
    parse.add_argument('--epoch', type=int, default=100)
    parse.add_argument('--imgsize', type=int, default=256)
    parse.add_argument('--dataPath', type=str, default='./selfie2anime', help='data root path')
    parse.add_argument('--weight', type=str, default='', help='load pre train weight')
    parse.add_argument('--savePath', type=str, default='./weights', help='weight save path')
    parse.add_argument('--numworker', type=int, default=4)
    parse.add_argument('--every', type=int, default=20, help='plot train result every * iters')
    parse.add_argument('--val_result_path', type=str, default='val_selfie2anime_result', help='validation result save path') 
    opt = parse.parse_args()
    return opt
 
 
if __name__ == '__main__':
    opt = cfg()
    print(opt)
    train(opt)