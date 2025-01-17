import torchvision
from tqdm import tqdm
import torch
import os


def train_one_epoch(Ga, HDa, LDa, Gb, HDb, LDb, train_loader, optim_G, optim_D, writer, loss, device, plot_every, epoch, l1_loss,
                    A_fake_pool, B_fake_pool):
    pd = tqdm(train_loader)
    loss_D, loss_G = 0, 0
    step = 0
    Ga.train()
    HDa.train()
    LDa.train()
    Gb.train()
    HDb.train()
    LDb.train()
    for idx, data in enumerate(pd):
        A_real = data[0].to(device)
        B_real = data[1].to(device)
        
        # 前向传递
        B_fake = Ga(A_real)  # Ga生成的假B
        A_rec = Gb(B_fake)  # Gb重构回的A
        A_fake = Gb(B_real)  # Gb生成的假A
        B_rec = Ga(A_fake)  # Ga重构回的B

        # 训练生成器   生成器包含六部分损失
        set_required_grad([HDa, LDa, HDb, LDb], requires_grad=False)  # 不更新判别器
        optim_G.zero_grad()
        ls_G = train_G(HDa=HDa, LDa= LDa, HDb=HDb, LDb=LDb, B_fake=B_fake, loss=loss, A_fake=A_fake, l1_loss=l1_loss,
                       A_rec=A_rec, A_real=A_real, B_rec=B_rec, B_real=B_real, Ga=Ga, Gb=Gb)
        ls_G.backward()
        optim_G.step()

        # 训练判别器
        set_required_grad([HDa, LDa, HDb, LDb], requires_grad=True)
        optim_D.zero_grad()
        # 从图像池中取出假图像
        A_fake_p = A_fake_pool.query(A_fake)
        B_fake_p = B_fake_pool.query(B_fake)
        ls_D = train_D(HDa=HDa, LDa=LDa, HDb=HDb, LDb = LDb, B_fake=B_fake_p, B_real=B_real, loss=loss, A_fake=A_fake_p, A_real=A_real)
        ls_D.backward()
        optim_D.step()

        loss_D += ls_D
        loss_G += ls_G

        pd.desc = 'train_{} G_loss: {} D_loss: {}'.format(epoch, ls_G.item(), ls_D.item())

        # 绘制训练结果
        if idx % plot_every == 0:
            writer.add_images(tag='epoch{}_Ga'.format(epoch), img_tensor=0.5 * (torch.cat([A_real, B_fake], 0) + 1),
                              global_step=step)
            writer.add_images(tag='epoch{}_Gb'.format(epoch), img_tensor=0.5 * (torch.cat([B_real, A_fake], 0) + 1),
                              global_step=step)
            step += 1
    mean_lsG = loss_G / len(train_loader)
    mean_lsD = loss_D / len(train_loader)
    return mean_lsG, mean_lsD


@torch.no_grad()
def val(Ga, HDa, LDa, Gb, HDb, LDb,val_loader, loss, device, l1_loss, epoch, val_result_path):
    pd = tqdm(val_loader)
    loss_D, loss_G = 0, 0
    Ga.eval()
    HDa.eval()
    LDa.eval()
    Gb.eval()
    HDb.eval()
    LDb.eval()
    all_loss = 10000
    for idx, item in enumerate(pd):
        A_real_img = item[0].to(device)
        B_real_img = item[1].to(device)

        B_fake_img = Ga(A_real_img)
        A_fake_img = Gb(B_real_img)

        A_rec = Gb(B_fake_img)
        B_rec = Ga(A_fake_img)

        # 判别器的loss
        ls_D = train_D(HDa=HDa, LDa=LDa, HDb=HDb, LDb = LDb, B_fake=B_fake_img, B_real=B_real_img, loss=loss, A_fake=A_fake_img,
                       A_real=A_real_img)
        
        # 生成器的loss
        ls_G = train_G(HDa=HDa, LDa= LDa, HDb=HDb, LDb = LDb, B_fake=B_fake_img, loss=loss, A_fake=A_fake_img, l1_loss=l1_loss,
                       A_rec=A_rec, A_real=A_real_img, B_rec=B_rec, B_real=B_real_img, Ga=Ga, Gb=Gb)

        loss_G += ls_G
        loss_D += ls_D
        pd.desc = 'val_{}: G_loss:{} D_Loss:{}'.format(epoch, ls_G.item(), ls_D.item())

        # 保存最好的结果
        all_ls = ls_G + ls_D
        if all_ls < all_loss:
            all_loss = all_ls
            best_image = torch.cat([A_real_img, B_fake_img, B_real_img, A_fake_img], 0)
    result_img = (best_image + 1) * 0.5
    if not os.path.exists(val_result_path):
        os.mkdir(val_result_path)

    torchvision.utils.save_image(result_img,  val_result_path + '/val_epoch{}_cycle.jpg'.format(epoch))


def set_required_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for params in net.parameters():
                params.requires_grad = requires_grad


def train_G(HDa, LDa, HDb, LDb, B_fake, loss, A_fake, l1_loss, A_rec, A_real, B_rec, B_real, Ga, Gb):
    # 判别损失
    HDa_out_fake = HDa(B_fake) # A判别器对由A生成的假B图像的判别结果
    LDa_out_fake = LDa(B_fake)
    # 计算生成器A的判别MSELoss
    Ga_gan_loss = loss(HDa_out_fake, torch.ones(HDa_out_fake.size()).cuda()) + loss(LDa_out_fake, torch.ones(LDa_out_fake.size()).cuda()) # Da判别的是图像是否为B，我们希望此时判别器认为是B，所以全为1，这里判别器不进行更新，所以全为1也没事 
    HDb_out_fake = HDb(A_fake) # B判别器对由B生成的假A图像的判别结果
    LDb_out_fake = LDb(A_fake)
    # 计算生成器B的判别MSELoss
    Gb_gan_loss = loss(HDb_out_fake, torch.ones(HDb_out_fake.size()).cuda())  + loss(LDb_out_fake, torch.ones(LDb_out_fake.size()).cuda())# Db判别的是图像是否为A，我们希望此时判别器认为是A，所以全为1，这里判别器不进行更新，所以全为1也没事 

    # 重建损失
    Cycle_A_loss = l1_loss(A_rec, A_real) * 10
    Cycle_B_loss = l1_loss(B_rec, B_real) * 10

    # identity loss
    Ga_id_out = Ga(B_real) # 生成器A对B的真实图像的转化结果，这里转化结果应该与B_real相似，所以也要最小化误差
    Gb_id_out = Gb(A_real) # 生成器B对A的真实图像的转化结果
    Ga_id_loss = l1_loss(Ga_id_out, B_real) * 10 * 0.5 # 生成器A对B的真实图像的转化结果与B的真实图像的L1Loss，这里转化结果应该与B_real相似，所以也要最小化误差
    Gb_id_loss = l1_loss(Gb_id_out, A_real) * 10 * 0.5 # 生成器B对A的真实图像的转化结果与A的真实图像的L1Loss

    # G的总损失
    ls_G = Ga_gan_loss + Gb_gan_loss + Cycle_A_loss + Cycle_B_loss + Ga_id_loss + Gb_id_loss

    return ls_G


def train_D(HDa, LDa, HDb, LDb,B_fake, B_real, loss, A_fake, A_real):
    # Da的loss
    HDa_fake_out = HDa(B_fake.detach()).squeeze() # detach()是为了不更新生成器A的参数
    LDa_fake_out = LDa(B_fake.detach()).squeeze()
    HDa_real_out = HDa(B_real).squeeze()
    LDa_real_out = LDa(B_real).squeeze()
    ls_Da1 = loss(HDa_fake_out, torch.zeros(HDa_fake_out.size()).cuda()) + loss(LDa_fake_out, torch.zeros(LDa_fake_out.size()).cuda()) # Da判别的是图像是否为B，我们希望此时判别器认为是B，所以全为0
    ls_Da2 = loss(HDa_real_out, torch.ones(HDa_real_out.size()).cuda()) + loss(LDa_real_out, torch.ones(LDa_real_out.size()).cuda())# Da判别的是图像是否为B，我们希望此时判别器认为是B，所以全为1
    ls_Da = (ls_Da1 + ls_Da2) * 0.5
    # Db的loss
    HDb_fake_out = HDb(A_fake.detach()).squeeze()
    LDb_fake_out = LDb(A_fake.detach()).squeeze()
    HDb_real_out = HDb(A_real.detach()).squeeze()
    LDb_real_out = LDb(A_real.detach()).squeeze()
    ls_Db1 = loss(HDb_fake_out, torch.zeros(HDb_fake_out.size()).cuda()) + loss(LDb_fake_out, torch.zeros(LDb_fake_out.size()).cuda())
    ls_Db2 = loss(HDb_real_out, torch.ones(HDb_real_out.size()).cuda()) + loss(LDb_real_out, torch.ones(LDb_real_out.size()).cuda())
    ls_Db = (ls_Db1 + ls_Db2) * 0.5

    # D的总损失
    ls_D = ls_Da + ls_Db
    return ls_D
