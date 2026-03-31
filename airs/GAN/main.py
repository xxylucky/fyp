from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os
import json
from data.tn3k import tn3kDataSet
from data.BUSI import BUSIDataSet
import models.dcgan as dcgan
import models.mlp as mlp
import warnings
from torch.nn import functional as F

warnings.filterwarnings("ignore", category=UserWarning)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--dataroot', help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nc', type=int, default=1, help='input image channels')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=10001, help='number of epochs to train for')  # 默认训练10000轮
    parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
    parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
    parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")  
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--clamp_lower', type=float, default=-0.01)
    parser.add_argument('--clamp_upper', type=float, default=0.01)
    parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
    parser.add_argument('--noBN', action='store_true', help='use batchnorm or not (only for DCGAN)')
    parser.add_argument('--mlp_G', action='store_true', help='use MLP for G')
    parser.add_argument('--mlp_D', action='store_true', help='use MLP for D')
    parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
    parser.add_argument('--experiment', default='/root/example/airs1/semi/code/pretrain/GAN', help='Where to store samples and models')  # 指定模型训练过程中保存模型参数（checkpoint）文件夹的名字。
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)') 
    parser.add_argument('--root', type=str, default='/root/example/airs1')  # 需要修改default为正确的路径/root/example/airs1
    parser.add_argument('--expID', type=int, default=1)  # 需要修改default：1--72 2--144 3--288
    opt = parser.parse_args()
    print(opt)

    if opt.experiment is None:
        opt.experiment = 'samples'
    os.system('mkdir {0}'.format(opt.experiment))

    opt.manualSeed = random.randint(1, 10000) # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

   
    if opt.dataset == 'tn3k':
        dataset = tn3kDataSet(opt.root, opt.expID, mode='train')
    elif opt.dataset == 'busi':  # 注意大小写！！！！！
        dataset = BUSIDataSet(opt.root, opt.expID, mode='train')
    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                            shuffle=True, num_workers=int(opt.workers))  

    ngpu = int(opt.ngpu) # number of gpu #1
    nz = int(opt.nz) # size of the latent z vector #100 
    ngf = int(opt.ngf) #64
    ndf = int(opt.ndf) #64
    nc = int(opt.nc) #input images channels #3
    n_extra_layers = int(opt.n_extra_layers) #Number of extra layers on gen and disc #0

    # write out generator config to generate images together wth training checkpoints (.pth)
    generator_config = {"imageSize": opt.imageSize, "nz": nz, "nc": nc, "ngf": ngf, "ngpu": ngpu, "n_extra_layers": n_extra_layers, "noBN": opt.noBN, "mlp_G": opt.mlp_G}
    with open(os.path.join(opt.experiment, "generator_config.json"), 'w') as gcfg:
        gcfg.write(json.dumps(generator_config)+"\n")

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    if opt.noBN:
        netG = dcgan.DCGAN_G_nobn(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)
    elif opt.mlp_G:
        netG = mlp.MLP_G(opt.imageSize, nz, nc, ngf, ngpu)
    else:
        netG = dcgan.DCGAN_G(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)

    # write out generator config to generate images together wth training checkpoints (.pth)
    generator_config = {"imageSize": opt.imageSize, "nz": nz, "nc": nc, "ngf": ngf, "ngpu": ngpu, "n_extra_layers": n_extra_layers, "noBN": opt.noBN, "mlp_G": opt.mlp_G}
    with open(os.path.join(opt.experiment, "generator_config.json"), 'w') as gcfg:
        gcfg.write(json.dumps(generator_config)+"\n")

    netG.apply(weights_init)
    if opt.netG != '': # load checkpoint if needed
        netG.load_state_dict(torch.load(opt.netG))

    if opt.mlp_D:
        netD = mlp.MLP_D(opt.imageSize, nz, nc, ndf, ngpu)
    else:
        netD = dcgan.DCGAN_D(opt.imageSize, nz, nc, ndf, ngpu, n_extra_layers)
        netD.apply(weights_init)

    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
    noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
    fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
    one = torch.FloatTensor([1])
    mone = one * -1

    if opt.cuda:
        print("using cuda ===================================== ")
        netD.cuda()
        netG.cuda()
        input = input.cuda()
        one, mone = one.cuda(), mone.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
    
    # setup optimizer
    if opt.adam:
        optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
    else:
        optimizerD = optim.RMSprop(netD.parameters(), lr = opt.lrD)
        optimizerG = optim.RMSprop(netG.parameters(), lr = opt.lrG)


    # ==================================================================
    # 新增：定义 WGAN-GP 论文中公式 (1) 要求的梯度惩罚函数
    # ==================================================================
    def calc_gradient_penalty(netD, real_data, fake_data, opt):
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1)
        if opt.cuda:
            alpha = alpha.cuda()
            
        # 混合真假数据
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = Variable(interpolates, requires_grad=True)
        
        disc_interpolates = netD(interpolates)
        
        ones = torch.ones(disc_interpolates.size())
        if opt.cuda:
            ones = ones.cuda()
            
        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=ones,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        # 公式 (1) 的最后一项：lambda * (||grad||_2 - 1)^2，这里 lambda 通常取 10
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
        return gradient_penalty

    # ==================================================================
    # 优化的训练循环开始
    # ==================================================================
    gen_iterations = 0
    for epoch in range(opt.niter):
        data_iter = iter(dataloader)
        i = 0

        # 定义变量用于累加一个 Epoch 内的 Loss，以便计算平均值
        epoch_errD = 0
        epoch_errG = 0
        batch_count = len(dataloader)

        while i < batch_count:
            ############################
            # (1) Update D network
            ###########################
            for p in netD.parameters():
                p.requires_grad = True 

            # 彻底摒弃 Diters=100 的设定，严格遵循正常比例 (通常为 5)
            Diters = opt.Diters
            j = 0
            while j < Diters and i < len(dataloader):
                j += 1
                
                # 【已删除】原有的 p.data.clamp_ (WGAN-GP 不需要，且极耗时)

                data = next(data_iter)
                i += 1

                # 直接获取已经在 transforms 里处理好的 64x64 的 label
                real_cpu = data['label'] 
                # 【已删除】原有的 F.interpolate 耗时操作

                if epoch == 0 and i == 0:
                    print(f"DEBUG - Tensor Max: {real_cpu.max().item():.4f}")
                    print(f"DEBUG - Tensor Min: {real_cpu.min().item():.4f}")
                
                # 确保格式为 float 且维度正确 (Batch, Channel, H, W)
                real_cpu = real_cpu.float()
                batch_size = real_cpu.size(0)

                if opt.cuda:
                    real_cpu = real_cpu.cuda()
                inputv = Variable(real_cpu)

                netD.zero_grad()

                # --- 1. Train with real ---
                errD_real = netD(inputv).mean()

                # --- 2. Train with fake ---
                noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
                noisev = Variable(noise, volatile=True) # 冻结 G 以加速
                fake = Variable(netG(noisev).data)
                errD_fake = netD(fake).mean()

                # --- 3. 计算 Gradient Penalty ---
                gradient_penalty = calc_gradient_penalty(netD, inputv.data, fake.data, opt)

                # --- 4. 汇总 D 的 Loss (WGAN-GP) ---
                # D 试图最大化 D(real) - D(fake)，等价于最小化 errD_fake - errD_real + GP
                errD = errD_fake - errD_real + gradient_penalty
                errD.backward()
                optimizerD.step()

            ############################
            # (2) Update G network
            ###########################
            for p in netD.parameters():
                p.requires_grad = False # 冻结 D
            
            netG.zero_grad()
            noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
            noisev = Variable(noise)
            fake = netG(noisev)
            
            # G 试图最大化 D(fake)，等价于最小化 -D(fake)
            errG = -netD(fake).mean()
            errG.backward()
            optimizerG.step()

            gen_iterations += 1
            i += 1

            # --- 【新增】累加当前 Batch 的 Loss ---
            epoch_errD += errD.item()
            epoch_errG += errG.item()
                
            # --- 原始的保存图片逻辑保持不变 ---
            if gen_iterations % 500 == 0:
                # 注意：因为 inputv 就是真实图片，这里直接用 inputv 的 data
                real_save = inputv.data.mul(0.5).add(0.5) if opt.cuda else inputv.mul(0.5).add(0.5)
                vutils.save_image(real_save, '{0}/real_samples.png'.format(opt.experiment))
                
                fake_save = netG(Variable(fixed_noise, volatile=True))
                # 只有保存用于可视化时，才做 256x256 的放大，不影响训练速度
                fake_save = F.interpolate(fake_save, size=(256, 256), mode='bilinear', align_corners=False)
                fake_save.data = fake_save.data.mul(0.5).add(0.5)
                vutils.save_image(fake_save.data, '{0}/fake_samples_{1}.png'.format(opt.experiment, gen_iterations))

        # <<<<<< 【关键修改】在这里跳出 while 循环 >>>>>>
        
        # 计算整个 Epoch 的平均 Loss
        avg_D = epoch_errD / len(dataloader)
        avg_G = epoch_errG / len(dataloader)

        # 打印这一轮的结果
        print('[%d/%d] End of Epoch | Avg_Loss_D: %.4f | Avg_Loss_G: %.4f | Total_Iter: %d'
              % (epoch, opt.niter, avg_D, avg_G, gen_iterations))

        # --- 每 1000 轮保存模型（在 for 循环内，while 循环外） ---
        if epoch % 1000 == 0:
            torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch))
            torch.save(netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(opt.experiment, epoch))