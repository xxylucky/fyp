import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
model_sam_path = os.path.join(root_dir, "model_sam")
if model_sam_path not in sys.path:
    sys.path.insert(0, model_sam_path)

import math
import os
import random
import warnings
from itertools import cycle

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.build_dataset import build_dataset
from models.build_model import build_model
from models.dc_gan import DCGAN_D
from opt import args
from utils.evaluate import evaluate
from utils.loss_topo_modified import BceDiceLoss, BettiMatchingLoss

warnings.filterwarnings("ignore", category=UserWarning)
from tensorboardX import SummaryWriter  # 动态图表记录训练过程中的指标变化

warnings.filterwarnings("ignore", category=UserWarning)  # 屏蔽所有的用户警告

def DeepSupSeg(pred, gt):
    # 计算模型预测结果（pred）与真实标签（gt）之间的损失值（Loss）。
    criterion = BceDiceLoss()
    loss = criterion(pred, gt)
    return loss

def lr_poly(base_lr, iter, max_iter, power):
    # 实现 Poly 学习率衰减策略。
    return base_lr * ((1-float(iter)/max_iter)**power)

def adjust_lr_rate(argsimizer, iter, total_batch):
    # 在训练过程中实时修改优化器的学习率。
    lr = lr_poly(args.lr, iter, args.nEpoch*total_batch, args.power)
    argsimizer.param_groups[0]['lr'] = lr
    return lr

# ------------------------ Topo ------------------------
def zero_like_loss(device):
    return torch.tensor(0.0, device=device)


def unpack_model_outputs(outputs):
    """
    Compatible with both:
      1) full-supervised models that may directly return mask
      2) your semi model that returns
         (mask, predboud, inpimg2, inpimg3, inpimg4, inpimg5, mask_binary)
    """
    if isinstance(outputs, (tuple, list)):
        if len(outputs) == 7:
            return outputs
        if len(outputs) == 1:
            return outputs[0], None, None, None, None, None, None
    return outputs, None, None, None, None, None, None


def build_topo_criterion():
    return BettiMatchingLoss(
        homology_dim=getattr(args, 'topo_hdim', 0),
        matched_weight=getattr(args, 'lambda_topo_match', 1.0),
        unmatched_weight=getattr(args, 'lambda_topo_unmatch', 1.0),
        include_target_unmatched=bool(getattr(args, 'topo_include_target_unmatched', 0)),
        backend_import=getattr(args, 'topo_backend_import', 'build.betti_matching'),
        backend_root=getattr(args, 'topo_backend_root', '/root/example/fyp/airs/Betti-Matching-3D-master'),
        warn_once=True,
    )


def use_topo_sup_now():
    return bool(getattr(args, 'use_topo_sup', 1))


def use_topo_semi_now(epoch: int):
    if not bool(getattr(args, 'use_topo_semi', 1)):
        return False
    start_epoch = int(getattr(args, 'use_semi', 0))
    # If you wanna keep semi-topo switch aligned with SAM refinement stage:
    # start_epoch = int(getattr(args, 'samAfter', getattr(args, 'use_semi', 0)))
    return epoch >= start_epoch


def train():
    """load data"""
    train_l_data, _ , valid_data = build_dataset(args)
    # batch_size: 一次处理多少张图；shuffle: 是否打乱顺序
    train_l_dataloader = DataLoader(train_l_data, args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_sign = False
    if valid_data is not None:  # 验证的时候使用验证集，一次处理1张图，不打乱顺序
        valid_sign = True
        valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=args.num_workers)
        val_total_batch = int(len(valid_data) / 1)
    
    """load model"""
    model = build_model(args)
    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mt, weight_decay=args.weight_decay)

    # train
    print('\n---------------------------------')
    print('Start training')
    print('---------------------------------\n')

    F1_best, F1_second_best, F1_third_best = 0, 0, 0
    best = 0
    for epoch in range(args.nEpoch):  # 外层循环：把所有数据看 nEpoch 遍
        model.train()  # 设置为训练模式（激活 Dropout 和 BatchNorm）
      
        print("Epoch: {}".format(epoch))
        total_batch = math.ceil(len(train_l_data) / args.batch_size)
        
        bar = tqdm(enumerate(train_l_dataloader), total=total_batch)

        for batch_id, data_l in bar:
            itr = total_batch * epoch + batch_id
            img, gt = data_l['image'], data_l['label']
            if args.GPUs:
                img = img.cuda()
                gt = gt.cuda()
            optim.zero_grad()           # 1. 清空上一步的梯度（类似计数器归零）
            mask = model(img)           # 2. 前向传播：AI 给出预测结果
            loss = DeepSupSeg(mask, gt) # 3. 计算误差：对比预测和答案
            loss.backward()             # 4. 反向传播：计算每个参数该如何调整
            
            optim.step()                # 执行调整：更新模型权重
            
            # 更新学习率（Poly 策略）
            adjust_lr_rate(optim, itr, total_batch)

        if valid_sign == True:  # 验证模式，每个epoch都验证以找到最优参数
            # 计算 Recall, Precision, Dice 等指标
            recall, specificity, precision, F1, F2, \
            ACC_overall, IoU_poly, IoU_bg, IoU_mean, dice = evaluate(model, valid_dataloader, val_total_batch)

            print("Valid Result:")
            print('recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f, dice: %.4f' \
                % (recall, specificity, precision, F1, F2, ACC_overall, IoU_poly, IoU_bg, IoU_mean, dice))

            if dice > best:
                best = dice
            print("Best Dice:: ", best)

            # 如果当前的 Dice 指标（重合度）是最好的，就保存模型参数。保存前三好的模型参数。
            if (F1 > F1_best):
                F1_best = F1
                torch.save(model.state_dict(), args.root + "/semi/checkpoint/" + args.ckpt_name + "/best.pth")
            elif(F1 > F1_second_best):
                F1_second_best = F1
                torch.save(model.state_dict(), args.root + "/semi/checkpoint/" + args.ckpt_name + "/second_best.pth")
            elif(F1 > F1_third_best):
                F1_third_best = F1
                torch.save(model.state_dict(), args.root + "/semi/checkpoint/" + args.ckpt_name + "/third_best.pth")

def train_semi():
    """load data"""
    # 调用 build_dataset 获取有标签、无标签和验证集
    train_l_data, train_u_data, valid_data = build_dataset(args)
    # batch_size: 一次处理多少张图；shuffle: 是否打乱顺序
    train_l_dataloader = DataLoader(train_l_data, args.batch_size, shuffle=True, num_workers=args.num_workers)
    train_u_dataloader = DataLoader(train_u_data, args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_sign = False
    if valid_data is not None:
        valid_sign = True
        valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=args.num_workers)
        val_total_batch = int(len(valid_data) / 1)
    
    """load model"""
    model = build_model(args)

    #topo
    topo_criterion = build_topo_criterion()

    # DSR
    netD = DCGAN_D(64, 100, 1, 64, 1, 0)
    netD.cuda()
    netD_weight = torch.load("pretrain/GAN/netD_epoch_10000.pth")
    netD.load_state_dict(netD_weight)
    netD.eval()  # 设为评估模式Evaluation，不更新netD的参数。

    # MedSAM
    from model_sam.medsam import load_medsam
    from utils.get_prompts import get_bbox256_torch 
    medsam_ckpt = os.path.join(args.root, 'model_sam/lite_medsam.pth')
    medsam = load_medsam(medsam_ckpt, cuda_num='cuda:0')
    medsam.eval()
    for param in medsam.parameters():
        param.requires_grad = False # 冻结 MedSAM 
    print("MedSAM loaded and frozen.")

    # 优化器 (Optimizer)
    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mt, weight_decay=args.weight_decay)

    # train
    print('\n---------------------------------')
    # print('Start training_semi with MedSAM Refinement and TOPO modified (After Epoch %d)'%(args.samAfter))
    print('Start training_semi')
    print('---------------------------------\n')
    aspp_on = str(getattr(args, 'model', '')).lower().endswith('aspp')
    topo_sup_on = bool(getattr(args, 'use_topo_sup', 1))
    topo_semi_on = bool(getattr(args, 'use_topo_semi', 1))
    topo_semi_start = int(getattr(args, 'use_semi', 0))
    print('[Stage Config] ASPP: {}'.format('ON' if aspp_on else 'OFF'))
    print('[Stage Config] MedSAM: {} (start epoch: {})'.format('ON' if int(getattr(args, 'samAfter', -1)) < args.nEpoch else 'OFF', args.samAfter))
    print('[Stage Config] TOPO(supervised): {}'.format('ON' if topo_sup_on else 'OFF'))
    print('[Stage Config] TOPO(unlabeled): {} (start epoch: {})'.format('ON' if topo_semi_on else 'OFF', topo_semi_start))
    print('')

    F1_best, F1_second_best, F1_third_best = 0, 0, 0
    best = 0
    
    for epoch in range(args.nEpoch):
        model.train()
        print("Epoch: {}".format(epoch))
        loader = iter(zip(cycle(train_l_dataloader), train_u_dataloader))
        bar = tqdm(range(len(train_u_dataloader)))
        
        semi_topo_on = use_topo_semi_now(epoch)
        sup_topo_on = use_topo_sup_now()

        for batch_id in bar:
            data_l, data_u = next(loader)
            total_batch = len(train_u_dataloader)
            itr = total_batch * epoch + batch_id
            
            img_l, gt = data_l['image'], data_l['label']
            img_u = data_u
            if args.GPUs:
                img_l = img_l.cuda()
                gt = gt.cuda()
                img_u = img_u.cuda()

            optim.zero_grad() # 梯度清零

            # 【有监督分支：同时也是教师模型对有标签数据的学习】
            # pred_l = model(img_l)        
            # mask = pred_l[0]             
            # loss_l = DeepSupSeg(mask, gt)  # 是GT和教师模型预测的损失

            # -------- labeled branch --------
            pred_l = model(img_l)
            mask_l, _, _, _, _, _, _ = unpack_model_outputs(pred_l)
            loss_l_seg = DeepSupSeg(mask_l, gt)
            loss_l_topo = topo_criterion(mask_l, gt) if sup_topo_on else zero_like_loss(mask_l.device)
            loss_l = loss_l_seg + getattr(args, 'lambda_topo_sup', 0.0) * loss_l_topo

            # 【无监督/一致性分支：教师-学生机制】
            # smoke test: 在opt.py中修改samAfter为0，使模型一开始就接入SAM
            if epoch >= args.samAfter:
                # 1. 教师模型生成初步伪标签 (不计算梯度)
                with torch.no_grad():
                    model.eval() # 切换为评估模式，关闭 Dropout/BatchNorm 更新
                    pred_u_teacher = model(img_u)
                    mask_teacher = pred_u_teacher[-1] # 取 mask_boud 作为初步预测
                    
                    # 2. 从教师预测中提取 Bbox 作为 Prompt
                    bboxes = get_bbox256_torch((mask_teacher > 0.5).float(), bbox_shift=3)
                    
                    # 3. img_u 转 3 通道喂给 MedSAM（保持 3 通道输入，不要重复已有 3 通道造成 9 通道）
                    img_u_3ch = img_u

                    # 4. MedSAM 生成精细化伪标签
                    sam_mask, _ = medsam(img_u_3ch, bboxes)
                    refined_pseudo_label = torch.sigmoid(sam_mask) > 0.5
                    refined_pseudo_label = refined_pseudo_label.float().detach()
                    if refined_pseudo_label.dim() == 3:
                        refined_pseudo_label = refined_pseudo_label.unsqueeze(1)

                # 5. 学生模型进行前向传播并学习精细化伪标签 (计算梯度)
                model.train() # 切回训练模式
                pred_u_student = model(img_u)
                _, predboud, inpimg2, inpimg3, inpimg4, inpimg5, _ = pred_u_student
                
                # 计算学生预测与 MedSAM 伪标签的损失
                loss_u_seg = DeepSupSeg(predboud, refined_pseudo_label)
                loss_u_topo = topo_criterion(predboud, refined_pseudo_label) if semi_topo_on else zero_like_loss(predboud.device)
                
            else:
                # 前 30 个 Epoch 直接使用内部预测进行一致性学习 =====
                model.train()
                pred_u_student = model(img_u)
                _, predboud, inpimg2, inpimg3, inpimg4, inpimg5, mask_boud = pred_u_student
                
                # 早期使用自身的粗糙 mask 作为伪标签 (使用 detach 截断梯度，模拟标准伪标签行为)
                loss_u_seg = DeepSupSeg(predboud, mask_boud.detach())
                loss_u_topo = topo_criterion(predboud, mask_boud.detach()) if semi_topo_on else zero_like_loss(predboud.device)


            # 【形状先验损失计算】(保持不变)
            shape_u_1 = F.interpolate(predboud, size=(64, 64), mode='bilinear', align_corners=False)
            shape_u_2 = F.interpolate(inpimg2, size=(64, 64), mode='bilinear', align_corners=False)
            shape_u_3 = F.interpolate(inpimg3, size=(64, 64), mode='bilinear', align_corners=False)
            shape_u_4 = F.interpolate(inpimg4, size=(64, 64), mode='bilinear', align_corners=False)
            shape_u_5 = F.interpolate(inpimg5, size=(64, 64), mode='bilinear', align_corners=False)
            loss_u_shape = (netD(shape_u_1) + netD(shape_u_2) + netD(shape_u_3) + netD(shape_u_4) + netD(shape_u_5)) / 5
            
            # 【总损失计算】
            loss_u = loss_u_seg + 0.1 * loss_u_shape + getattr(args, 'lambda_topo_semi', 0.0) * loss_u_topo
            loss = 2 * loss_l + loss_u
            
            loss.backward()
            optim.step()
            adjust_lr_rate(optim, itr, total_batch)
            
        # ============ 验证与保存  ============
        model.eval()
        if valid_sign == True:
            # 计算 Recall, Precision, Dice 等指标.最后两个返回值是 list_name 和 list_point，可以根据需要保存或分析
            recall, specificity, precision, F1, F2, \
            ACC_overall, IoU_poly, IoU_bg, IoU_mean, dice, _, _ = evaluate(model, valid_dataloader, val_total_batch)

            print("Valid Result:")
            print('recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f, dice: %.4f' \
                % (recall, specificity, precision, F1, F2, ACC_overall, IoU_poly, IoU_bg, IoU_mean,dice))
            
            # 将验证指标写入 TensorBoard，方便后续分析
            writer.add_scalar('Val/Dice', dice, epoch)
            writer.add_scalar('Val/IoU_Mean', IoU_mean, epoch)
            writer.add_scalar('Val/Recall', recall, epoch)
            # 如果你有计算训练阶段的 Loss，也可以记录
            writer.add_scalar('Train/Loss_l_Seg', loss_l_seg, epoch)
            writer.add_scalar('Train/Loss_l_Topo', loss_l_topo, epoch)
            writer.add_scalar('Train/Loss_u_Seg', loss_u_seg, epoch)
            writer.add_scalar('Train/Loss_u_Shape', loss_u_shape, epoch)
            writer.add_scalar('Train/Loss_u_Topo', loss_u_topo, epoch)
            writer.add_scalar('Train/Loss', loss, epoch)

            if dice > best:
                best = dice
            print("Best Dice:: ", best)

            if (F1 > F1_best):
                F1_best = F1
                torch.save(model.state_dict(), args.root + "/semi/checkpoint/" + args.ckpt_name + "/best.pth")
            elif(F1 > F1_second_best):
                F1_second_best = F1
                torch.save(model.state_dict(), args.root + "/semi/checkpoint/" + args.ckpt_name + "/second_best.pth")
            elif(F1 > F1_third_best):
                F1_third_best = F1
                torch.save(model.state_dict(), args.root + "/semi/checkpoint/" + args.ckpt_name + "/third_best.pth")

def test():
  
    print('loading data......')
    test_data = build_dataset(args)  # 加载测试集：只有测试图片
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.num_workers)  # 批量大小为1，逐张检查。不打乱顺序。
    total_batch = int(len(test_data) / 1)
    model = build_model(args)
    model.eval()  # 评估模式，关闭dropout。

    # 调用 evaluate 函数计算最终指标
    recall, specificity, precision, F1, F2, \
            ACC_overall, IoU_poly, IoU_bg, IoU_mean, dice = evaluate(model, test_dataloader, total_batch)
    
    print("Test Result:")
    print('recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f, dice: %.4f' \
                % (recall, specificity, precision, F1, F2, ACC_overall, IoU_poly, IoU_bg, IoU_mean,dice))

if __name__ == '__main__':
    import random
    import numpy as np
    import torch
    # 固定随机种子，确保每次运行结果一致（可复现性）
    seed = 19  # 宇宙终极答案，或者填你喜欢的数字
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 为了保证卷积计算的完全一致性，牺牲一丁点速度
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 拼接保存权重的文件夹路径
    checkpoint_name = os.path.join(args.root, 'semi/checkpoint/' + args.ckpt_name)
    if not os.path.exists(checkpoint_name):  # 如果文件夹不存在，就自动创建一个
        os.makedirs(checkpoint_name)
    else:
        pass
    
    # 创建一个 SummaryWriter 实例，用于记录训练过程中的指标变化，方便后续使用 TensorBoard 可视化分析
    writer = SummaryWriter(log_dir=os.path.join(checkpoint_name, 'logs'))

    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPUs
    from datetime import datetime
    print(datetime.now())
    if args.manner == 'full':
        print('---{}-Seg Train---'.format(args.dataset))
        train()
    elif args.manner =='semi':
        print('---{}-seg Semi-Train--'.format(args.dataset))
        train_semi()
    elif args.manner == 'test':
        print('---{}-Seg Test---'.format(args.dataset))
        test()
    print('Done')


