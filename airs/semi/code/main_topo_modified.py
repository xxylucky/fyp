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


def DeepSupSeg(pred, gt):
    criterion = BceDiceLoss()
    return criterion(pred, gt)


# ------------------------ helpers ------------------------
def lr_poly(base_lr, itr, max_iter, power):
    return base_lr * ((1 - float(itr) / max_iter) ** power)


def adjust_lr_rate(optimizer, itr, total_batch):
    lr = lr_poly(args.lr, itr, args.nEpoch * total_batch, args.power)
    optimizer.param_groups[0]['lr'] = lr
    return lr


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
        backend_root=getattr(args, 'topo_backend_root', '/hy-tmp/airs/Betti-Matching-3D-master'),
        warn_once=True,
    )


def use_topo_sup_now():
    return bool(getattr(args, 'use_topo_sup', 1))


def use_topo_semi_now(epoch: int):
    if not bool(getattr(args, 'use_topo_semi', 1)):
        return False
    start_epoch = int(getattr(args, 'use_semi', 0))
    return epoch >= start_epoch


# ------------------------ training ------------------------
def train():
    """Full supervised training."""
    train_l_data, _, valid_data = build_dataset(args)
    train_l_dataloader = DataLoader(train_l_data, args.batch_size, shuffle=True, num_workers=args.num_workers)

    valid_sign = valid_data is not None
    if valid_sign:
        valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=args.num_workers)
        val_total_batch = int(len(valid_data) / 1)

    model = build_model(args)
    topo_criterion = build_topo_criterion()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mt, weight_decay=args.weight_decay)

    print('\n---------------------------------')
    print('Start training')
    print('---------------------------------\n')

    F1_best, F1_second_best, F1_third_best = 0, 0, 0
    best = 0

    for epoch in range(args.nEpoch):
        model.train()
        print(f"Epoch: {epoch}")
        total_batch = math.ceil(len(train_l_data) / args.batch_size)
        bar = tqdm(enumerate(train_l_dataloader), total=total_batch)

        for batch_id, data_l in bar:
            itr = total_batch * epoch + batch_id
            img, gt = data_l['image'], data_l['label']
            if args.GPUs:
                img = img.cuda()
                gt = gt.cuda()

            optimizer.zero_grad()
            outputs = model(img)
            mask, _, _, _, _, _, _ = unpack_model_outputs(outputs)

            loss_seg = DeepSupSeg(mask, gt)
            loss_topo = topo_criterion(mask, gt) if use_topo_sup_now() else zero_like_loss(mask.device)
            loss = loss_seg + getattr(args, 'lambda_topo_sup', 0.0) * loss_topo

            loss.backward()
            optimizer.step()
            adjust_lr_rate(optimizer, itr, total_batch)

            bar.set_description(
                f"loss={loss.item():.4f} seg={loss_seg.item():.4f} topo_sup={loss_topo.item():.4f}"
            )

        if valid_sign:
            recall, specificity, precision, F1, F2, ACC_overall, IoU_poly, IoU_bg, IoU_mean, dice = evaluate(
                model, valid_dataloader, val_total_batch
            )

            print("Valid Result:")
            print(
                'recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, ACC_overall: %.4f, '
                'IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f, dice: %.4f'
                % (recall, specificity, precision, F1, F2, ACC_overall, IoU_poly, IoU_bg, IoU_mean, dice)
            )

            if dice > best:
                best = dice
            print("Best Dice:: ", best)

            if F1 > F1_best:
                F1_best = F1
                torch.save(model.state_dict(), args.root + "/semi/checkpoint/" + args.ckpt_name + "/best.pth")
            elif F1 > F1_second_best:
                F1_second_best = F1
                torch.save(model.state_dict(), args.root + "/semi/checkpoint/" + args.ckpt_name + "/second_best.pth")
            elif F1 > F1_third_best:
                F1_third_best = F1
                torch.save(model.state_dict(), args.root + "/semi/checkpoint/" + args.ckpt_name + "/third_best.pth")


def train_semi():
    """Semi-supervised training with staged topo learning.

    Stage logic:
      - supervised topo: always controlled by args.use_topo_sup
      - semi topo: enabled only when epoch >= args.use_semi and args.use_topo_semi == 1

    Important:
      pseudo label = mask_binary.detach() from unlabeled forward pass
    """
    train_l_data, train_u_data, valid_data = build_dataset(args)
    train_l_dataloader = DataLoader(train_l_data, args.batch_size, shuffle=True, num_workers=args.num_workers)
    train_u_dataloader = DataLoader(train_u_data, args.batch_size, shuffle=True, num_workers=args.num_workers)

    valid_sign = valid_data is not None
    if valid_sign:
        valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=args.num_workers)
        val_total_batch = int(len(valid_data) / 1)

    model = build_model(args)
    topo_criterion = build_topo_criterion()

    netD = DCGAN_D(64, 100, 1, 64, 1, 0)
    netD.cuda()
    netD_weight = torch.load("/root/example/fyp/airs/semi/code/pretrain/GAN/netD_epoch_10000.pth")
    netD.load_state_dict(netD_weight)
    netD.eval()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mt, weight_decay=args.weight_decay)

    print('\n---------------------------------')
    print('Start training_semi')
    print('---------------------------------\n')

    F1_best, F1_second_best, F1_third_best = 0, 0, 0
    best = 0

    for epoch in range(args.nEpoch):
        model.train()
        print(f"Epoch: {epoch}")
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

            optimizer.zero_grad()

            # -------- labeled branch --------
            pred_l = model(img_l)
            mask_l, _, _, _, _, _, _ = unpack_model_outputs(pred_l)
            loss_l_seg = DeepSupSeg(mask_l, gt)
            loss_l_topo = topo_criterion(mask_l, gt) if sup_topo_on else zero_like_loss(mask_l.device)
            loss_l = loss_l_seg + getattr(args, 'lambda_topo_sup', 0.0) * loss_l_topo

            # -------- unlabeled branch --------
            pred_u = model(img_u)
            mask_u, predboud, inpimg2, inpimg3, inpimg4, inpimg5, mask_boud = unpack_model_outputs(pred_u)
            pseudo = mask_boud.detach()

            loss_u_seg = DeepSupSeg(predboud, pseudo)
            loss_u_topo = topo_criterion(predboud, pseudo) if semi_topo_on else zero_like_loss(predboud.device)

            shape_u_1 = F.interpolate(predboud, size=(64, 64), mode='bilinear', align_corners=False)
            shape_u_2 = F.interpolate(inpimg2, size=(64, 64), mode='bilinear', align_corners=False)
            shape_u_3 = F.interpolate(inpimg3, size=(64, 64), mode='bilinear', align_corners=False)
            shape_u_4 = F.interpolate(inpimg4, size=(64, 64), mode='bilinear', align_corners=False)
            shape_u_5 = F.interpolate(inpimg5, size=(64, 64), mode='bilinear', align_corners=False)
            loss_u_shape = (netD(shape_u_1) + netD(shape_u_2) + netD(shape_u_3) + netD(shape_u_4) + netD(shape_u_5)) / 5

            loss_u = (
                loss_u_seg
                + 0.1 * loss_u_shape
                + getattr(args, 'lambda_topo_semi', 0.0) * loss_u_topo
            )

            # keep your original overall balance: 2 * labeled + unlabeled
            loss = 2 * loss_l + loss_u
            loss.backward()
            optimizer.step()
            adjust_lr_rate(optimizer, itr, total_batch)

            bar.set_description(
                f"loss={loss.item():.4f} | l_seg={loss_l_seg.item():.4f} l_topo={loss_l_topo.item():.4f} "
                f"u_seg={loss_u_seg.item():.4f} u_topo={loss_u_topo.item():.4f} shape={loss_u_shape.item():.4f} "
                f"semi_topo={'on' if semi_topo_on else 'off'}"
            )

        model.eval()
        if valid_sign:
            recall, specificity, precision, F1, F2, ACC_overall, IoU_poly, IoU_bg, IoU_mean, dice, _, _= evaluate(
                model, valid_dataloader, val_total_batch
            )

            print("Valid Result:")
            print(
                'recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, ACC_overall: %.4f, '
                'IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f, dice: %.4f'
                % (recall, specificity, precision, F1, F2, ACC_overall, IoU_poly, IoU_bg, IoU_mean, dice)
            )

            if dice > best:
                best = dice
            print("Best Dice:: ", best)

            if F1 > F1_best:
                F1_best = F1
                torch.save(model.state_dict(), args.root + "/semi/checkpoint/" + args.ckpt_name + "/best.pth")
            elif F1 > F1_second_best:
                F1_second_best = F1
                torch.save(model.state_dict(), args.root + "/semi/checkpoint/" + args.ckpt_name + "/second_best.pth")
            elif F1 > F1_third_best:
                F1_third_best = F1
                torch.save(model.state_dict(), args.root + "/semi/checkpoint/" + args.ckpt_name + "/third_best.pth")


def test():
    print('loading data......')
    test_data = build_dataset(args)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.num_workers)
    total_batch = int(len(test_data) / 1)
    model = build_model(args)
    model.eval()

    recall, specificity, precision, F1, F2, ACC_overall, IoU_poly, IoU_bg, IoU_mean, dice = evaluate(
        model, test_dataloader, total_batch
    )

    print("Test Result:")
    print(
        'recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, ACC_overall: %.4f, '
        'IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f, dice: %.4f'
        % (recall, specificity, precision, F1, F2, ACC_overall, IoU_poly, IoU_bg, IoU_mean, dice)
    )


if __name__ == '__main__':
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    checkpoint_name = os.path.join(args.root, 'semi/checkpoint/' + args.ckpt_name)
    if not os.path.exists(checkpoint_name):
        os.makedirs(checkpoint_name)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPUs
    if args.manner == 'full':
        print('---{}-Seg Train---'.format(args.dataset))
        train()
    elif args.manner == 'semi':
        print('---{}-seg Semi-Train--'.format(args.dataset))
        train_semi()
    elif args.manner == 'test':
        print('---{}-Seg Test---'.format(args.dataset))
        test()
    print('Done')

    os.system("shutdown")
