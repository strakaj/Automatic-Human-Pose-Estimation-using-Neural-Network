# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import argparse
from ast import arg
import datetime
import json
import random
import time
from pathlib import Path
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import deformable_potr.util.misc as utils
from deformable_potr.engine import evaluate, train_one_epoch, save_log_imgs, save_training_summary
from deformable_potr.models import build_model

from coco_dataset.coco_dataset import COCODataset
import albumentations as A


def get_args_parser():
    parser = argparse.ArgumentParser('Deformable POTR Module', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--save_epoch', default=5, type=int, help='interval of saving the model (in epochs)')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--sgd', action='store_true')
    parser.add_argument('--scheduler', default=[], type=str, nargs='*',
                        help='Name of scheduler followed by values of parameters.')

    # Parametrs: Deformable DETR Variants
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    parser.add_argument('--num_deconvs', default=2, type=int)
    parser.add_argument('--heatmap_size_x', default=48, type=int)
    parser.add_argument('--heatmap_size_y', default=64, type=int)
    parser.add_argument('--sigma', default=2, type=int)

    # Parameters: Model
    #   -> Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    #   -> Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=17, type=int,
                        help="Number of query slots (number of landmarks to be regressed)")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # Parameters: Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    #   -> Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    #   -> Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # Parameters: Dataset
    parser.add_argument('--train_data_path',
                        default="/auto/plzen1/home/strakajk/Datasets/COCO/annotations/person_keypoints_train2017.json",
                        type=str, help="Path to the training dataset coco annotations.")
    parser.add_argument('--eval_data_path',
                        default="/auto/plzen1/home/strakajk/Datasets/COCO/annotations/person_keypoints_val2017.json",
                        type=str, help="Path to the training dataset coco annotations.")
    parser.add_argument('--images_path_train', default="/auto/plzen1/home/strakajk/Datasets/COCO/train2017",
                        type=str, help="Path to the image folder.")
    parser.add_argument('--images_path_eval', default="/auto/plzen1/home/strakajk/Datasets/COCO/val2017",
                        type=str, help="Path to the image folder.")
    parser.add_argument('--transformations', default=['totensor', 'normalize'], type=str, nargs='*')
    parser.add_argument('--augmentations', default=[], type=str, nargs='*')
    parser.add_argument('--crop_w', default=192, type=int)
    parser.add_argument('--crop_h', default=256, type=int)
    parser.add_argument('--relative_positions', action='store_true', help='')

    parser.add_argument('--output_dir', default='', help="Path for saving of the resulting weights and overall model")
    parser.add_argument('--device', default='cuda', help="Device to be used for training and testing")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--init_weights', default='', help='Init network with custom weights (path to weights)')
    parser.add_argument('--resume', default='', help='Resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true', help="Determines whether to perform evaluation on each epoch.")
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    #parser.add_argument('--p_augment', default=0.5, type=float, help="Probability of applying augmentation.")
    parser.add_argument('--encoded', default=0, type=int,
                        help="Whether to read the encoded data (=1) or the decoded data (=0) into memory.")
    #parser.add_argument('--sequence_length', default=0, type=int, help='Number of video frames to process (0 or 3)')
    return parser


def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


def create_scheduler(scheduler, optimizer, epochs, lr):
    sch_name = scheduler[0]
    if sch_name == 'CosineAnnealingLR':
        ep = epochs - 1
        end_lr = lr / 100
        if len(scheduler) >= 2:
            ep = int(scheduler[1])
        if len(scheduler) == 3:
            end_lr = float(scheduler[2])
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, ep, end_lr)

    elif sch_name == 'ExponentialLR':
        gamma = 0.96
        if len(scheduler) == 2:
            end_lr = float(scheduler[1])
        if end_lr < 0.1:
            gamma = (end_lr / lr) ** (1 / (epochs - 1))
        else:
            gamma = end_lr
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

    elif sch_name == 'MultiStepLR':
        milestones = [int(m) for m in scheduler[1].split(",")]
        end_lr = float(scheduler[2])
        if end_lr < 0.1:
            gamma = (end_lr / lr) ** (1 / (len(milestones)))
        else:
            gamma = end_lr
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma)

    elif sch_name == 'StepLR':
        step = 10
        gamma = 0.1
        if len(scheduler) >= 2:
            step = int(scheduler[1])
        if len(scheduler) == 3:
            end_lr = float(scheduler[2])
        if end_lr < 0.1:
            gamma = (end_lr / lr) ** (1 / (int((epochs - 1) / step)))
        else:
            gamma = end_lr
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step, gamma)

    elif sch_name == 'OneCycleLR':
        warm_up_ep = int(scheduler[1])
        end_lr = float(scheduler[2])
        pct_start = warm_up_ep / epochs
        an_st = scheduler[3]
        if an_st == 'lin':
            an_st = 'linear'
        ep = epochs
        if len(scheduler) >= 5:
            ep = int(scheduler[4])
            pct_start = warm_up_ep / ep

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=ep, max_lr=lr, div_factor=10,
                                                           final_div_factor=(lr / end_lr) / 10, pct_start=pct_start,
                                                           anneal_strategy=an_st)

    elif sch_name == 'CyclicLR':
        div = float(scheduler[1])
        ssu = int(scheduler[2])
        mode = scheduler[3]
        gamma = 1
        if mode == 'exp_range':
            gamma = float(scheduler[4])
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr / div, max_lr=lr, step_size_up=ssu,
                                                         mode=mode, gamma=gamma, cycle_momentum=False)

    elif sch_name == 'CosineAnnealingLR':
        tm = float(scheduler[1])
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tm)

    else:
        return None
    print(lr_scheduler)
    return lr_scheduler


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion = build_model(args)
    model.to(device)

    #print("MODEL: ")
    #print(model)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters:", n_parameters)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    trans = []
    if 'totensor' in args.transformations:
        trans.append(transforms.ToTensor())
    elif 'normalize' in args.transformations:
        trans.append(normalize)

    augmentations = None
    if args.augmentations:
        if args.augmentations:
            aug_list = []
            possible_augs = {
                'flip': A.HorizontalFlip(p=0.5),
                'vflip': A.VerticalFlip(p=0.2),
                'flip_1': A.HorizontalFlip(p=1),
                'rbc': A.RandomBrightnessContrast(p=0.6, brightness_limit=0.15, contrast_limit=0.15),
                'shift': A.RGBShift(p=0.3, r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
                'rot': A.Rotate(p=0.4, limit=5, border_mode=1),
                'blur': A.GaussianBlur(p=0.5, blur_limit=(3, 5)),
                'rbc_2': A.RandomBrightnessContrast(p=0.7, brightness_limit=0.2, contrast_limit=0.2),
                'shift_2': A.RGBShift(p=0.5, r_shift_limit=15, g_shift_limit=15, b_shift_limit=15),
                'rot_2': A.Rotate(p=0.5, limit=10, border_mode=1),
                'blur_2': A.OneOf([
                    A.GaussianBlur(p=0.6, blur_limit=(3, 5)),
                    A.MotionBlur((3, 5), p=0.5)
                ], p=0.5),
                'down': A.Downscale(0.8, 0.9, interpolation=0, p=0.4)
            }

            for k in possible_augs:
                if k in args.augmentations:
                    aug_list.append(possible_augs[k])

            augmentations = A.ReplayCompose(
                aug_list,
                keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
                bbox_params=A.BboxParams(format='coco', label_fields=['class_labels'])
            )
    print("Augmentations: ")
    print(augmentations)

    dataset_train = COCODataset(args.train_data_path,
                                args.images_path_train,
                                transforms.Compose(trans),
                                relative_positions=args.relative_positions,
                                augmentations=augmentations,
                                return_vis=True,
                                crop_w=args.crop_w,
                                crop_h=args.crop_h,
                                heatmap_size=[args.heatmap_size_x, args.heatmap_size_y], sigma=args.sigma)

    dataset_eval = COCODataset(args.eval_data_path,
                               args.images_path_eval,
                               transforms.Compose(trans),
                               relative_positions=args.relative_positions,
                               return_vis=True,
                               crop_w=args.crop_w,
                               crop_h=args.crop_h,
                               heatmap_size=[args.heatmap_size_x, args.heatmap_size_y], sigma=args.sigma)

    print('Train samples: ', len(dataset_train))
    print('Evaluation samples: ', len(dataset_eval))

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    if args.eval:
        sampler_eval = torch.utils.data.SequentialSampler(dataset_eval)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, num_workers=args.num_workers,
                                   shuffle=False,
                                   pin_memory=True)
    if args.eval:
        print('Eval true')
        data_loader_eval = DataLoader(dataset_eval, 1, sampler=sampler_eval, drop_last=False, shuffle=False,
                                      num_workers=args.num_workers, pin_memory=True)
    print('Data loaders created')
    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]

    for n, p in model_without_ddp.named_parameters():
        print(n)

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n,
                                                                                                   args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if
                       match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if
                       match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
        print('Optimizer SGD')
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
        print('Optimizer AdamW')

    if args.scheduler and len(args.scheduler) > 1:
        lr_scheduler = create_scheduler(args.scheduler, optimizer, args.epochs, args.lr)
        if lr_scheduler is None:
            print("Wrong input data for scheduler")
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.resume:
        lr_scheduler.step(args.start_epoch)
        print("SCHEDULER RESET: ")
        print(lr_scheduler)

    output_dir = Path(args.output_dir)

    if args.init_weights:
        checkpoint = torch.load(args.init_weights, map_location=device)
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location=device)
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print(
                    'Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1

        if args.eval:
            test_stats = evaluate(model, criterion, data_loader_eval, device)

    best_train_loss = None
    best_val_loss = None

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch)
        lr_scheduler.step()

        if args.eval:
            test_stats = evaluate(model, criterion, data_loader_eval, device)

        if args.output_dir:
            checkpoint_paths = [os.path.join(output_dir, 'checkpoint.pth')]
            # extra checkpoint before LR drop and every N epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_epoch == 0:
                checkpoint_paths.append(os.path.join(output_dir, f'checkpoint{epoch:04}.pth'))
                if args.eval:
                    log_imgs_path = os.path.join(output_dir, f'log_imgs_{epoch:04}.png')
                    save_log_imgs(model, criterion, data_loader_eval, device, log_imgs_path)
            if best_train_loss is None or train_stats["loss"] < best_train_loss:
                checkpoint_paths.append(os.path.join(output_dir, 'checkpoint_best_train_loss.pth'))
                best_train_loss = train_stats["loss"]
            if args.eval:
                if best_val_loss is None or test_stats["loss"] < best_val_loss:
                    checkpoint_paths.append(os.path.join(output_dir, 'checkpoint_best_val_loss.pth'))
                    best_val_loss = test_stats["loss"]

            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.eval:
            log_stats.update({f'test_{k}': v for k, v in test_stats.items()})

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    args_dic = vars(args)
    args_dic['time'] = total_time_str
    args_dic['n_train'] = len(dataset_train)
    args_dic['best_train_loss'] = best_train_loss
    args_dic['end_train_loss'] = train_stats["loss"]

    if args.eval:
        args_dic['n_eval'] = len(dataset_eval)
        args_dic['best_val_loss'] = best_val_loss
        args_dic['end_val_loss'] = test_stats["loss"]

    input_data_file_name = f"input_log.txt"
    summary_image_file_name = f'summary.png'
    if args.start_epoch > 0:
        input_data_file_name = f"input_log_{args.start_epoch}-{args.epochs}.txt"
        summary_image_file_name = f'summary_{args.start_epoch}-{args.epochs}.png'
    with (output_dir / input_data_file_name).open("a") as f:
        f.write(json.dumps(args_dic) + "\n")

    if args.eval:
        sum_path = os.path.join(output_dir, summary_image_file_name)
        log_path = output_dir / "log.txt"
        input_log_path = output_dir / input_data_file_name
        save_training_summary(model, data_loader_eval, device, log_path, input_log_path, sum_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args_in = parser.parse_args()
    if args_in.output_dir:
        Path(args_in.output_dir).mkdir(parents=True, exist_ok=True)
    main(args_in)
