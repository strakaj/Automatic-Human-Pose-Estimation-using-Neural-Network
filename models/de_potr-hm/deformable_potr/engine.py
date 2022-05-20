# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""

import statistics
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Iterable
import torch
import torchvision.transforms as transforms
import deformable_potr.util.misc as utils
from coco_dataset.coco_dataset import get_max_preds

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int):
    model.train()
    criterion.train()
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    losses_all = []
    print("epoch: ", epoch)
    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for item_index, (samples, joints, vis, target, target_weight) in enumerate(data_loader):
        samples = [item.to(device, dtype=torch.float32) for item in samples]
        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        outputs = model(samples)

        loss_dict = criterion(outputs, target, target_weight)
        losses_all.append(float(loss_dict["loss_coords"].item()))

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if (item_index + 1) % print_freq == 0:
            print(header, "[{0}/{1}]".format(item_index + 1, len(data_loader)),
                  "lr: " + str(optimizer.param_groups[0]["lr"]), "loss: " + str(losses_all[-1]))

    # gather the stats from all processes
    print(header, "Averaged stats:", "lr: " + str(optimizer.param_groups[0]["lr"]),
          "loss: " + str(statistics.mean(losses_all)))

    return {"lr": optimizer.param_groups[0]["lr"], "loss": statistics.mean(losses_all)}


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, print_freq=10):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for samples, joints, vis, target, target_weight in metric_logger.log_every(data_loader, print_freq, header):
        samples = [item.to(device, dtype=torch.float32) for item in samples]
        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        outputs = model(samples)

        loss_dict = criterion(outputs, target, target_weight)

        metric_logger.update(loss=loss_dict["loss_coords"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def save_log_imgs(model, criterion, data_loader, device, path):
    colors = ['red', 'orangered', 'sienna', 'orange', 'gold', 'yellow', 'limegreen', 'green', 'turquoise', 'teal',
              'steelblue', 'navy', 'slateblue', 'indigo', 'violet', 'purple', 'hotpink']
    radius = 2

    normalize_invers = transforms.Normalize(mean=[-0.485 * 2, -0.456 * 2, -0.406 * 2],
                                            std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

    model.eval()
    criterion.eval()

    max_l = 5
    data_l = len(data_loader.dataset)
    if data_l < max_l:
        ids = list(range(0, data_l))
    else:
        ids = list(range(0, max_l))
    out_img = []
    out_trg = []
    out_prd = []

    for i, (samples, joints, vis, target, target_weight) in enumerate(data_loader):
        if not (i in ids):
            continue
        samples = [item.to(device, dtype=torch.float32) for item in samples]
        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        outputs = model(samples)

        # targets = targets.detach().cpu().numpy()
        outputs = outputs.detach().cpu().numpy()
        outputs, maxv = get_max_preds(outputs)

        for s in samples:
            out_img.append(s)
        for t in joints:
            if data_loader.dataset.relative_positions:
                out_trg.append(t.cpu().numpy() * [data_loader.dataset.image_width, data_loader.dataset.image_height])
            else:
                out_trg.append(t)
        for p in outputs:
            if data_loader.dataset.relative_positions:
                out_prd.append(p * [data_loader.dataset.image_width, data_loader.dataset.image_height])
            else:
                out_prd.append(p)

        if len(out_img) >= len(ids):
            break

    rows = len(ids)
    fig, axs = plt.subplots(rows, 2, figsize=(16, rows * 10))

    for i, s in enumerate(out_img[0:len(ids)]):
        ax = axs[i]
        t = out_trg[i]
        p = out_prd[i]

        if (torch.min(s) < 0):
            s = normalize_invers(s)

        img = np.array(s.tolist())
        img = np.moveaxis(img, 0, 2)

        ax[0].imshow(img)
        for j, joint in enumerate(t):
            x, y = joint[0:2]
            c = plt.Circle((x, y), radius=radius, color=colors[j])
            ax[0].add_patch(c)
            ax[0].set_title('Target')

        ax[1].imshow(img)
        for j, joint in enumerate(p):
            x, y = joint[0:2]
            c = plt.Circle((x, y), radius=radius, color=colors[j])
            ax[1].add_patch(c)
            ax[1].set_title('Prediction')

    plt.savefig(path)


@torch.no_grad()
def save_training_summary(model, data_loader, device, log_path, input_log_path, path):
    colors = ['red', 'orangered', 'sienna', 'orange', 'gold', 'yellow', 'limegreen', 'green', 'turquoise', 'teal',
              'steelblue', 'navy', 'slateblue', 'indigo', 'violet', 'purple', 'hotpink']
    radius = 2

    normalize_invers = transforms.Normalize(mean=[-0.485 * 2, -0.456 * 2, -0.406 * 2],
                                            std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

    max_l = 5
    data_l = len(data_loader.dataset)
    if data_l < max_l:
        ids = list(range(0, data_l))
    else:
        ids = list(range(0, max_l))

    loss_data, input_data = load_data(log_path, input_log_path)
    ep = []
    ls = []
    lst = []
    for d in loss_data:
        ls.append(d['train_loss'] / input_data['batch_size'])
        lst.append(d['test_loss'] / input_data['batch_size'])
        ep.append(d['epoch'])

    rows = len(ids) + 1
    ratios = [2] * (rows - 1)
    ratios = [1, *ratios]
    fig = plt.figure(figsize=(16, 8 * rows), constrained_layout=True)
    spec = fig.add_gridspec(rows, 2, height_ratios=ratios)

    ax00 = fig.add_subplot(spec[0, 0])
    ax00.plot(ep, ls, label='train loss')
    ax00.set_xlabel('Epoch')
    ax00.set_ylabel('Loss')
    ax00.grid()

    if isinstance(input_data['augmentations'], list):
        aug = {' '.join([str(elem) + ',' for elem in input_data['augmentations']]).strip(',')}
    else:
        aug = input_data['augmentations']

    ax01 = fig.add_subplot(spec[0, 1])
    annotate_axes(ax01, f"learning rate: {input_data['lr']}\n"
                        f"backbone learning rate: {input_data['lr_backbone']}\n"
                        f"epoch: {input_data['epochs']}\n"
                        f"leraning rate drop: {input_data['lr_drop']}\n"
                        f"batch size: {input_data['batch_size']}\n"
                        f"train images: {input_data['n_train']}             eval images: {input_data['n_eval']}\n"
                        f"sgd: {input_data['sgd']}\n"
                        f"scheduler: {' '.join([str(elem) + ',' for elem in input_data['scheduler']]).strip(',')}\n"
                        f"relative positions: {input_data['relative_positions']}\n"
                        f"best train loss: {np.round(input_data['best_train_loss'] / input_data['batch_size'], 3)} (end: {np.round(input_data['end_train_loss'] / input_data['batch_size'], 3)})\n"
                        f"best validation loss: {np.round(input_data['best_val_loss'], 3)} (end: {np.round(input_data['end_val_loss'], 3)})\n"
                        f"transformations: {' '.join([str(elem) + ',' for elem in input_data['transformations']]).strip(',')}\n"
                        f"augmentations: {aug}\n"
                        f"encoder layers: {input_data['enc_layers']}            decoder layers: {input_data['dec_layers']}\n"
                        f"heads: {input_data['nheads']}\n"
                        f"time: {input_data['time']}"
                  )
    ax01.set_xticks([])
    ax01.set_yticks([])

    model.eval()
    for idx, (samples, joints, vis, target, target_weight) in enumerate(data_loader):
        if not (idx in ids):
            continue
        samples = [item.to(device, dtype=torch.float32) for item in samples]
        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        outputs = model(samples)

        if (torch.min(samples[0]) < 0):
            samples[0] = normalize_invers(samples[0])

        # targets = targets.cpu().numpy()
        outputs = outputs.detach().cpu().numpy()
        outputs, maxv = get_max_preds(outputs)

        img = np.array(samples[0].tolist())
        img = np.moveaxis(img, 0, 2)

        ax10 = fig.add_subplot(spec[idx + 1, 0])
        ax11 = fig.add_subplot(spec[idx + 1, 1])

        for j, joint in enumerate(joints[0]):
            if data_loader.dataset.relative_positions:
                x, y = joint[0:2].cpu().numpy() * [data_loader.dataset.image_width, data_loader.dataset.image_height]
            else:
                x, y = joint[0:2]
            c = plt.Circle((x, y), radius=radius, color=colors[j])
            ax10.imshow(img)
            ax10.add_patch(c)
            ax10.set_xticks([])
            ax10.set_yticks([])
            ax11.set_title('Target')

        for j, joint in enumerate(outputs[0]):
            if data_loader.dataset.relative_positions:
                x, y = joint[0:2] * [data_loader.dataset.image_width, data_loader.dataset.image_height]
            else:
                x, y = joint[0:2]
            c = plt.Circle((x, y), radius=radius, color=colors[j])
            ax11.imshow(img)
            ax11.add_patch(c)
            ax11.set_xticks([])
            ax11.set_yticks([])
            ax11.set_title('Prediction')

    plt.savefig(path)

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.plot(ep, ls, label='train loss')
    ax.plot(ep, lst, label='test loss')
    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid()
    loss_img_path = os.path.dirname(path)
    loss_img_path = os.path.join(loss_img_path, 'loss.png')
    plt.savefig(loss_img_path)


def load_data(log_path, input_log_path):
    loss_data = []
    with open(log_path, 'r') as f:
        for l in f:
            loss_data.append(json.loads(l))

    with open(input_log_path, 'r') as f:
        t = f.read()
        t = t.replace('\'', '\"')
        t = t.replace('None', '\"None\"')
        t = t.replace('True', 'true')
        t = t.replace('False', 'false')

        input_data = json.loads(t)
    return loss_data, input_data


def annotate_axes(ax, text, fontsize=15):
    offset = 0.02
    ax.text(offset, 1 - offset, text, transform=ax.transAxes,
            ha="left", va="top", fontsize=fontsize, color="black")
