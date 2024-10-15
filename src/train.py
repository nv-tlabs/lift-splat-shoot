"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""
from typing import Optional, Any

import torch
from time import time

from matplotlib import pyplot as plt
from pytorch3d.loss import chamfer, chamfer_distance
from pytorch3d.structures import Pointclouds
from tensorboardX import SummaryWriter
import numpy as np
import os

from .models import compile_model
from .data import compile_data
from .tools import SimpleLoss, get_batch_iou, get_val_info


def visualize_gt_pred_pc(gt_pc, pred_pc, filepath: Optional[str] = None):
    gt_pc_vis = gt_pc.view(-1, 3).detach().cpu().numpy()
    pred_pc_vis = pred_pc.view(-1, 3).detach().cpu().numpy()

    # Assuming that gt_pc_vis and pred_pc_vis are 2D arrays with shape (n_points, 2)
    xs_gt, ys_gt = gt_pc_vis[:, 0], gt_pc_vis[:, 1]
    xs_pred, ys_pred = pred_pc_vis[:, 0], pred_pc_vis[:, 1]

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)  # Adding 3D projection

    # Plotting the first set of points with the first color map
    img_pred = ax.scatter(xs_pred, ys_pred, c=pred_pc_vis[:, 2], cmap='Reds')

    import ipdb; ipdb.set_trace()

    img_gt = ax.scatter(xs_gt, ys_gt, c=gt_pc_vis[:, 2], cmap='Blues')

    # Plotting the second set of points with the second color map

    # Creating color bars for each scatter plot
    fig.colorbar(img_gt, ax=ax, shrink=0.5, aspect=5, label='Ground Truth')
    fig.colorbar(img_pred, ax=ax, shrink=0.5, aspect=5, label='Prediction')

    # Setting the labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    fig.savefig(filepath)


def point_cloud_loss(gt_pc: Pointclouds, pred_pc: Pointclouds, mode: str = 'bidirectional'):

    # Get the length of individual point clouds within the batch.
    gt_lens = [len(p) for p in gt_pc.points_list()]
    pred_lens = [len(p) for p in pred_pc.points_list()]

    assert mode in ['bidirectional', 'gt_first', 'pred_first']
    if mode == 'bidirectional':
        chamdist, _ = chamfer_distance(gt_pc, pred_pc, single_directional=False, x_lengths=gt_lens, y_lengths=pred_lens)
    elif mode == 'gt_first':
        chamdist, _ = chamfer_distance(gt_pc, pred_pc, single_directional=True, x_lengths=gt_lens, y_lengths=pred_lens)
    elif mode == 'pred_first':
        chamdist, _ = chamfer_distance(pred_pc, gt_pc, single_directional=True, x_lengths=pred_lens, y_lengths=gt_lens)

    return chamdist


def train(version,
          dataroot='~/lss/data/',
          nepochs=10000,
          gpuid=1,
          H=900, W=1600,
          resize_lim=(0.193, 0.225),
          final_dim=(128, 352),
          bot_pct_lim=(0.0, 0.22),
          rot_lim=(-5.4, 5.4),
          rand_flip=True,
          ncams=5,
          max_grad_norm=5.0,
          pos_weight=2.13,
          logdir='./runs',
          xbound=[-50.0, 50.0, 0.5],
          ybound=[-50.0, 50.0, 0.5],
          zbound=[-10.0, 10.0, 20.0],
          dbound=[4.0, 45.0, 1.0],
          bsz=4,
          nworkers=10,
          lr=1e-3,
          weight_decay=1e-7,
          pc_loss_weight=5e-2,
          vis_dir='./visualize',
          experiment_name='baseline'
          ):
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {
        'resize_lim': resize_lim,
        'final_dim': final_dim,
        'rot_lim': rot_lim,
        'H': H, 'W': W,
        'rand_flip': rand_flip,
        'bot_pct_lim': bot_pct_lim,
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'Ncams': ncams,
    }
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='segmentationdata')

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    model = compile_model(grid_conf, data_aug_conf, outC=1)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = SimpleLoss(pos_weight).cuda(gpuid)

    writer = SummaryWriter(logdir=f'{logdir}/{experiment_name}')
    val_step = 1000 if version == 'mini' else 10000

    if not os.path.exists(f'{vis_dir}/{experiment_name}'):
        os.makedirs(f'{vis_dir}/{experiment_name}')

    model.train()
    counter = 0
    for epoch in range(nepochs):
        np.random.seed()
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs, lidar_pc) in enumerate(trainloader):
            t0 = time()
            opt.zero_grad()
            pred_pc, preds = model(
                imgs.to(device),
                rots.to(device),
                trans.to(device),
                intrins.to(device),
                post_rots.to(device),
                post_trans.to(device)
            )
            binimgs = binimgs.to(device)
            loss = loss_fn(preds, binimgs)

            # Calculate the loss for the GT and the Pred PC.
            # lidar_pc = lidar_pc.permute(0, 2, 1).to(device)
            lidar_pc = lidar_pc.to(device)
            pred_pc = Pointclouds(pred_pc)
            pc_loss = point_cloud_loss(gt_pc=lidar_pc, pred_pc=pred_pc, mode='bidirectional')

            # Visualize the GT and Pred point cloud from the birds eye view with different color maps.
            if counter % 100 == 4:
                lidar_pc_vis = lidar_pc.points_list()[0]
                pred_pc_vis = pred_pc.points_list()[0]
                visualize_gt_pred_pc(gt_pc=lidar_pc_vis, pred_pc=pred_pc_vis,
                                     filepath=f'{vis_dir}/{experiment_name}/gt_pred_pc_{counter}')

            total_loss = loss + pc_loss * pc_loss_weight
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            t1 = time()

            if counter % 10 == 0:
                print(counter, loss.item())
                print(f"Epoch: {epoch}, Iter: {batchi}, Total Loss: {total_loss.item()}, Seg Loss: {loss.item()}, PC Loss {pc_loss.item()}")
                writer.add_scalar('train/total_loss', total_loss, counter)
                writer.add_scalar('train/loss_seg', loss, counter)
                writer.add_scalar('train/loss_pc', pc_loss, counter)

            if counter % 50 == 0:
                _, _, iou = get_batch_iou(preds, binimgs)
                writer.add_scalar('train/iou', iou, counter)
                writer.add_scalar('train/epoch', epoch, counter)
                writer.add_scalar('train/step_time', t1 - t0, counter)

            if counter % val_step == 0:
                val_info = get_val_info(model, valloader, loss_fn, device)
                print('VAL', val_info)
                writer.add_scalar('val/loss', val_info['loss'], counter)
                writer.add_scalar('val/iou', val_info['iou'], counter)

            if counter % val_step == 0:
                model.eval()
                mname = os.path.join(logdir, "model{}.pt".format(counter))
                print('saving', mname)
                torch.save(model.state_dict(), mname)
                model.train()

            counter += 1
