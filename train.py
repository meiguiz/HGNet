import datetime
import os
import sys
import argparse
import logging


import cv2


import torch
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary

import tensorboardX

from utils.visualisation.gridshow import gridshow
from models.rgbdfusion import Fusion
from utils.dataset_processing import evaluation
from utils.data import get_dataset
from models import get_network
from models.common import post_process_output
from utils.dataset_processing.evaluation import calculate_GIOU_predata

logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description='Train Dense Attention GraspNet')

    # Network
    parser.add_argument('--network', type=str, default='hgnet-grasp', help='Network Name in .models')
    parser.add_argument('--pretrained-model', type=str, default='', help='use pretrained model to train')

    # Dataset & Data & Training
    parser.add_argument('--dataset', type=str, default='cornell', help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', type=str, default='/media/meiguiz/HIKVISION/ssdcdataset/cornell', help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1, help='Use RGB image for training (0/1)')
    parser.add_argument('--split', type=float, default=0.95, help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split for cross validation.')
    parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')

    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=500, help='Batches per Epoch')
    parser.add_argument('--val-batches', type=int, default=50, help='Validation Batches')

    # Logging etc.
    parser.add_argument('--description', type=str, default='', help='Training description')
    parser.add_argument('--outdir', type=str, default='output/models/hggrasp', help='Training Output Directory')
    parser.add_argument('--logdir', type=str, default='tensorboard/hggrasp', help='Log directory')
    parser.add_argument('--vis', action='store_true', help='Visualise the training process')

    args = parser.parse_args()
    return args


def validate(net, device, val_data, batches_per_epoch):
    """
    Run validation.
    :param net: Network
    :param device: Torch device
    :param val_data: Validation Dataset
    :param batches_per_epoch: Number of batches to run
    :return: Successes, Failures and Losses
    """
    net.eval()

    results = {
        'correct': 0,
        'failed': 0,
        'loss': 0,
        'losses': {

        }
    }

    ld = len(val_data)

    with torch.no_grad():
        batch_idx = 0
        while batch_idx < batches_per_epoch:
            for x, y, didx, rot, zoom_factor in val_data:
                batch_idx += 1
                if batches_per_epoch is not None and batch_idx >= batches_per_epoch:
                    break

                xc = x.to(device)
                yc = [yy.to(device) for yy in y]
                lossd = compute_loss(net, xc, yc, didx, rot, zoom_factor, val_data)

                loss = lossd['loss']

                results['loss'] += loss.item()/ld
                for ln, l in lossd['losses'].items():
                    if ln not in results['losses']:
                        results['losses'][ln] = 0
                    results['losses'][ln] += l.item()/ld

                q_out, ang_out, w_out = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                            lossd['pred']['sin'], lossd['pred']['width'])

                s = evaluation.calculate_iou_match(q_out, ang_out,
                                                   val_data.dataset.get_gtbb(didx, rot, zoom_factor),
                                                   no_grasps=1,
                                                   grasp_width=w_out,
                                                   )

                if s:
                    results['correct'] += 1
                else:
                    results['failed'] += 1

    return results

def compute_GIOU_loss(max_iou, union, enclose_area):
    #for balancing loss of several tasks
    GIOU_loss = 0.5 - (max_iou - (enclose_area - union) / enclose_area)
    return GIOU_loss

def compute_loss(net, xc, yc, idx, rot, zoom_factor, train_data):
    y_pos, y_cos, y_sin, y_width = yc
    pos_pred, cos_pred, sin_pred, width_pred = net(xc)

    p_loss = F.mse_loss(pos_pred, y_pos)
    cos_loss = F.mse_loss(cos_pred, y_cos)*2
    sin_loss = F.mse_loss(sin_pred, y_sin)*4
    width_loss = F.mse_loss(width_pred, y_width)

    with torch.no_grad():
        GIOU_loss = 0
        for pos_pred, cos_pred, sin_pred, width_pred, idx, rot, zoom_factor in zip(pos_pred, cos_pred, sin_pred, width_pred, idx, rot, zoom_factor):
            q_out, ang_out, w_out = post_process_output(pos_pred, cos_pred, sin_pred, width_pred)

            max_iou, union, enclose_area = calculate_GIOU_predata(q_out, ang_out,
                                               train_data.dataset.get_gtbb(idx, rot, zoom_factor),
                                               no_grasps=1,
                                               grasp_width=w_out,
                                               )
            GIOU_loss += compute_GIOU_loss(max_iou, union, enclose_area)
    GIOU_loss = GIOU_loss / 100.0
    GIOU_loss = torch.tensor(GIOU_loss, requires_grad=True).cuda()
    return {
        'loss': p_loss + cos_loss + sin_loss + width_loss + GIOU_loss,
        # 'loss': p_loss + cos_loss + sin_loss + width_loss,
        'losses': {
            'p_loss': p_loss,
            'cos_loss': cos_loss,
            'sin_loss': sin_loss,
            'width_loss': width_loss,
            'GIOU_loss': GIOU_loss
        },
        'pred': {
            'pos': pos_pred,
            'cos': cos_pred,
            'sin': sin_pred,
            'width': width_pred

        }
    }

def train(epoch, net, device, train_data, optimizer, batches_per_epoch, vis=False):
    """
    Run one training epoch
    :param epoch: Current epoch
    :param net: Network
    :param device: Torch device
    :param train_data: Training Dataset
    :param optimizer: Optimizer
    :param batches_per_epoch:  Data batches to train on
    :param vis:  Visualise training progress
    :return:  Average Losses for Epoch
    """
    results = {
        'loss': 0,
        'losses': {
        }
    }

    net.train()

    batch_idx = 0
    # Use batches per epoch to make training on different sized datasets (cornell/jacquard) more equivalent.
    while batch_idx < batches_per_epoch:
        for x, y, idx, rot, zoom_factor in train_data:
            batch_idx += 1
            if batch_idx >= batches_per_epoch:
                break
            xc = x.to(device)
            yc = [yy.to(device) for yy in y]
            lossd = compute_loss(net, xc, yc, idx, rot, zoom_factor, train_data)
            loss = lossd['loss']

            p_loss = lossd['losses']['p_loss']
            cos_loss = lossd['losses']['cos_loss']
            sin_loss = lossd['losses']['sin_loss']
            width_loss = lossd['losses']['width_loss']
            GIOU_loss = lossd['losses']['GIOU_loss']
            if batch_idx % 100 == 0:
            # if batch_idx % 10 == 0:

                logging.info('Epoch: {}, Batch: {}, loss: {}, p_loss: {:0.4f},cos_loss: {:0.4f},sin_loss: {:0.4f},width_loss: {:0.4f},'
                             'GIOU_loss: {:0.4f},'.format(epoch, batch_idx, loss.item(), p_loss.item(), cos_loss.item(), sin_loss.item(), width_loss.item(), GIOU_loss.item()))
                # logging.info(
                # 'Epoch: {}, Batch: {}, loss: {}, p_loss: {:0.4f},cos_loss: {:0.4f},sin_loss: {:0.4f},width_loss: {:0.4f},'
                # .format(epoch, batch_idx, loss.item(), p_loss.item(), cos_loss.item(),
                #                              sin_loss.item(), width_loss.item()))
            results['loss'] += loss.item()
            for ln, l in lossd['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Display the images
            if vis:
                imgs = []
                n_img = min(4, x.shape[0])
                for idx in range(n_img):
                    imgs.extend([x[idx,].numpy().squeeze()] + [yi[idx,].numpy().squeeze() for yi in y] + [
                        x[idx,].numpy().squeeze()] + [pc[idx,].detach().cpu().numpy().squeeze() for pc in lossd['pred'].values()])
                gridshow('Display', imgs,
                         [(xc.min().item(), xc.max().item()), (0.0, 1.0), (0.0, 1.0), (-1.0, 1.0), (0.0, 1.0)] * 2 * n_img,
                         [cv2.COLORMAP_BONE] * 10 * n_img, 10)
                cv2.waitKey(2)

    results['loss'] /= batch_idx
    for l in results['losses']:
        results['losses'][l] /= batch_idx

    return results


def run():
    args = parse_args()

    # Vis window
    if args.vis:
        cv2.namedWindow('Display', cv2.WINDOW_NORMAL)

    # Set-up output directories
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    net_desc = '{}_{}'.format(dt, '_'.join(args.description.split()))

    save_folder = os.path.join(args.outdir, net_desc)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    tb = tensorboardX.SummaryWriter(os.path.join(args.logdir, net_desc))

    # Load Dataset
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)

    train_dataset = Dataset(args.dataset_path, start=0.0, end=args.split, ds_rotate=args.ds_rotate,
                            random_rotate=True, random_zoom=True,
                            include_depth=args.use_depth, include_rgb=args.use_rgb)
    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_dataset = Dataset(args.dataset_path, start=args.split, end=1.0, ds_rotate=args.ds_rotate,
                          random_rotate=False, random_zoom=False,
                          include_depth=args.use_depth, include_rgb=args.use_rgb)
    val_data = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )
    logging.info('Done')

    # Load the network
    logging.info('Loading Network...')
    input_channels = 1*args.use_depth + 3*args.use_rgb
    if args.pretrained_model != '':
        print("load model from %s ..." % args.pretrained_model)
        net = torch.load(args.pretrained_model)
    else:
        HGGrasp = get_network(args.network)
        net = HGGrasp(angle_cls=180)

        # net = ggcnn(input_channels=input_channels)
        # network = get_network(args.network)
        # net = network(
        #     input_channels=input_channels,
        #     dropout=False,
        #     prob=0,
        #     channel_size=32
        # )
        net = nn.DataParallel(net)


    device = torch.device("cuda:0")
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(),0.001)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 45, 60, 75], gamma=0.5)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50, 80], gamma=0.1)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6, 12, 18, 24, 30], gamma=0.5)

    logging.info('Done')


    summary(net, (input_channels, 300, 300))
    f = open(os.path.join(save_folder, 'arch.txt'), 'w')
    sys.stdout = f
    summary(net, (input_channels, 300, 300))
    sys.stdout = sys.__stdout__
    f.close()

    best_iou = 0.0
    for epoch in range(args.epochs):
        logging.info('Beginning Epoch {:02d}'.format(epoch))
        train_results = train(epoch, net, device, train_data, optimizer, args.batches_per_epoch, vis=args.vis)
        scheduler.step()

        # Log training losses to tensorboard
        tb.add_scalar('loss/train_loss', train_results['loss'], epoch)
        for n, l in train_results['losses'].items():
            tb.add_scalar('train_loss/' + n, l, epoch)

        # Run Validation
        logging.info('Validating...')
        test_results = validate(net, device, val_data, args.val_batches)
        logging.info('%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                     test_results['correct']/(test_results['correct']+test_results['failed'])))

        # Log validation results to tensorbaord
        tb.add_scalar('loss/IOU', test_results['correct'] / (test_results['correct'] + test_results['failed']), epoch)
        tb.add_scalar('loss/val_loss', test_results['loss'], epoch)
        for n, l in test_results['losses'].items():
            tb.add_scalar('val_loss/' + n, l, epoch)

        # Save best performing network
        iou = test_results['correct'] / (test_results['correct'] + test_results['failed'])
        if  epoch == 0 or (epoch % 1) == 0:
            torch.save(net, os.path.join(save_folder, 'epoch_%02d_iou_%0.4f' % (epoch, iou)))
            torch.save(net.state_dict(), os.path.join(save_folder, 'epoch_%02d_iou_%0.4f_statedict.pt' % (epoch, iou)))
            best_iou = iou


if __name__ == '__main__':
    run()
