#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao, Pengliang Ji
@Contact: ta19@mails.tsinghua.edu.cn, jpl1723@buaa.edu.cn
@File: main_semseg.py
@Time: 2021/7/20 7:49 PM
"""

from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data import S3DIS, ScanNet
from model import DGCNN_semseg
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
from plyfile import PlyData, PlyElement
from torch.utils.tensorboard import SummaryWriter
from csv import writer

global scene_id
scene_id = 0
global val_scenes
val_scenes = []
global room_seg
room_seg = []
global room_pred
room_pred = []
global visual_warning
visual_warning = True

def _init_():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/'+args.exp_name):
        os.makedirs('outputs/'+args.exp_name)
    if not os.path.exists('outputs/'+args.exp_name+'/'+'models'):
        os.makedirs('outputs/'+args.exp_name+'/'+'models')
    os.system('cp main_semseg.py outputs'+'/'+args.exp_name+'/'+'main_semseg.py.backup')
    os.system('cp model.py outputs' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py outputs' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py outputs' + '/' + args.exp_name + '/' + 'data.py.backup')


def calculate_sem_IoU(pred_np, seg_np, visual=False, num_sem_labels=13):
    I_all = np.zeros(num_sem_labels)
    U_all = np.zeros(num_sem_labels)
    for sem_idx in range(seg_np.shape[0]):
        for sem in range(num_sem_labels):
            I = np.sum(np.logical_and(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            U = np.sum(np.logical_or(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            I_all[sem] += I
            U_all[sem] += U
    if visual:
        for sem in range(num_sem_labels):
            if U_all[sem] == 0:
                I_all[sem] = 1
                U_all[sem] = 1
    return I_all / U_all 


def export(data, seg, pred):
    global scene_id
    global val_scenes
    # global room_seg, room_pred
    # global visual_warning
    # visu = visu.split('_')
    b = data.shape[0]
    for i in range(0, b):
        scene = val_scenes[scene_id + i]
        output_scene_dir = f'outputs/{args.exp_name}/visualization/{scene}'
        if not os.path.exists(output_scene_dir):
            os.makedirs(output_scene_dir)
        
        scene_seg = seg[i].unsqueeze(0).cpu().numpy()
        scene_pred = pred[i].unsqueeze(0).cpu().numpy()
        xyz = data[i, :3].cpu().numpy()
        xyz_label = np.concatenate((xyz, scene_pred), axis=0).transpose()
        xyz_label_gt = np.concatenate((xyz, scene_seg), axis=0).transpose()
        f = open(f'{output_scene_dir}/{scene}.txt', "a")
        f_gt = open(f'{output_scene_dir}/{scene}_gt.txt', "a")
        np.savetxt(f, xyz_label) 
        np.savetxt(f_gt, xyz_label_gt) 
    scene_id += b
        
def train(args, io):
    train_loader = DataLoader(ScanNet(args=args, partition='train', num_points=args.num_points), 
                            num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(ScanNet(args=args, partition='val', num_points=args.num_points), 
                            num_workers=8, batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'dgcnn':
        model = DGCNN_semseg(args).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))

    model = nn.DataParallel(model)
    if args.model_root:
        model.load_state_dict(torch.load(os.path.join(args.model_root, 'model_None.t7')))
        print(f"Continue training with model from {args.model_root}")
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, 20, 0.5, args.epochs)

    criterion = cal_loss

    writer = SummaryWriter()

    best_test_iou = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        train_label_seg = []
        for data, seg, weights, idx, label_list in train_loader:
            print(data.shape)
            data, seg = data.to(device), seg.to(device)
            #data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            seg_pred = model(data)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, args.num_sem_labels), seg.view(-1,1).squeeze())
            loss.backward()
            opt.step()
            pred = seg_pred.max(dim=2)[1]               # (batch_size, num_points)
            count += batch_size
            train_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()                  # (batch_size, num_points)
            pred_np = pred.detach().cpu().numpy()       # (batch_size, num_points)
            train_true_cls.append(seg_np.reshape(-1))       # (batch_size * num_points)
            train_pred_cls.append(pred_np.reshape(-1))      # (batch_size * num_points)
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        train_ious = calculate_sem_IoU(train_pred_seg, train_true_seg)
        train_miou = np.mean(train_ious)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, train iou: %.6f' % (epoch, 
                                                                                                  train_loss*1.0/count,
                                                                                                  train_acc,
                                                                                                  avg_per_class_acc,
                                                                                                  train_miou)
        io.cprint(outstr)
        writer.add_scalar('Loss/train', train_loss*1.0/count, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('avg.Accuracy/train', avg_per_class_acc, epoch)
        writer.add_scalar('mIoU/train', train_miou, epoch)
        
                
        ####################
        # Val
        ####################
        val_loss = 0.0
        count = 0.0
        model.eval()
        val_true_cls = []
        val_pred_cls = []
        val_true_seg = []
        val_pred_seg = []
        for data, seg in val_loader:
            data, seg = data.to(device), seg.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            seg_pred = model(data)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, args.num_sem_labels), seg.view(-1,1).squeeze())
            pred = seg_pred.max(dim=2)[1]
            count += batch_size
            val_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            val_true_cls.append(seg_np.reshape(-1))
            val_pred_cls.append(pred_np.reshape(-1))
            val_true_seg.append(seg_np)
            val_pred_seg.append(pred_np)
        val_true_cls = np.concatenate(val_true_cls)
        val_pred_cls = np.concatenate(val_pred_cls)
        test_acc = metrics.accuracy_score(val_true_cls, val_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(val_true_cls, val_pred_cls)
        val_true_seg = np.concatenate(val_true_seg, axis=0)
        val_pred_seg = np.concatenate(val_pred_seg, axis=0)
        test_ious = calculate_sem_IoU(val_pred_seg, val_true_seg)
        test_miou = np.mean(test_ious)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (epoch,
                                                                                            val_loss*1.0/count,
                                                                                            test_acc,
                                                                                            avg_per_class_acc,
                                                                                            test_miou)
        io.cprint(outstr)
        writer.add_scalar('Loss/test', val_loss*1.0/count, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)
        writer.add_scalar('avg.Accuracy/test', avg_per_class_acc, epoch)
        writer.add_scalar('mIoU/test', test_miou, epoch)
        if np.mean(test_ious) >= best_test_iou:
            best_test_iou = np.mean(test_ious)
            torch.save(model.state_dict(), 'outputs/%s/models/model_None.t7' % (args.exp_name))


def test(args, io):
    if args.visu:
        global val_scenes
        val_scenes = np.loadtxt('val_scenes.txt',dtype="U")

    device = torch.device("cuda" if args.cuda else "cpu")
                
    #Try to load models
    #semseg_colors = test_loader.dataset.semseg_colors
    if args.model == 'dgcnn':
        model = DGCNN_semseg(args).to(device)
    else:
        raise Exception("Not implemented")

    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(os.path.join(args.model_root, 'model_None.t7')))
    model = model.eval()

    dataset = ScanNet(args=args, partition='val', num_points=args.num_points)
    val_loader = DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    criterion = cal_loss

    ####################
    # Val
    ####################
    val_loss = 0.0
    count = 0.0
    model.eval()
    val_true_cls = []
    val_pred_cls = []
    val_true_seg = []
    val_pred_seg = []
    for data, seg in val_loader:
        data, seg = data.to(device), seg.to(device)
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        seg_pred = model(data)
        seg_pred = seg_pred.permute(0, 2, 1).contiguous()
        loss = criterion(seg_pred.view(-1, args.num_sem_labels), seg.view(-1,1).squeeze())
        pred = seg_pred.max(dim=2)[1]
        count += batch_size
        val_loss += loss.item() * batch_size
        seg_np = seg.cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        val_true_cls.append(seg_np.reshape(-1))
        val_pred_cls.append(pred_np.reshape(-1))
        val_true_seg.append(seg_np)
        val_pred_seg.append(pred_np)
        if args.visu:
            export(data, seg, pred) 
    val_true_cls = np.concatenate(val_true_cls)
    val_pred_cls = np.concatenate(val_pred_cls)
    val_acc = metrics.accuracy_score(val_true_cls, val_pred_cls)
    avg_per_class_acc = metrics.balanced_accuracy_score(val_true_cls, val_pred_cls)
    val_true_seg = np.concatenate(val_true_seg, axis=0)
    val_pred_seg = np.concatenate(val_pred_seg, axis=0)
    val_ious = calculate_sem_IoU(val_pred_seg, val_true_seg)
    val_miou = np.mean(val_ious)
    val_loss = val_loss*1.0/count
    outstr = 'loss: %.6f, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (val_loss, val_acc, avg_per_class_acc, val_miou)
    io.cprint(outstr)

    # Open file in append mode
    with open('evaluation_results.csv', 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow([val_loss, val_acc, avg_per_class_acc, val_miou, args.model_root])

    # Open file in append mode
    with open('evaluation_results.csv', 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow([val_loss, val_acc, avg_per_class_acc, val_miou, args.model_root])


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['dgcnn'],
                        help='Model to use, [dgcnn]')
    parser.add_argument('--dataset', type=str, default='S3DIS', metavar='N',
                        choices=['ScanNet'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', dest='use_sgd', action='store_true',
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--weight_decay', type=float, default=0.0001, metavar='eta',
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--scannet_dataset_alternative', type=str, default=None, metavar='N',
                        help='alternative scannet dataset', choices=["fps", "subvolume", "subvolume_normal"])
    parser.add_argument('--num_points', type=int, default=4096,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--num_features', type=int, default=3, metavar='N',
                        help='num of features (3=XYZ, 6=XYZRGB, 9=XYZRGBnormal', choices=[3, 6, 9])
    parser.add_argument('--num_sem_labels', type=int, default=13, metavar='N',
                        help='num of semanic labels')
    parser.add_argument('--model_root', type=str, default='', metavar='N',
                        help='Pretrained model root')
    parser.add_argument('--visu', type=bool, default=False,
                        help='visualize the model')
    args = parser.parse_args()

    _init_()

    io = IOStream('outputs/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
