#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM

Modified by 
@Author: An Tao, Pengliang Ji
@Contact: ta19@mails.tsinghua.edu.cn, jpl1723@buaa.edu.cn
@Time: 2021/7/20 7:49 PM
"""


import os
import sys
import glob
import h5py
import numpy as np
import torch
import json
import cv2
import pickle
import open3d as o3d
from plyfile import PlyData, PlyElement
from torch.utils.data import Dataset


def download_modelnet40():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % ('modelnet40_ply_hdf5_2048', DATA_DIR))
        os.system('rm %s' % (zipfile))


def download_shapenetpart():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data')):
        www = 'https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % ('hdf5_data', os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data')))
        os.system('rm %s' % (zipfile))


def download_S3DIS():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data')):
        www = 'https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % ('indoor3d_sem_seg_hdf5_data', DATA_DIR))
        os.system('rm %s' % (zipfile))
    if not os.path.exists(os.path.join(DATA_DIR, 'Stanford3dDataset_v1.2_Aligned_Version')):
        if not os.path.exists(os.path.join(DATA_DIR, 'Stanford3dDataset_v1.2_Aligned_Version.zip')):
            print('Please download Stanford3dDataset_v1.2_Aligned_Version.zip \
                from https://goo.gl/forms/4SoGp4KtH1jfRqEj2 and place it under data/')
            sys.exit(0)
        else:
            zippath = os.path.join(DATA_DIR, 'Stanford3dDataset_v1.2_Aligned_Version.zip')
            os.system('unzip %s' % (zippath))
            os.system('mv %s %s' % ('Stanford3dDataset_v1.2_Aligned_Version', DATA_DIR))
            os.system('rm %s' % (zippath))


def load_data_cls(partition):
    download_modelnet40()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', '*%s*.h5'%partition)):
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def load_data_partseg(partition):
    download_shapenetpart()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    all_seg = []
    if partition == 'trainval':
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*train*.h5')) \
               + glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*val*.h5'))
    else:
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*%s*.h5'%partition))
    for h5_name in file:
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        seg = f['pid'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
        all_seg.append(seg)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_seg = np.concatenate(all_seg, axis=0)
    return all_data, all_label, all_seg


def prepare_test_data_semseg():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(os.path.join(DATA_DIR, 'stanford_indoor3d')):
        os.system('python prepare_data/collect_indoor3d_data.py')
    if not os.path.exists(os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data_test')):
        os.system('python prepare_data/gen_indoor3d_h5.py')


def load_data_semseg(partition, test_area):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    download_S3DIS()
    prepare_test_data_semseg()
    if partition == 'train':
        data_dir = os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data')
    else:
        data_dir = os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data_test')
    with open(os.path.join(data_dir, "all_files.txt")) as f:
        all_files = [line.rstrip() for line in f]
    with open(os.path.join(data_dir, "room_filelist.txt")) as f:
        room_filelist = [line.rstrip() for line in f]
    data_batchlist, label_batchlist = [], []
    for f in all_files:
        file = h5py.File(os.path.join(DATA_DIR, f), 'r+')
        data = file["data"][:]
        label = file["label"][:]
        data_batchlist.append(data)
        label_batchlist.append(label)
    data_batches = np.concatenate(data_batchlist, 0)
    seg_batches = np.concatenate(label_batchlist, 0)
    test_area_name = "Area_" + test_area
    train_idxs, test_idxs = [], []
    for i, room_name in enumerate(room_filelist):
        if test_area_name in room_name:
            test_idxs.append(i)
        else:
            train_idxs.append(i)
    if partition == 'train':
        all_data = data_batches[train_idxs, ...]
        all_seg = seg_batches[train_idxs, ...]
    else:
        all_data = data_batches[test_idxs, ...]
        all_seg = seg_batches[test_idxs, ...]
    return all_data, all_seg


def load_color_partseg():
    colors = []
    labels = []
    f = open("prepare_data/meta/partseg_colors.txt")
    for line in json.load(f):
        colors.append(line['color'])
        labels.append(line['label'])
    partseg_colors = np.array(colors)
    partseg_colors = partseg_colors[:, [2, 1, 0]]
    partseg_labels = np.array(labels)
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_size = 1350
    img = np.zeros((1350, 1890, 3), dtype="uint8")
    cv2.rectangle(img, (0, 0), (1900, 1900), [255, 255, 255], thickness=-1)
    column_numbers = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
    column_gaps = [320, 320, 300, 300, 285, 285]
    color_size = 64
    color_index = 0
    label_index = 0
    row_index = 16
    for row in range(0, img_size):
        column_index = 32
        for column in range(0, img_size):
            color = partseg_colors[color_index]
            label = partseg_labels[label_index]
            length = len(str(label))
            cv2.rectangle(img, (column_index, row_index), (column_index + color_size, row_index + color_size),
                          color=(int(color[0]), int(color[1]), int(color[2])), thickness=-1)
            img = cv2.putText(img, label, (column_index + int(color_size * 1.15), row_index + int(color_size / 2)),
                              font,
                              0.76, (0, 0, 0), 2)
            column_index = column_index + column_gaps[column]
            color_index = color_index + 1
            label_index = label_index + 1
            if color_index >= 50:
                cv2.imwrite("prepare_data/meta/partseg_colors.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                return np.array(colors)
            elif (column + 1 >= column_numbers[row]):
                break
        row_index = row_index + int(color_size * 1.3)
        if (row_index >= img_size):
            break


def load_color_semseg():
    colors = []
    labels = []
    f = open("prepare_data/meta/semseg_colors.txt")
    for line in json.load(f):
        colors.append(line['color'])
        labels.append(line['label'])
    semseg_colors = np.array(colors)
    semseg_colors = semseg_colors[:, [2, 1, 0]]
    partseg_labels = np.array(labels)
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_size = 1500
    img = np.zeros((500, img_size, 3), dtype="uint8")
    cv2.rectangle(img, (0, 0), (img_size, 750), [255, 255, 255], thickness=-1)
    color_size = 64
    color_index = 0
    label_index = 0
    row_index = 16
    for _ in range(0, img_size):
        column_index = 32
        for _ in range(0, img_size):
            color = semseg_colors[color_index]
            label = partseg_labels[label_index]
            length = len(str(label))
            cv2.rectangle(img, (column_index, row_index), (column_index + color_size, row_index + color_size),
                          color=(int(color[0]), int(color[1]), int(color[2])), thickness=-1)
            img = cv2.putText(img, label, (column_index + int(color_size * 1.15), row_index + int(color_size / 2)),
                              font,
                              0.7, (0, 0, 0), 2)
            column_index = column_index + 200
            color_index = color_index + 1
            label_index = label_index + 1
            if color_index >= 13:
                cv2.imwrite("prepare_data/meta/semseg_colors.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                return np.array(colors)
            elif (column_index >= 1280):
                break
        row_index = row_index + int(color_size * 1.3)
        if (row_index >= img_size):
            break  
    

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    theta = np.pi*2 * np.random.uniform()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) # random rotation (x,z)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data_cls(partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


class ShapeNetPart(Dataset):
    def __init__(self, num_points, partition='train', class_choice=None):
        self.data, self.label, self.seg = load_data_partseg(partition)
        self.cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4, 
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9, 
                       'motor': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
        self.seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        self.index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
        self.num_points = num_points
        self.partition = partition        
        self.class_choice = class_choice
        self.partseg_colors = load_color_partseg()
        
        if self.class_choice != None:
            id_choice = self.cat2id[self.class_choice]
            indices = (self.label == id_choice).squeeze()
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.seg = self.seg[indices]
            self.seg_num_all = self.seg_num[id_choice]
            self.seg_start_index = self.index_start[id_choice]
        else:
            self.seg_num_all = 50
            self.seg_start_index = 0
            
      
    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        seg = self.seg[item][:self.num_points]
        if self.partition == 'trainval':
            # pointcloud = translate_pointcloud(pointcloud)
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
        return pointcloud, label, seg

    def __len__(self):
        return self.data.shape[0]


class S3DIS(Dataset):
    def __init__(self, num_points=4096, partition='train', test_area='1'):
        self.data, self.seg = load_data_semseg(partition, test_area)
        self.num_points = num_points
        self.partition = partition    
        self.semseg_colors = load_color_semseg()

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        seg = self.seg[item][:self.num_points]
        if self.partition == 'train':
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
        seg = torch.LongTensor(seg)
        return pointcloud, seg

    def __len__(self):
        return self.data.shape[0]

class ScanNet(Dataset):
    def __init__(self, path, partition='train'):
        self.partition = partition
        if partition == 'train':
            self.data = glob.glob(os.path.join(path, "scans/*/"))
        if partition == 'test':
            self.data = glob.glob(os.path.join(path, "scans_test/*/"))

    def __getitem__(self, idx):
        pcd_in = glob.glob(os.path.join(self.data[idx], "*clean_2.ply"))[0]
        pcd_label = glob.glob(os.path.join(self.data[idx], "*.labels.ply"))[0]
        pointcloud = PlyData.read(pcd_label)
        vertex, face = pointcloud['vertex'].data, pointcloud['face'].data
        pcd = o3d.io.read_point_cloud(pcd_label)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        return points, colors
    
    def __len__(self):
        return len(self.data)

class ScanNet_Subvolume(Dataset):
    
    # def __init__(self, path, block_point_threshhold=4096, box_size=1.5, partition='train'):
    #     self.partition = partition
    #     if partition == 'train':
    #         self.data = glob.glob(os.path.join(path, "scans/*/"))
    #         print(len(self.data))
    #     if partition == 'test':
    #         self.data = glob.glob(os.path.join(path, "scans_test/*/"))
    #     self.block_point_threshhold = block_point_threshhold
        
    # def __getitem__(self, idx):
    #     subvolume_points = []
    #     step_size = 0.5
    #     volume_size = 1.5
    #     tolerance = 0.0
    #     # Read input scene as point cloud (N, 3)
    #     pcd_in = glob.glob(os.path.join(self.data[idx], "*.labels.ply"))[0]
    #     data = np.load(pcd_in, allow_pickle=True)
    #     scene_points = data[:, 0:3]
    #     colors = data[:, 3:6]
    #     instance_labels = data[:, 6]
    #     semantic_labels = data[:, 7]
    #     print(scene_points.shape)
    #     print(colors.shape)
    #     print(instance_labels.shape)
    #     print(semantic_labels.shape)
    #     pcd = o3d.io.read_point_cloud(pcd_in)
    #     points = np.asarray(pcd.points)

    #     # Calculate number of subvolumes
    #     coordinate_max = np.max(points[:, 0:3], axis=0)
    #     coordinate_min = np.min(points[:, 0:3], axis=0)
    #     print(coordinate_max)
    #     print(coordinate_min)
    #     n_subvolume_x = np.ceil(coordinate_max[0] - coordinate_min[0] / step_size).astype(np.int8)
    #     n_subvolume_y = np.ceil(coordinate_max[1] - coordinate_min[1] / step_size).astype(np.int8)
    #     print(n_subvolume_x)
    #     print(n_subvolume_y)
    #     subvolume_centers = []
    #     block_points = []
    #     # Iterate over number of subvolumes to assign points to given subvolumes
    #     for i in range(n_subvolume_x):
    #         for j in range(n_subvolume_y):
    #             current_min = coordinate_min + [i * step_size, j * step_size, 0]
    #             current_max = current_min + [volume_size, volume_size, coordinate_max[2] - coordinate_min[2]]
    #             # Give points back which lies in the subvolume
    #             current_indices = np.where(np.sum((points[:, 0:3] >= (current_min - tolerance)) * (points[:, 0:3] <= (current_max + tolerance)), axis=1) == 3)[0]
    #             current_points = points[current_indices]
    #             subvolume_points.append(current_points)
    #             subvolume_centers.append((current_min + current_max) / 2.0)
    #             block_points.append(current_points.shape[0])
    #             print(subvolume_centers[-1])
    #             print(block_points[-1])
    #     # Merge subvolumes if they are below threshhold
    #     for i in range(0, len(block_points) - 1):
    #         if block_points[i] + block_points[i + 1] <= self.block_point_threshhold:
    #             # get index of the nearest block
    #             # merge
    #             subvolume_centers.pop(i)

    #     for num_points in block_points:
    #         if num_points > self.block_point_threshhold:
    #             # split
    #             pass
    #     # Divide subvolumes if they are above threshhold

    #     # Return list of subvolumes 
    #     return subvolume_points

    def __init__(self, root, split='train', num_class = 21, block_points = 8192, with_rgb = True):
        self.root = root
        self.split = split
        self.with_rgb = with_rgb
        self.block_points = block_points
        self.point_num = []
        self.data_filename = os.path.join(self.root, 'scannet_%s_rgb21c_pointid.pickle'%(split))
        with open(self.data_filename,'rb') as fp:
            self.scene_points_list = pickle.load(fp)
            self.semantic_labels_list = pickle.load(fp)
            self.scene_points_id = pickle.load(fp)
            self.scene_points_num = pickle.load(fp)
        if split=='train':
            labelweights = np.zeros(num_class)
            for seg in self.semantic_labels_list:
                self.point_num.append(seg.shape[0])
                tmp,_ = np.histogram(seg,range(num_class+1))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            #self.labelweights = 1/np.log(1.2+labelweights)
            self.labelweights = np.power(np.amax(labelweights) / labelweights, 1/3.0)
        else:
            self.labelweights = np.ones(num_class)
            for seg in self.semantic_labels_list:
                self.point_num.append(seg.shape[0])
    
    def chunks(self, l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]
    
    def split_data(self, data, idx):
        new_data = []
        for i in range(len(idx)):
            new_data += [np.expand_dims(data[idx[i]], axis = 0)]
        return new_data
    
    def nearest_dist(self, block_center, block_center_list):
        num_blocks = len(block_center_list)
        dist = np.zeros(num_blocks)
        for i in range(num_blocks):
            dist[i] = np.linalg.norm(block_center_list[i] - block_center, ord = 2) #i->j
        return np.argsort(dist)[0]

    def __getitem__(self, index):
        delta = 0.5
        if self.with_rgb:
            point_set_ini = self.scene_points_list[index]
        else:
            point_set_ini = self.scene_points_list[index][:, 0:3]
        semantic_seg_ini = self.semantic_labels_list[index].astype(np.int32)
        coordmax = np.max(point_set_ini[:, 0:3],axis=0)
        coordmin = np.min(point_set_ini[:, 0:3],axis=0)
        nsubvolume_x = np.ceil((coordmax[0]-coordmin[0])/delta).astype(np.int32)
        nsubvolume_y = np.ceil((coordmax[1]-coordmin[1])/delta).astype(np.int32)
        point_sets = []
        semantic_segs = []
        sample_weights = []
        point_idxs = []
        block_center = []
        for i in range(nsubvolume_x):
            for j in range(nsubvolume_y):
                curmin = coordmin+[i*delta,j*delta,0]
                curmax = curmin+[1.5,1.5,coordmax[2]-coordmin[2]]
                curchoice = np.sum((point_set_ini[:,0:3]>=(curmin-0.2))*(point_set_ini[:,0:3]<=(curmax+0.2)),axis=1)==3
                curchoice_idx = np.where(curchoice)[0]
                cur_point_set = point_set_ini[curchoice,:]
                cur_semantic_seg = semantic_seg_ini[curchoice]
                if len(cur_semantic_seg)==0:
                    continue
                mask = np.sum((cur_point_set[:,0:3]>=(curmin-0.001))*(cur_point_set[:,0:3]<=(curmax+0.001)),axis=1)==3
                sample_weight = self.labelweights[cur_semantic_seg]
                sample_weight *= mask # N
                point_sets.append(cur_point_set) # 1xNx3/6
                semantic_segs.append(cur_semantic_seg) # 1xN
                sample_weights.append(sample_weight) # 1xN
                point_idxs.append(curchoice_idx) #1xN
                block_center.append((curmin[0:2] + curmax[0:2]) / 2.0)

        # merge small blocks
        num_blocks = len(point_sets)
        block_idx = 0
        while block_idx < num_blocks:
            if point_sets[block_idx].shape[0] > 4096:
                block_idx += 1
                continue
            
            small_block_data = point_sets[block_idx].copy()
            small_block_seg = semantic_segs[block_idx].copy()
            small_block_smpw = sample_weights[block_idx].copy()
            small_block_idxs = point_idxs[block_idx].copy()
            small_block_center = block_center[block_idx].copy()
            point_sets.pop(block_idx)
            semantic_segs.pop(block_idx)
            sample_weights.pop(block_idx)
            point_idxs.pop(block_idx)
            block_center.pop(block_idx)
            nearest_block_idx = self.nearest_dist(small_block_center, block_center)
            point_sets[nearest_block_idx] = np.concatenate((point_sets[nearest_block_idx], small_block_data), axis = 0)
            semantic_segs[nearest_block_idx] = np.concatenate((semantic_segs[nearest_block_idx], small_block_seg), axis = 0)
            sample_weights[nearest_block_idx] = np.concatenate((sample_weights[nearest_block_idx], small_block_smpw), axis = 0)
            point_idxs[nearest_block_idx] = np.concatenate((point_idxs[nearest_block_idx], small_block_idxs), axis = 0)
            num_blocks = len(point_sets)

        #divide large blocks
        num_blocks = len(point_sets)
        div_blocks = []
        div_blocks_seg = []
        div_blocks_smpw = []
        div_blocks_idxs = []
        div_blocks_center = []
        for block_idx in range(num_blocks):
            cur_num_pts = point_sets[block_idx].shape[0]

            point_idx_block = np.array([x for x in range(cur_num_pts)])
            if point_idx_block.shape[0]%self.block_points != 0:
                makeup_num = self.block_points - point_idx_block.shape[0]%self.block_points
                np.random.shuffle(point_idx_block)
                point_idx_block = np.concatenate((point_idx_block,point_idx_block[0:makeup_num].copy()))

            np.random.shuffle(point_idx_block)

            sub_blocks = list(self.chunks(point_idx_block, self.block_points))

            div_blocks += self.split_data(point_sets[block_idx], sub_blocks)
            div_blocks_seg += self.split_data(semantic_segs[block_idx], sub_blocks)
            div_blocks_smpw += self.split_data(sample_weights[block_idx], sub_blocks)
            div_blocks_idxs += self.split_data(point_idxs[block_idx], sub_blocks)
            div_blocks_center += [block_center[block_idx].copy() for i in range(len(sub_blocks))]
        div_blocks = np.concatenate(tuple(div_blocks),axis=0)
        div_blocks_seg = np.concatenate(tuple(div_blocks_seg),axis=0)
        div_blocks_smpw = np.concatenate(tuple(div_blocks_smpw),axis=0)
        div_blocks_idxs = np.concatenate(tuple(div_blocks_idxs),axis=0)
        return div_blocks, div_blocks_seg, div_blocks_smpw, div_blocks_idxs
    def __len__(self):
        return len(self.scene_points_list)

    # def get_closest_subvolume(self, subvolume, )

if __name__ == '__main__':
    # train = ModelNet40(1024)
    # test = ModelNet40(1024, 'test')
    # data, label = train[0]
    # print(data.shape)
    # print(label.shape)

    # trainval = ShapeNetPart(2048, 'trainval')
    # test = ShapeNetPart(2048, 'test')
    # data, label, seg = trainval[0]
    # print(data.shape)
    # print(label.shape)
    # print(seg.shape)

    # train = S3DIS(4096)
    # test = S3DIS(4096, 'test')
    # data, seg = train[0]
    # print(data.shape)
    # print(seg.shape)

    train = ScanNet_Subvolume("/cluster/52/scannet")
    train[0]
    # test = ScanNet("/cluster/52/scannet", 'test')
    # print(test.data)

