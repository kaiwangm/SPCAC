import sys
import os
import numpy as np

# import torch
import torch.utils.data as data

# load ply
import open3d as o3d
import h5py

class sensat_urban(data.Dataset):
    def __init__(self, dir_path='./datasets/sensat_urban', depth=10, split="train", use_hdf5=True):
        self.dir_path = dir_path
        self.depth = depth
        self.split = split

        self.data_path_list = []

        if use_hdf5 == False:
            for root, dirs, files in os.walk(dir_path):
                self.data_path_list.extend([os.path.join(root, x) for x in files])

            pointcloud_num = len(self.data_path_list)

            if split == "train":
                self.data_path_list = self.data_path_list[:int(
                    pointcloud_num*0.95)]
            elif split == "test":
                self.data_path_list = self.data_path_list[int(
                    pointcloud_num*0.95):]
            
            self.length = len(self.data_path_list)
        else:
            hdf5_path = './datasets/sensat_urban.hdf5'
            if not os.path.exists(hdf5_path):
                from dataset.datasets import _ensure_hdf5_available
                _ensure_hdf5_available(hdf5_path, 'sensat_urban')
            self.hdf5_object = h5py.File(hdf5_path, 'r')

            self.length = len(self.hdf5_object['dataset'][self.split])
        
        self.use_hdf5 = use_hdf5
        self.items = dict()

    def __getitem__(self, index):
        # load ply point cloud
        real_index = index
        if real_index in self.items:
            return self.items[real_index]

        if self.use_hdf5 == False:
            pc = o3d.io.read_point_cloud(self.data_path_list[index])

            points = np.asarray(pc.points)
            colors = np.asarray(pc.colors)
        else:
            idx_name = str(real_index)
            points = np.asarray(
                self.hdf5_object['dataset'][self.split][idx_name]['points'][:])
            colors = np.asarray(
                self.hdf5_object['dataset'][self.split][idx_name]['colors'][:])

        # self.items[real_index] = (points, colors)

        return points, colors

    def __len__(self):
        return self.length
