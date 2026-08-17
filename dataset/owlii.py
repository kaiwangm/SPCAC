import sys
import os
import numpy as np

import torch
import torch.utils.data as data

# load ply
import open3d as o3d
# load hdf5 (fast loading)
import h5py


def _split_has_ply(dir_path, split):
    """True if ``dir_path/{split}`` contains at least one PLY file."""
    split_path = os.path.join(dir_path, split)
    if not os.path.isdir(split_path):
        return False
    for root, _, files in os.walk(split_path):
        if any(name.lower().endswith('.ply') for name in files):
            return True
    return False


def _collect_ply_paths(dir_path, split):
    split_path = os.path.join(dir_path, split)
    paths = []
    for root, _, files in os.walk(split_path):
        for name in files:
            if name.lower().endswith('.ply'):
                paths.append(os.path.join(root, name))
    paths.sort()
    return paths


class owlii(data.Dataset):
    def __init__(self, dir_path, depth, split, use_hdf5=True):
        self.dir_path = dir_path
        self.depth = depth
        self.split = split
        self.items = dict()

        hdf5_path = './datasets/owlii.hdf5'
        # Sequence profiles (e.g. dancer11/) use local PLYs instead of the shared
        # owlii.hdf5, which mixes sequences and would yield identical RD points.
        prefer_ply = _split_has_ply(dir_path, split)
        self.use_hdf5 = bool(use_hdf5) and not prefer_ply

        if self.use_hdf5:
            if not os.path.exists(hdf5_path):
                from dataset.datasets import _ensure_hdf5_available
                _ensure_hdf5_available(hdf5_path, 'owlii')
            if os.path.exists(hdf5_path):
                self.hdf5_object = h5py.File(hdf5_path, 'r')
                self.length = len(self.hdf5_object['dataset'][split])
                return
            self.use_hdf5 = False

        self.data_path_list = _collect_ply_paths(dir_path, split)
        self.length = len(self.data_path_list)

    def __getitem__(self, index):
        # load point cloud
        real_index = index
        if real_index in self.items:
            return self.items[real_index]

        if self.use_hdf5:
            idx_name = str(real_index)
            points = np.asarray(
                self.hdf5_object['dataset'][self.split][idx_name]['points'][:])
            colors = np.asarray(
                self.hdf5_object['dataset'][self.split][idx_name]['colors'][:])
        else:
            if self.split == "test":
                print(self.data_path_list[index])

            pc = o3d.io.read_point_cloud(self.data_path_list[index])

            points = np.asarray(pc.points)
            colors = np.asarray(pc.colors)

        if self.depth == 10:
            # 11 -> 10 bit
            points = points // 2
            # unique
            points, unique_indices = np.unique(
                points, axis=0, return_index=True)
            colors = colors[unique_indices]

        return points, colors

    def __len__(self):
        return self.length

    def get_from_path(self, path):
        pc = o3d.io.read_point_cloud(path)

        points = np.asarray(pc.points)
        colors = np.asarray(pc.colors)

        points = torch.from_numpy(points).int().unsqueeze(0)
        colors = torch.from_numpy(colors).float().unsqueeze(0)

        return points, colors
