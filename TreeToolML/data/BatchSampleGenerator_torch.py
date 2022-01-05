"""
Created on Mon July 11 18:50:39 2020

@author: Haifeng Luo
"""

import pclpy
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, "utils"))
import TreeToolML.utils.py_util as py_util
from TreeToolML.utils.tictoc import bench_dict
from torch.utils.data import Dataset
from TreeToolML.Libraries.open3dvis import open3dpaint


class tree_dataset(Dataset):
    def __init__(self, trainingdata_path, num_points, return_centers=False, normal_filter=False):
        self.files = py_util.get_data_set(trainingdata_path)
        self.path = trainingdata_path
        self.num_points = num_points
        self.return_centers = return_centers
        self.normal_filter = normal_filter

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        bench_dict["loader"].gstep()
        temp_point_set = py_util.load_data(os.path.join(self.path, self.files[index]))

        temp_xyz = py_util.data_preprocess(temp_point_set)

        object_label = temp_point_set[:, 3]
        unique_object_label = np.unique(object_label)

        temp_multi_objects_sample = []
        temp_multi_objects_centers = []
        bench_dict["loader"].step("start")
        for j in range(np.size(unique_object_label)):
            ###for each object
            temp_index = np.where(object_label == unique_object_label[j])
            temp_index_object_xyz = temp_xyz[temp_index[0], :]
            ###object_label
            temp_object_label = np.expand_dims(object_label[temp_index[0]], axis=-1)
            ###center point
            bench_dict["loader"].step("compute 1")
            down_points = py_util.downsample(temp_index_object_xyz)
            filtered_points = py_util.normal_filter(
                down_points, 0.05, 0.2, 0.1
            )
            bench_dict["loader"].step("filter")
            if len(filtered_points) < len(down_points) * 0.1:
                temp_object_center_xyz = np.mean(temp_index_object_xyz, 0)
            else:
                temp_object_center_xyz = py_util.trunk_center(filtered_points)
                if len(temp_object_center_xyz) == 0:
                    temp_object_center_xyz = np.mean(filtered_points, 0)
            bench_dict["loader"].step("trunk")

            temp_direction_label = temp_object_center_xyz - temp_index_object_xyz
            temp_xyz_direction_label_concat = np.concatenate(
                [temp_index_object_xyz, temp_direction_label, temp_object_label],
                axis=-1,
            )
            temp_multi_objects_sample.append(temp_xyz_direction_label_concat)
            temp_multi_objects_centers.append(temp_object_center_xyz)
            bench_dict["loader"].step("compute other")

        temp_multi_objects_sample = np.vstack(temp_multi_objects_sample)

        if self.normal_filter:
            d_indexes = py_util.downsample(temp_multi_objects_sample[:,:3], 0.01, return_idx=True)
            _temp_multi_objects_sample = temp_multi_objects_sample[d_indexes]
            indexes = py_util.normal_filter(_temp_multi_objects_sample[:,:3], return_indexes=True)
            _temp_multi_objects_sample = _temp_multi_objects_sample[indexes]
        
            if len(_temp_multi_objects_sample) > self.num_points:
                _temp_multi_objects_sample = py_util.shuffle_data(_temp_multi_objects_sample)
                temp_multi_objects_sample = _temp_multi_objects_sample[: self.num_points, :]
            else:
                temp_multi_objects_sample = py_util.shuffle_data(temp_multi_objects_sample)
                temp_multi_objects_sample = temp_multi_objects_sample[: self.num_points, :]
        else:
            temp_multi_objects_sample = py_util.shuffle_data(temp_multi_objects_sample)
            temp_multi_objects_sample = temp_multi_objects_sample[: self.num_points, :]            
        ###
        training_xyz = temp_multi_objects_sample[:, :3]
        training_direction_label = temp_multi_objects_sample[:, 3:-1]
        training_object_label = temp_multi_objects_sample[:, -1]
        bench_dict["loader"].step("shuffle")
        bench_dict["loader"].gstop()

        if self.return_centers:
            return (
                training_xyz,
                training_direction_label,
                training_object_label,
                temp_multi_objects_centers,
            )
        else:
            return training_xyz, training_direction_label, training_object_label
