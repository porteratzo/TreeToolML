"""
Created on Mon July 11 18:50:39 2020

@author: Haifeng Luo
"""

import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, "utils"))
import py_util
from torch.utils.data import Dataset
import glob


class tree_dataset(Dataset):
    def __init__(self, trainingdata_path, num_points):
        self.files = py_util.get_data_set(trainingdata_path)
        self.path = trainingdata_path
        self.num_points = num_points

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        temp_point_set = py_util.load_data(os.path.join(self.path, self.files[index]))

        temp_xyz = temp_point_set[:, :3]
        temp_xyz = py_util.normalize(temp_xyz)

        object_label = temp_point_set[:, 3]
        unique_object_label = np.unique(object_label)

        temp_multi_objects_sample = []
        for j in range(np.size(unique_object_label)):
            ###for each object
            temp_index = np.where(object_label == unique_object_label[j])
            temp_index_object_xyz = temp_xyz[temp_index[0], :]
            ###object_label
            temp_object_label = np.expand_dims(object_label[temp_index[0]], axis=-1)
            ###center point
            temp_object_center_xyz = py_util.compute_object_center(
                temp_index_object_xyz
            )
            ###deta_x + x = center_point ---->deta_x = center_point - x
            temp_direction_label = temp_object_center_xyz - temp_index_object_xyz
            ####[x, y, z, deta_x, deta_y, deta_z]
            temp_xyz_direction_label_concat = np.concatenate(
                [temp_index_object_xyz, temp_direction_label, temp_object_label],
                axis=-1,
            )
            ####
            temp_multi_objects_sample.append(temp_xyz_direction_label_concat)

        temp_multi_objects_sample = np.vstack(temp_multi_objects_sample)
        ###
        temp_multi_objects_sample = py_util.shuffle_data(temp_multi_objects_sample)
        temp_multi_objects_sample = temp_multi_objects_sample[: self.num_points, :]
        ###
        training_xyz = temp_multi_objects_sample[:, :3]
        training_direction_label = temp_multi_objects_sample[:, 3:-1]
        training_object_label = temp_multi_objects_sample[:, -1]

        return training_xyz, training_direction_label, training_object_label
