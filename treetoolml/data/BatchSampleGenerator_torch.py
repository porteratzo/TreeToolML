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
import treetoolml.utils.py_util as py_util
from treetoolml.utils.tictoc import bench_dict
from torch.utils.data import Dataset
from treetoolml.data.data_gen_utils.dataloaders import downsample
from treetoolml.Libraries.open3dvis import open3dpaint


class tree_dataset(Dataset):
    def __init__(
        self, trainingdata_path, num_points, return_centers=False, normal_filter=False, distances=False
    ):
        self.files = py_util.get_data_set(trainingdata_path)
        self.path = trainingdata_path
        self.num_points = num_points
        self.return_centers = return_centers
        self.normal_filter = normal_filter
        self.distances = distances

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        while True:
            result = self.get_tree(index)
            if len(result[0]) == self.num_points:
                return result
            else:
                needed_n = self.num_points - len(result[0])
                newpoints = np.random.choice(np.arange(len(result[0])), needed_n)
                n_result_0 = np.vstack([result[0], result[0][newpoints]])
                n_result_1 = np.vstack([result[1], result[1][newpoints]])
                n_result_2 = np.hstack([result[2], result[2][newpoints]])
                return [n_result_0, n_result_1, n_result_2]

    def get_tree(self, index):
        bench_dict["loader"].gstep()
        data = py_util.load_data(os.path.join(self.path, self.files[index]))
        temp_point_set = data["cloud"]
        temp_centers = data["centers"]
        object_label = temp_point_set[:, 3]
        bench_dict["loader"].step("load data")
        temp_xyz = py_util.data_preprocess(temp_point_set)
        bench_dict["loader"].step("preprocess")
        downsample_idx = py_util.downsample(temp_xyz, 0.02, True)
        bench_dict["loader"].step("start_downsample")

        downsample_temp_xyz = temp_xyz[downsample_idx]
        downsample_object_label = object_label[downsample_idx]

        filterd_temp_xyz = downsample_temp_xyz
        filterd_object_label = downsample_object_label

        bench_dict["loader"].step("start_normal")

        unique_object_label = np.unique(filterd_object_label)

        temp_multi_objects_sample = []
        temp_multi_objects_centers = []
        bench_dict["loader"].step("start")
        for j in range(np.size(unique_object_label)):
            ###get single object
            temp_index = np.where(filterd_object_label == unique_object_label[j])
            center_index = np.where(temp_centers[:, 3] == unique_object_label[j])
            temp_index_object_xyz = filterd_temp_xyz[temp_index[0], :]
            temp_object_center_xyz = temp_centers[:, :3][center_index[0]][0]
            temp_object_label = np.expand_dims(
                filterd_object_label[temp_index[0]], axis=-1
            )
            bench_dict["loader"].step("compute 1")
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
            # d_indexes = py_util.downsample(temp_multi_objects_sample[:,:3], 0.02, return_idx=True)
            indexes = py_util.normal_filter(
                temp_multi_objects_sample[:, :3],
                return_indexes=True,
                search_radius=0.06,
            )
            _temp_multi_objects_sample = temp_multi_objects_sample[indexes]
            bench_dict["loader"].step("get centers")

            if len(_temp_multi_objects_sample) > self.num_points:
                _temp_multi_objects_sample = py_util.shuffle_data(
                    _temp_multi_objects_sample
                )
                temp_multi_objects_sample = _temp_multi_objects_sample[
                    : self.num_points, :
                ]
            else:
                temp_multi_objects_sample = py_util.shuffle_data(
                    temp_multi_objects_sample
                )
                temp_multi_objects_sample = temp_multi_objects_sample[
                    : self.num_points, :
                ]
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

class tree_dataset_cloud(tree_dataset):
    def get_tree(self, index):
        bench_dict["loader"].gstep()
        data = py_util.load_data(os.path.join(self.path, self.files[index]))
        temp_point_set = data["cloud"]
        temp_centers = data["centers"]
        points = temp_point_set[:, :3]
        object_label = temp_point_set[:, 3]
        unique_object_label = np.unique(object_label)

        temp_multi_objects_sample = []
        temp_multi_objects_centers = []
        bench_dict["loader"].step("start")
        for j in range(np.size(unique_object_label)):
            ###get single object
            temp_index = np.where(object_label == unique_object_label[j])
            center_index = np.where(temp_centers[:, 3] == unique_object_label[j])
            temp_index_object_xyz = points[temp_index[0], :]
            temp_object_center_xyz = temp_centers[:, :3][center_index[0]][0]
            temp_object_label = np.expand_dims(
                object_label[temp_index[0]], axis=-1
            )
            bench_dict["loader"].step("compute 1")
            temp_direction_label = temp_object_center_xyz - temp_index_object_xyz
            if self.distances:
                distances_label = np.linalg.norm(temp_direction_label[:,:2], axis=1).reshape([-1,1])
                temp_direction_label = temp_direction_label / np.linalg.norm(temp_direction_label, axis=1).reshape(
                    [-1, 1])
                temp_direction_label = np.hstack([temp_direction_label,distances_label])
            temp_xyz_direction_label_concat = np.concatenate(
                [temp_index_object_xyz, temp_direction_label, temp_object_label],
                axis=-1,
            )
            temp_multi_objects_sample.append(temp_xyz_direction_label_concat)
            temp_multi_objects_centers.append(temp_object_center_xyz)
            bench_dict["loader"].step("compute other")

        temp_multi_objects_sample = np.vstack(temp_multi_objects_sample)

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
