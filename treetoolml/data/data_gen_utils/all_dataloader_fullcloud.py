from numpy.random.mtrand import normal
from treetoolml.data.data_gen_utils.custom_loaders import *
from treetoolml.IndividualTreeExtraction.utils.py_util import normalize
from tqdm import tqdm
import os
import numpy as np
from treetoolml.utils.tictoc import bench_dict
import treetoolml.utils.py_util as py_util
from treetoolml.Libraries.open3dvis import open3dpaint
from treetoolml.data.data_gen_utils.dataloaders import data_loader, data_loader_fullcloud


def eulerAnglesToRotationMatrix(theta):
    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta[0]), -np.sin(theta[0])],
            [0, np.sin(theta[0]), np.cos(theta[0])],
        ]
    )
    R_y = np.array(
        [
            [np.cos(theta[1]), 0, np.sin(theta[1])],
            [0, 1, 0],
            [-np.sin(theta[1]), 0, np.cos(theta[1])],
        ]
    )
    R_z = np.array(
        [
            [np.cos(theta[2]), -np.sin(theta[2]), 0],
            [np.sin(theta[2]), np.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


class all_data_loader_cloud:
    def __init__(
        self, onlyTrees=False, preprocess=False, default=True, train_split=False, new_paris=False
    ) -> None:
        self.loader_list = {}
        self.tree_list = []
        self.non_tree_list = []
        self.full_cloud = data_loader_fullcloud(onlyTrees, preprocess, train_split)
        self.loader_list['full_filter'] = self.full_cloud
        # self.non_tree_list.append(self.iqumulus)
        self.tree_list.append(self.full_cloud)

    def load_all(self, dir=None):
        for key, value in tqdm(self.loader_list.items()):
            value.load_data(dir)

        print("full_cloud trees:", len(self.full_cloud.get_trees()))

    def get_random_background(self, radius=10):
        choice = np.random.choice(len(self.non_tree_list))
        return self.non_tree_list[choice].get_random_background(radius)

    def get_random_forground(self, train=False):
        choice = np.random.choice(len(self.tree_list))
        return self.tree_list[choice].get_random_forground(train)

    def get_tree_cluster(
        self,
        max_trees=4,
        split=None,
        translation_xy=4,
        translation_z=0.2,
        scale=0.2,
        xy_rotation=0,
        dist_between=3,
        do_normalize=False,
        zero_floor = True,
    ):
        bench_dict['get cluster'].gstep()
        number_of_trees = np.random.randint(1, max_trees + 1)
        scale = scale
        xyrotmin = -np.deg2rad(xy_rotation)
        xyrotmax = np.deg2rad(xy_rotation)
        cluster = []
        cluster_center = []
        centers = []
        bench_dict['get cluster'].step('start')
        for i in range(number_of_trees):
            tree, _center = self.get_random_forground(split)
            bench_dict['get cluster'].step('forground')

            # R = eulerAnglesToRotationMatrix([0,0,np.random.uniform(0,np.pi*2)])
            R = eulerAnglesToRotationMatrix(
                [
                    np.random.uniform(xyrotmin, xyrotmax),
                    np.random.uniform(xyrotmin, xyrotmax),
                    np.random.uniform(0, np.pi * 2),
                ]
            )
            bench_dict['get cluster'].step('rot')
            while True:
                translation = np.array(
                    [
                        np.random.uniform(-translation_xy, translation_xy),
                        np.random.uniform(-translation_xy, translation_xy),
                        np.random.uniform(-translation_z, translation_z),
                    ]
                )

                if len(cluster_center) == 0:
                    break
                dists = np.linalg.norm(cluster_center - translation, axis=1)
                if np.min(dists) > dist_between:
                    break
            bench_dict['get cluster'].step('tras')

            scaler = 1 + scale * (np.random.rand() - 0.5)
            rt = np.eye(4)
            rt[:3, 3] = translation
            rt[:3, :3] = scaler * R
            htree = np.hstack([tree, np.ones_like(tree[:, 0:1])])
            hcenters = np.hstack([_center, 1]).reshape(1,4)
            new_tree = (rt @ htree.T).T[:,:3]
            new_center = (rt @ hcenters.T).T[:,:3]
            bench_dict['get cluster'].step('transform')
            cluster.append(new_tree)
            cluster_center.append(translation)
            centers.append(new_center)
            bench_dict['get cluster'].step('append')

        labels = [n * np.ones_like(i[:, :1]) for n, i in enumerate(cluster)]
        bench_dict['get cluster'].gstop()
        return cluster, labels, centers

    def get_tree_centers(self, temp_xyz, labels, return_cloud=False, return_filtered=False):
        bench_dict["loader"].gstep()
        downsample_idx = py_util.downsample(temp_xyz, 0.01, True)
        bench_dict["loader"].step("start_downsample")

        filterd_temp_xyz = temp_xyz[downsample_idx]
        filterd_object_label = labels[downsample_idx]

        bench_dict["loader"].step("start_normal")
        
        unique_object_label = np.unique(filterd_object_label)
        temp_multi_objects_centers = []
        bench_dict["loader"].step("start")
        for j in range(np.size(unique_object_label)):
            ###get single object
            temp_index = np.where(filterd_object_label == unique_object_label[j])
            temp_index_object_xyz = filterd_temp_xyz[temp_index[0], :]
            bench_dict["loader"].step("compute 1")
            down_points = filterd_temp_xyz[filterd_object_label.flatten() == unique_object_label[j]]
            bench_dict["loader"].step("compute down1")
            filter_idx = py_util.normal_filter(
                down_points, 0.1, 0.4, 0.1, return_indexes=True
            )
            filtered_points = down_points[filter_idx]
            bench_dict["loader"].step("filter")
            if len(filtered_points) < len(down_points) * 0.1:
                temp_object_center_xyz = np.mean(temp_index_object_xyz, 0)
            else:
                temp_object_center_xyz = py_util.trunk_center(filtered_points)
                if len(temp_object_center_xyz) == 0:
                    temp_object_center_xyz = np.mean(filtered_points, 0)
            bench_dict["loader"].step("trunk")
            temp_multi_objects_centers.append([temp_object_center_xyz,unique_object_label[j]])
            bench_dict["loader"].step("compute other")
        bench_dict["loader"].gstop()

        if return_cloud:
            if return_filtered:
                indexes = py_util.normal_filter(filterd_temp_xyz, return_indexes=True, search_radius=0.1, verticality_threshold=0.4, curvature_threshold=0.1)
                out_filterd_temp_xyz = filterd_temp_xyz[indexes]
                out_filterd_object_label = filterd_object_label[indexes]
            else:
                out_filterd_temp_xyz = filterd_temp_xyz
                out_filterd_object_label = filterd_object_label
            return temp_multi_objects_centers, out_filterd_temp_xyz, out_filterd_object_label
        
        return temp_multi_objects_centers