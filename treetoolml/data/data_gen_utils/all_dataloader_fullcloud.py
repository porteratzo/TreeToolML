from tqdm import tqdm
import numpy as np
from tictoc import bench_dict
import treetoolml.utils.py_util as py_util
from treetoolml.data.data_gen_utils.dataloaders import data_loader_fullcloud


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
        self,
        onlyTrees=False,
        preprocess=False,
        default=True,
        train_split=False,
        new_paris=False,
        normal_filter=False,
    ) -> None:
        self.loader_list = {}
        self.tree_list = []
        self.non_tree_list = []
        self.normal_filter = normal_filter
        self.full_cloud = data_loader_fullcloud(onlyTrees, preprocess, train_split)
        self.loader_list["full_filter"] = self.full_cloud
        # self.non_tree_list.append(self.iqumulus)
        self.tree_list.append(self.full_cloud)

    def load_all(self, dir=None):
        for key, value in tqdm(self.loader_list.items()):
            value.load_data(dir)

        print("full_cloud trees:", len(self.full_cloud.get_trees()))

    def get_random_background(self, radius=10):
        choice = np.random.choice(len(self.non_tree_list))
        return self.non_tree_list[choice].get_random_background(radius)

    def get_random_forground(self, train=False, trunks=False):
        choice = np.random.choice(len(self.tree_list))
        return self.tree_list[choice].get_random_forground(train, trunks)

    def get_tree_cluster(
        self,
        max_trees=4,
        split=None,
        translation_xy=4,
        translation_z=0.2,
        min_height=2,
        max_height=8,
        xy_rotation=0,
        dist_between=3,
        do_normalize=False,
        zero_floor=True,
        center_method=0,
        use_trunks=False,
        noise=0.0,
    ):
        bench_dict["get cluster"].gstep()
        number_of_trees = np.random.randint(1, max_trees + 1)
        xyrotmin = -np.deg2rad(xy_rotation)
        xyrotmax = np.deg2rad(xy_rotation)
        cluster = []
        cluster_center = []
        centers = []
        if use_trunks:
            trunks = []
        bench_dict["get cluster"].step("start")
        for i in range(number_of_trees):
            if use_trunks:
                tree_no_noise, _center, trunk = self.get_random_forground(split, use_trunks)
            else:
                tree_no_noise, _center = self.get_random_forground(split, use_trunks)
            bench_dict["get cluster"].step("forground")

            # R = eulerAnglesToRotationMatrix([0,0,np.random.uniform(0,np.pi*2)])
            R = eulerAnglesToRotationMatrix(
                [
                    np.random.uniform(xyrotmin, xyrotmax),
                    np.random.uniform(xyrotmin, xyrotmax),
                    np.random.uniform(0, np.pi * 2),
                ]
            )
            bench_dict["get cluster"].step("rot")
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
            bench_dict["get cluster"].step("tras")

            scaler = np.random.uniform(min_height, max_height)
            rt = np.eye(4)
            rt[:3, 3] = translation
            rt[:3, :3] = scaler * R
            if noise != 0.0:
                rand_noise = (
                    np.random.uniform(0.1, noise)
                    * (scaler - min_height)
                    / (max_height - min_height)
                )
                noise_cloud = np.random.rand(int(tree_no_noise.shape[0] * rand_noise), 3) - [
                    0.5,
                    0.5,
                    0,
                ]
                noise_cloud = noise_cloud * [0.2, 0.2, 1]
                tree = np.vstack([tree_no_noise, noise_cloud])
                if use_trunks:
                    trunk = np.hstack([trunk, np.repeat(False, len(noise_cloud))])
            else:
                tree = tree_no_noise
            bench_dict["get cluster"].step("noise")
            transform_points = np.vstack([tree, _center[np.newaxis, :]])
            transform_points = np.hstack(
                [transform_points, np.ones_like(transform_points[:, 0:1])]
            ).T
            bench_dict["get cluster"].step("stack")
            tranform_mat = (rt @ transform_points).T
            new_tree = tranform_mat[:-1, :3]
            new_center = tranform_mat[-1:, :3]
            bench_dict["get cluster"].step("transform")
            if use_trunks:
                trunks.append(trunk)
            bench_dict["get cluster"].step("appe")
            cluster.append(new_tree)
            cluster_center.append(translation)
            if center_method:
                centers.append(new_center)
            else:
                centers.append(np.mean(new_tree, axis=0))
            bench_dict["get cluster"].step("append")

        labels = np.vstack([n * np.ones_like(i[:, :1]) for n, i in enumerate(cluster)])
        if self.normal_filter:
            d_indexes = py_util.downsample(np.vstack(cluster), 0.06, return_idx=True)
            cluster = np.vstack(cluster)[d_indexes]
            if use_trunks:
                trunks = np.hstack(trunks)[d_indexes]
            labels = labels[d_indexes]
        else:
            cluster = np.vstack(cluster)
            if use_trunks:
                trunks = np.hstack(trunks)
            # indexes = py_util.normal_filter(
            #    np.vstack(dcluster),
            #    return_indexes=True,
            #    search_radius=0.2,
            # )
            # fcluster = dcluster[indexes]
            # labels = labels[indexes]
            # cluster = fcluster

        bench_dict["get cluster"].step("filter")
        bench_dict["get cluster"].gstop()
        if use_trunks:
            return cluster, labels, centers, trunks
        else:
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
            # get single object
            temp_index = np.where(filterd_object_label == unique_object_label[j])
            temp_index_object_xyz = filterd_temp_xyz[temp_index[0], :]
            bench_dict["loader"].step("compute 1")
            down_points = filterd_temp_xyz[filterd_object_label.flatten() == unique_object_label[j]]
            bench_dict["loader"].step("compute down1")
            filter_idx = py_util.normal_filter(down_points, 0.1, 0.4, 0.1, return_indexes=True)
            filtered_points = down_points[filter_idx]
            bench_dict["loader"].step("filter")
            if len(filtered_points) < len(down_points) * 0.1:
                temp_object_center_xyz = np.mean(temp_index_object_xyz, 0)
            else:
                temp_object_center_xyz = py_util.trunk_center(filtered_points)
                if len(temp_object_center_xyz) == 0:
                    temp_object_center_xyz = np.mean(filtered_points, 0)
            bench_dict["loader"].step("trunk")
            temp_multi_objects_centers.append([temp_object_center_xyz, unique_object_label[j]])
            bench_dict["loader"].step("compute other")
        bench_dict["loader"].gstop()

        if return_cloud:
            if return_filtered:
                indexes = py_util.normal_filter(
                    filterd_temp_xyz,
                    return_indexes=True,
                    search_radius=0.1,
                    verticality_threshold=0.4,
                    curvature_threshold=0.1,
                )
                out_filterd_temp_xyz = filterd_temp_xyz[indexes]
                out_filterd_object_label = filterd_object_label[indexes]
            else:
                out_filterd_temp_xyz = filterd_temp_xyz
                out_filterd_object_label = filterd_object_label
            return temp_multi_objects_centers, out_filterd_temp_xyz, out_filterd_object_label

        return temp_multi_objects_centers
