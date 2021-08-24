import open3d as o3d
import numpy as np
from plyfile import PlyData
import glob
import pandas as pd
import os
from data_gen_utils.dataloaders import data_loader, downsample


grid_size = 0.08
class iqumulus_loader(data_loader):
    def __init__(self, onlyTrees=False, preprocess=False) -> None:
        super().__init__(onlyTrees=onlyTrees, preprocess=preprocess)

    def load_data(self, dir_path="datasets/IQmulus/Cassette_GT.ply"):
        if not self.data_loaded:
            with open(dir_path, "rb") as f:
                plydata = PlyData.read(f)

            point_cloud = np.vstack(
                [
                    np.array(plydata.elements[0].data["x"]),
                    np.array(plydata.elements[0].data["y"]),
                    np.array(plydata.elements[0].data["z"]),
                ]
            ).T
            label = np.array(plydata.elements[0].data["class"]).astype(np.int32)
            instance = np.array(plydata.elements[0].data["id"]).astype(np.float32).reshape(-1, 1)

            if self.preprocess:
                sub_points, sub_feat, sub_labels = downsample(
                        point_cloud, features=instance, labels=label, grid_size=grid_size
                    )
                self.point_cloud = sub_points
                self.instances = sub_feat
                self.labels = sub_labels
            else:
                self.point_cloud = point_cloud
                self.instances = instance
                self.labels = label
            self.data_loaded = True
            self.tree_label = 304020000.0

        if self.onlyTrees:
            if self.trees is None:
                self.get_trees()
            self.point_cloud = None


class tropical_loader(data_loader):
    def __init__(self, onlyTrees=False, preprocess=False) -> None:
        super().__init__(onlyTrees=onlyTrees, preprocess=preprocess)

    def load_data(self, dir_path="datasets/lucid/1_LidarTreePoinCloudData/4_LidarTreePoinCloudData/*"):
        if not self.data_loaded:
            tree_dirs = glob.glob(dir_path)
            trees = []
            for i in tree_dirs:
                point_cloud = pd.read_csv(i, header=None, delimiter=" ").to_numpy().astype(np.float32)

                if self.preprocess:
                    sub_points, sub_feat, sub_labels = downsample(
                            point_cloud, grid_size=grid_size
                        )
                    trees.append(sub_points)
                else:
                    trees.append(point_cloud)
            self.point_cloud = np.vstack(trees)
            self.labels = np.ones_like(self.point_cloud[:, 0])
            self.instances = np.concatenate(
                [np.ones_like(i[:, 0]) * n for n, i in enumerate(trees)]
            )
            

            self.data_loaded = True
            self.tree_label = 1.0

        if self.onlyTrees:
            if self.trees is None:
                self.get_trees()
            self.point_cloud = None


class open_loader(data_loader):
    def __init__(self, onlyTrees=False, preprocess=False) -> None:
        super().__init__(onlyTrees=onlyTrees, preprocess=preprocess)

    def load_data(self, dir_path="datasets/open_tree/*.pcd"):
        if not self.data_loaded:
            trees_dir = glob.glob(dir_path)
            labels = []

            trees = []
            for i in trees_dir:
                label = os.path.basename(i).split("_")[0]
                if label not in ['tree', 'Octree-terrain.pcd']:
                    continue
                labels.append(label)
                point_cloud = np.asarray(o3d.io.read_point_cloud(i).points).astype(np.float32)
                if self.preprocess:
                    sub_points, sub_feat, sub_labels = downsample(
                            point_cloud, grid_size=grid_size
                        )
                    trees.append(sub_points)
                else:
                    trees.append(point_cloud)
            self.point_cloud = np.vstack([i for i in trees])
            unique_labels = set(labels)
            label_dict = {i: n for n, i in enumerate(sorted(unique_labels))}
            self.labels = np.concatenate(
                [
                    np.ones_like(i[:, 0]) * label_dict[l]
                    for n, (i, l) in enumerate(zip(trees, labels))
                ]
            )
            self.instances = np.concatenate(
                [
                    np.ones_like(i[:, 0]) * n
                    for n, i in enumerate(trees)
                ]
            )

            self.data_loaded = True
            self.tree_label = label_dict["tree"]

        if self.onlyTrees:
            if self.trees is None:
                self.get_trees()
            self.point_cloud = None


class toronto_loader(data_loader):
    def __init__(self, onlyTrees=False, preprocess=False) -> None:
        super().__init__(onlyTrees=onlyTrees, preprocess=preprocess)

    def load_data(self, dir_path="datasets/Toronto3D/*.ply"):
        if not self.data_loaded:
            point_clouds = []
            labels = []
            for i in glob.glob(dir_path):
                with open(i, "rb") as f:
                    plydata = PlyData.read(f)

                point_cloud = np.vstack(
                    [
                        np.array(plydata.elements[0].data["x"]),
                        np.array(plydata.elements[0].data["y"]),
                        np.array(plydata.elements[0].data["z"]),
                    ]
                ).T
                label = np.array(
                    plydata.elements[0].data["scalar_Label"], dtype=np.int32
                ).reshape((-1))
                sub_points, sub_feat, sub_labels = downsample(
                    point_cloud, labels=label, grid_size=grid_size
                )

                labels.append(sub_labels)
                point_clouds.append(sub_points)

            self.labels = np.concatenate(labels).reshape(-1, 1)
            self.instances = np.concatenate(labels).reshape(-1)
            self.point_cloud = np.vstack(point_clouds)
            self.tree_label = 3
            self.toronto_loaded = True
        if self.onlyTrees:
            if self.trees is None:
                self.get_trees()
            self.point_cloud = None


class paris_loader(data_loader):
    def __init__(self, onlyTrees=False, preprocess=False) -> None:
        super().__init__(onlyTrees=onlyTrees, preprocess=preprocess)

    def load_data(self, dir_path="datasets/Paris_Lille3D/training_50_classes/*.ply"):
        if not self.data_loaded:
            point_clouds = []
            labels = []
            instances = []
            smallest_instance = 0
            for i in glob.glob(dir_path):
                print(os.path.basename(i))
                with open(i, "rb") as f:
                    plydata = PlyData.read(f)

                point_cloud = np.vstack(
                    [
                        np.array(plydata.elements[0].data["x"]),
                        np.array(plydata.elements[0].data["y"]),
                        np.array(plydata.elements[0].data["z"]),
                    ]
                ).T

                instance = np.array(plydata.elements[0].data["label"]).reshape(-1, 1)
                label = np.array(
                    plydata.elements[0].data["class"], dtype=np.int32
                ).reshape((-1))
                sub_points, sub_feat, sub_labels = downsample(
                    point_cloud, features=instance, labels=label, grid_size=grid_size
                )
                
                instances.append(sub_feat)
                labels.append(sub_labels)
                point_clouds.append(sub_points)

            self.labels = np.concatenate(labels)
            self.instances = np.concatenate(instances).reshape(
                -1,
            )
            self.point_cloud = np.vstack(point_clouds)
            self.data_loaded = True
            self.tree_label = 304020000.0

        if self.onlyTrees:
            if self.trees is None:
                self.get_trees()
            self.point_cloud = None


class semantic3D(data_loader):
    def __init__(self, onlyTrees=False, preprocess=False) -> None:
        super().__init__(onlyTrees=onlyTrees, preprocess=preprocess)

    def load_data(self, dir_path="datasets/Semantic3D/*.txt"):
        if not self.data_loaded:
            point_clouds = []
            labels = []
            instances = []
            smallest_instance = 0
            self.tree_label = 3.0
            for i in glob.glob(dir_path):
                try:
                    print(os.path.basename(i))
                    pc = pd.read_csv(i,
                            header=None,
                            delim_whitespace=True,
                            dtype=np.float32).values

                    points = pc[:, 0:3]
                    feat = pc[:, [4, 5, 6]]

                    points = np.array(points, dtype=np.float32)
                    feat = np.array(feat, dtype=np.float32)

                    labels_array = pd.read_csv(i.replace(".txt", ".labels"),
                                        header=None,
                                        delim_whitespace=True,
                                        dtype=np.int32).values
                    labels_array = np.array(labels_array, dtype=np.int32).reshape((-1,))

                    sub_points, sub_feat, sub_labels = downsample(
                        points, features=feat, labels=labels_array, grid_size=grid_size
                    )
                    if np.isin(self.tree_label, sub_labels):
                        print(np.unique(sub_feat))
                        smallest_instance = np.max(sub_feat)
                        instances.append(sub_feat)
                        labels.append(sub_labels)
                        point_clouds.append(sub_points)
                except:
                    continue

            self.labels = np.concatenate(labels)
            self.instances = np.concatenate(instances).reshape(
                -1,
            )
            self.point_cloud = np.vstack(point_clouds)
            self.data_loaded = True
            

        if self.onlyTrees:
            if self.trees is None:
                self.get_trees()
            self.point_cloud = None
