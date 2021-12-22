from numpy.random.mtrand import normal
from TreeToolML.data.data_gen_utils.custom_loaders import *
from TreeToolML.IndividualTreeExtraction.utils.py_util import normalize
from tqdm import tqdm
import os
import numpy as np


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


class all_data_loader:
    def __init__(
        self, onlyTrees=False, preprocess=False, default=True, train_split=False
    ) -> None:
        self.loader_list = {}
        self.tree_list = []
        self.non_tree_list = []
        self.default = default
        if default:
            # self.iqumulus = iqumulus_loader(onlyTrees, preprocess)
            self.tropical = tropical_loader(onlyTrees, preprocess)
            self.open = open_loader(onlyTrees, preprocess)
            self.paris = paris_loader(onlyTrees, preprocess)
            self.toronto = toronto_loader(onlyTrees, preprocess)
        else:
            # self.iqumulus = data_loader(onlyTrees, preprocess, train_split)
            self.tropical = data_loader(onlyTrees, preprocess, train_split)
            self.open = data_loader(onlyTrees, preprocess, train_split)
            self.paris = data_loader(onlyTrees, preprocess, train_split)
            self.toronto = data_loader(onlyTrees, preprocess, train_split)

        # self.loader_list["iqumulus"] = self.iqumulus
        self.loader_list["tropical"] = self.tropical
        self.loader_list["open"] = self.open
        self.loader_list["paris"] = self.paris
        self.loader_list["toronto"] = self.toronto

        # self.non_tree_list.append(self.iqumulus)
        self.non_tree_list.append(self.open)
        self.non_tree_list.append(self.paris)
        self.non_tree_list.append(self.toronto)
        self.tree_list.append(self.open)
        self.tree_list.append(self.paris)
        self.tree_list.append(self.tropical)

    def load_all(self, dir=None):
        if self.default:
            for key, value in tqdm(self.loader_list.items()):
                if key.find("semantic") == -1:
                    value.load_data()
                else:
                    path = key.split("_")[1]
                    value.load_data("datasets/Semantic3D/" + path + "*.txt")

        else:
            for key, value in tqdm(self.loader_list.items()):
                value.load_data(os.path.join(dir, key + ".ply"))

        print("opentree trees:", len(self.open.get_trees()))
        print("paris trees:", len(self.paris.get_trees()))
        print("tropical trees:", len(self.tropical.get_trees()))

    def get_random_background(self, radius=10):
        choice = np.random.choice(len(self.non_tree_list))
        return self.non_tree_list[choice].get_random_background(radius)

    def get_random_forground(self, train=False):
        choice = np.random.choice(len(self.tree_list))
        return self.tree_list[choice].get_random_forground(train)

    def get_tree_cluster(
        self,
        max_trees=4,
        train=False,
        translation_xy=4,
        translation_z=0.2,
        scale=0.2,
        xy_rotation=0,
        dist_between=3,
        do_normalize=False,
        zero_floor = True,
    ):
        number_of_trees = np.random.randint(1, max_trees + 1)
        scale = scale
        xyrotmin = -np.deg2rad(xy_rotation)
        xyrotmax = np.deg2rad(xy_rotation)
        cluster = []
        cluster_center = []
        for i in range(number_of_trees):
            tree = self.get_random_forground(train)

            if do_normalize:
                tree = normalize(tree)

            if zero_floor:
                tree = tree - np.multiply(np.min(tree,0),[0,0,1])

            # R = eulerAnglesToRotationMatrix([0,0,np.random.uniform(0,np.pi*2)])
            R = eulerAnglesToRotationMatrix(
                [
                    np.random.uniform(xyrotmin, xyrotmax),
                    np.random.uniform(xyrotmin, xyrotmax),
                    np.random.uniform(0, np.pi * 2),
                ]
            )
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

            scaler = 1 + scale * (np.random.rand() - 0.5)
            rt = np.eye(4)
            rt[:3, 3] = translation
            rt[:3, :3] = scaler * R
            htree = np.hstack([tree, np.ones_like(tree[:, 0:1])])
            new_tree = (rt @ htree.T).T[:,:3]
            cluster.append(new_tree)
            cluster_center.append(translation)

        labels = [n * np.ones_like(i[:, :1]) for n, i in enumerate(cluster)]
        return cluster, labels
