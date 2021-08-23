import open3d as o3d
import numpy as np
from plyfile import PlyData, PlyElement
from collections import defaultdict
try:
    from open3d.ml.contrib import subsample
    use_ml = True
except ImportError:
    use_ml = False
from sklearn.model_selection import train_test_split


def save_cloud(dir, points, labels=None, instances=None):
    assert len(points.shape) == 2, "Not valid point_cloud"
    elems = points.shape[0]
    dtype_list = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if labels is None:
        label = np.array([])
    else:
        label = labels
        dtype_list.append(("class", "f4"))

    if instances is None:
        instance = np.array([])
    else:
        instance = instances
        dtype_list.append(("id", "f4"))

    array = np.vstack(
        [points.T, label.reshape(-1, elems), instance.reshape(-1, elems)]
    ).T

    vertex = np.empty(
        len(array),
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("class", "i4"), ("id", "f4")],
    )
    vertex["x"] = array[:, 0]
    vertex["y"] = array[:, 1]
    vertex["z"] = array[:, 2]
    if instances is not None:
        vertex["class"] = array[:, 3]
    if instances is not None:
        vertex["id"] = array[:, 4]
    el = PlyElement.describe(vertex, "vertex")

    with open(dir, "wb") as f:
        PlyData([el]).write(f)


def load_cloud(dir):
    with open(dir, "rb") as f:
        plydata = PlyData.read(f)

    point_cloud = np.vstack(
        [
            np.array(plydata.elements[0].data["x"]),
            np.array(plydata.elements[0].data["y"]),
            np.array(plydata.elements[0].data["z"]),
        ]
    ).T
    try:
        labels = np.array(plydata.elements[0].data["class"])
    except:
        labels = []
    try:
        instances = np.array(plydata.elements[0].data["id"])
    except:
        instances = []

    return point_cloud, labels, instances


def downsample(points, features=None, labels=None, grid_size=0.6):
    out_points = []
    out_features = []
    out_labels = []
    if points.dtype != np.float32:
        points = points.astype(np.float32)
    use_ml = False
    if use_ml:
        if (features is None) and (labels is None):
            out_points = subsample(points, sampleDl=grid_size)
        elif labels is None:
            out_points, out_features = subsample(
                points, features=features, sampleDl=grid_size
            )
        elif features is None:
            out_points, out_labels = subsample(points, classes=labels, sampleDl=grid_size)
        else:
            out_points, out_features, out_labels = subsample(
                points, features=features, classes=labels, sampleDl=grid_size
            )

        return out_points, out_features, out_labels
    else:
        if (features is None) :
            features = np.zeros(points.shape[0])
        if (labels is None):
            labels = np.zeros(points.shape[0])
        points = np.random.shuffle(points)
        features = np.random.shuffle(features)
        labels = np.random.shuffle(labels)

        idx = np.arange(points.shape[0])
        sub_idx = idx[:int(points.shape[0]/10)]
        out_points = points[:int(points.shape[0]/10)]
        out_features = features[:int(points.shape[0]/10)]
        out_labels = labels[:int(points.shape[0]/10)]
        if False:
            feature_label_zeros = np.vstack([features.flatten(), labels.flatten(), np.zeros_like(labels.flatten())]).T
            
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(feature_label_zeros)
            down = pcd.voxel_down_sample(grid_size)
            out_points = np.asarray(down.points)
            out_feature_label_zeros = np.asarray(down.colors)
            out_features = np.around(out_feature_label_zeros[:,0].reshape(-1,1))
            out_labels = np.around(out_feature_label_zeros[:,1])
        
        return out_points, out_features, out_labels

    


class data_loader:
    def __init__(self, onlyTrees=False, preprocess=False, train_split=False) -> None:
        self.data_loaded = False
        self.trees = None
        self.point_cloud = None
        self.labels = None
        self.instances = None
        self.non_trees = None
        self.onlyTrees = onlyTrees
        self.tree_label = None
        self.kd_tree_non_trees = None
        self.non_trees_pointcloud = None
        self.preprocess = preprocess
        self.train_split = train_split

    def load_data(self, dir):
        point_cloud, labels, instances = load_cloud(dir)
        assert((len(point_cloud)>0) &(len(labels)>0) & (len(instances)>0))
        self.point_cloud = point_cloud
        self.labels = labels
        self.instances = instances
        self.tree_label = True
        self.data_loaded = True

    def get_trees(self, unique=True):
        # Only process if pointcloud is available
        if self.point_cloud is not None:
            # get all tree points
            bool_idx = self.labels == self.tree_label
            tree_points = self.point_cloud[bool_idx]
            tree_instances = self.instances[bool_idx]

            # if unique get instances
            if unique:
                unique_trees = defaultdict(list)
                for i in np.unique(tree_instances):
                    unique_trees[i].append(tree_points[tree_instances == i])
                unique_trees = [
                    unique_trees[i][0]
                    if len(unique_trees[i]) == 1
                    else np.vstack(unique_trees[i])
                    for i in unique_trees.keys()
                ]
                self.trees = unique_trees
            else:
                self.trees = tree_points
            if self.train_split:
                indexes = np.arange(len(self.trees))
                self.train_trees, self.test_trees = train_test_split(indexes, test_size=0.2)
        return self.trees

    def get_non_trees(self, unique=True):
        if self.point_cloud is not None:
            bool_idx = self.labels != self.tree_label
            non_tree_points = self.point_cloud[bool_idx]
            non_tree_instances = self.instances[bool_idx].flatten()

            if unique:
                non_trees = defaultdict(list)
                for i in np.unique(non_tree_instances):
                    non_trees[i].append(non_tree_points[non_tree_instances == i])
                non_trees = [
                    non_trees[i][0]
                    if len(non_trees[i]) == 1
                    else np.vstack(non_trees[i])
                    for i in non_trees.keys()
                ]
                self.non_trees = non_trees
            else:
                self.non_trees = non_tree_points
        return self.non_trees

    def get_random_background(self, radius=20):
        if self.non_trees_pointcloud is None:
            if self.non_trees is not None:
                point_cloud = self.non_trees
            else:
                point_cloud = self.get_non_trees(unique=False)
            self.non_trees_pointcloud = o3d.geometry.PointCloud()
            self.non_trees_pointcloud.points = o3d.utility.Vector3dVector(point_cloud)
        point_cloud_len = len(self.non_trees_pointcloud.points)
        if self.kd_tree_non_trees is None:
            self.kd_tree_non_trees = o3d.geometry.KDTreeFlann(self.non_trees_pointcloud)
        [k, idx, _] = self.kd_tree_non_trees.search_radius_vector_3d(
            self.non_trees_pointcloud.points[np.random.choice(point_cloud_len)],
            radius,
        )
        smaller_background = np.asarray(self.non_trees_pointcloud.points)[idx[1:], :]
        # height = np.min(smaller_background[:,2])
        height = np.percentile(smaller_background[:, 2], 5)

        center = np.mean(smaller_background, axis=0)
        smaller_background = smaller_background - (center[0], center[1], height)

        return smaller_background

    def get_random_forground(self, train=False):
        if self.trees is not None:
            point_cloud = self.trees
        else:
            point_cloud = self.get_trees(unique=True)

        if self.train_split:
            if train:
                choice = np.random.choice(self.train_trees)
            else:
                choice = np.random.choice(self.test_trees)
            smaller_forground = point_cloud[choice]
            while len(smaller_forground) < 60:
                if train:
                    choice = np.random.choice(self.train_trees)
                else:
                    choice = np.random.choice(self.test_trees)
                smaller_forground = point_cloud[choice]            
        else:
            point_cloud_len = len(point_cloud)
            choice = np.random.choice(point_cloud_len)
            smaller_forground = point_cloud[choice]
            while len(smaller_forground) < 60:
                choice = np.random.choice(point_cloud_len)
                smaller_forground = point_cloud[choice]
        height = np.min(smaller_forground[:, 2])
        # height = np.percentile(smaller_forground[:,2],1)
        center = np.mean(smaller_forground, axis=0)
        smaller_forground = smaller_forground - (center[0], center[1], height)

        dists = np.linalg.norm(smaller_forground[:, :2], axis=1)
        smaller_forground = smaller_forground[dists < 20]
        if len(smaller_forground) < 60:
            print('simthing')

        return smaller_forground
