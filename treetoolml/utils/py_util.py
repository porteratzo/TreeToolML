import numpy as np
import random
import math
import os
# import pclpy
import open3d as o3d
import treetool.seg_tree as seg_tree

EPS = 1e-8

def downsample(point_cloud, leaf_size=0.005, return_idx=False, min_bound=np.array([-100, -100, -100]), max_bound=np.array([100, 100, 100])):
    if return_idx:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        _, rest, _ = pcd.voxel_down_sample_and_trace(
            voxel_size=leaf_size,
            min_bound=min_bound,
            max_bound=max_bound,
        )
        return rest[rest != -1]
    else:
        return seg_tree.voxelize(point_cloud, leaf_size)


def outliers(points, min_n=6, radius=0.4, organized=True):
    _points = seg_tree.radius_outlier_removal(points, min_n, radius, organized)
    return _points[~np.all(np.isnan(_points), axis=1)]


def combine_IOU(vis_dict):
    vis_dict_ = {n: i for n, i in enumerate(vis_dict)}
    discard_list = []
    ious = []
    for key1, _points in list(vis_dict_.items()):
        if key1 in discard_list:
            continue

        for key2, _points2 in list(vis_dict_.items()):
            if key1 == key2:
                continue

            if key2 in discard_list:
                continue

            min_xyz1 = np.min(_points, axis=0)
            max_xyz1 = np.max(_points, axis=0)

            min_xyz2 = np.min(_points2, axis=0)
            max_xyz2 = np.max(_points2, axis=0)

            box1 = [min_xyz1[0], min_xyz1[1], max_xyz1[0], max_xyz1[1]]
            box2 = [min_xyz2[0], min_xyz2[1], max_xyz2[0], max_xyz2[1]]
            iou = bb_intersection_over_union(box1, box2)
            ious.append(iou)
            if iou > 0.2:
                vis_dict_[key1] = np.vstack([vis_dict_[key1], vis_dict_[key2]])
                vis_dict_.pop(key2)
                discard_list.append(key2)
    return list(vis_dict_.values())


def data_preprocess(temp_point_set):
    temp_xyz = temp_point_set[:, :3]
    temp_xyz = normalize_2(temp_xyz)
    return temp_xyz


def load_data(path):
    try:
        return np.load(path, allow_pickle=True)
    except:
        return np.loadtxt(path)


def get_data_set(data_path):
    files_set = os.listdir(data_path)
    files_set = sorted(files_set, key=lambda x: int(x.split(".")[0]))
    # random.shuffle(files_set)
    return files_set


def get_train_val_set(trainingdata_path, val_rate=0.20):
    train_set = []
    val_set = []
    all_train_set = os.listdir(trainingdata_path)
    random.shuffle(all_train_set)
    total_num = len(all_train_set)
    val_num = int(val_rate * total_num)
    for j in range(len(all_train_set)):
        if j < val_num:
            val_set.append(all_train_set[j])
        else:
            train_set.append(all_train_set[j])
    return train_set, val_set


def normalize_2(sample_xyz):
    #min height at 0
    centerd_tree = sample_xyz - np.multiply(np.min(sample_xyz, 0), [0, 0, 1])
    #center xy
    centerd_tree = centerd_tree - np.multiply(np.mean(centerd_tree, axis=0), [1, 1, 0])
    # normalize into unit sphere
    centerd_tree /= max(np.max(np.linalg.norm(centerd_tree, axis=1)), EPS)
    return centerd_tree


def normalize_2_center_return_scale(sample_xyz, center):
    floor = np.multiply(np.min(sample_xyz, 0), [0, 0, 1])
    centered_tree = sample_xyz - floor
    xy_center = np.multiply(np.mean(centered_tree, axis=0), [1, 1, 0])
    centered_tree = centered_tree - xy_center

    centered_center = center - floor
    centered_center = centered_center - xy_center
    # normalize into unit sphere
    return_tree = centered_tree / np.max(np.linalg.norm(centered_tree, axis=1))
    return_center = centered_center / np.max(np.linalg.norm(centered_tree, axis=1))
    return return_tree, return_center, np.max(np.linalg.norm(centered_tree, axis=1))


def normalize(sample_xyz):
    min_xyz = np.min(sample_xyz, axis=0)
    max_xyz = np.max(sample_xyz, axis=0)
    deta_central_xyz = (max_xyz - min_xyz) / 2.0
    central_xyz = deta_central_xyz + min_xyz
    n_data = sample_xyz - central_xyz
    # normalize into unit sphere
    n_data /= np.max(np.linalg.norm(n_data, axis=1))
    return n_data


def compute_object_center(sample_xyz):
    min_xyz = np.min(sample_xyz, axis=0)
    max_xyz = np.max(sample_xyz, axis=0)
    deta_central_xyz = (max_xyz - min_xyz) / 2.0
    central_xyz = deta_central_xyz + min_xyz
    return central_xyz


def jitter_point_cloud(sample_xyz, Jitter_argument, sigma=0.001, clip=0.05):
    if np.random.random() < Jitter_argument:
        N, C = sample_xyz.shape
        assert clip > 0
        jittered_data = np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
        sample_xyz += jittered_data
    return sample_xyz


def shuffle_data(data):
    idx = np.arange(np.size(data, 0))
    np.random.shuffle(idx)
    return data[idx, ...]


def ratation(sample_xyz, Rotation_argument):
    if np.random.random() < Rotation_argument:
        ###
        rot = random.uniform(0, 2 * math.pi)
        rotation_matrix = [
            [math.cos(rot), math.sin(rot), 0],
            [-math.sin(rot), math.cos(rot), 0],
            [0, 0, 1],
        ]
        sample_xyz = np.dot(sample_xyz, rotation_matrix)
    return sample_xyz


def ratation_angle(sample_xyz, angel):
    rot = angel / 180.0
    rotation_matrix = [
        [math.cos(rot), math.sin(rot), 0],
        [-math.sin(rot), math.cos(rot), 0],
        [0, 0, 1],
    ]
    sample_xyz = np.dot(sample_xyz, rotation_matrix)
    return sample_xyz


def transfer_xy(sample_xyz, x_d, y_d):
    temp_ones = np.ones([np.size(sample_xyz, 0), 1])
    sample_xyz = np.concatenate([sample_xyz, temp_ones], axis=-1)

    transfer_matrix = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [x_d, y_d, 0, 1]]
    sample_xyz = np.dot(sample_xyz, transfer_matrix)
    return sample_xyz[:, :3]


def farthest_point_sample(xyz, npoint):
    N, _ = xyz.shape
    centroids = []
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids.append(farthest)
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = int(np.where(distance == np.max(distance))[0][0])
    return centroids


def makesphere(centroid=[0, 0, 0], radius=1, dense=90):
    n = np.arange(0, 360, int(360 / dense))
    n = np.deg2rad(n)
    x, y = np.meshgrid(n, n)
    x = x.flatten()
    y = y.flatten()
    sphere = np.vstack(
        [
            centroid[0] + np.sin(x) * np.cos(y) * radius,
            centroid[1] + np.sin(x) * np.sin(y) * radius,
            centroid[2] + np.cos(x) * radius,
        ]
    ).T
    return sphere


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def get_center_scale(sample_xyz):
    central_xyz = np.multiply(np.min(sample_xyz, 0), [0, 0, 1]) + np.multiply(np.mean(sample_xyz, axis=0), [1, 1, 0])
    n_data = sample_xyz - central_xyz
    # normalize into unit sphere
    scale = np.max(np.linalg.norm(n_data, axis=1))
    return central_xyz, scale


def normal_filter(
    subject_cloud,
    search_radius=0.05,
    verticality_threshold=0.3,
    curvature_threshold=0.3,
    return_indexes=False,
):

    non_ground_normals, non_ground_curvature = seg_tree.extract_normals(subject_cloud, search_radius)

    # remove Nan points
    non_nan_mask = np.bitwise_not(np.isnan(non_ground_normals[:, 0]))
    non_nan_cloud = subject_cloud[non_nan_mask]
    non_nan_normals = non_ground_normals[non_nan_mask]
    non_nan_curvature = non_ground_curvature[non_nan_mask]


    # get mask by filtering verticality and curvature
    verticality = np.dot(non_nan_normals, [[0], [0], [1]])
    verticality_mask = (verticality < verticality_threshold) & (
        -verticality_threshold < verticality
    )
    curvature_mask = non_nan_curvature < curvature_threshold
    verticality_curvature_mask = verticality_mask.ravel() & curvature_mask.ravel()

    only_horizontal_points = non_nan_cloud[verticality_curvature_mask]



    if return_indexes:
        out_index = non_nan_mask
        out_index[non_nan_mask] = verticality_curvature_mask
        return out_index
    else:
        return only_horizontal_points


def seg_normals(
    filtered_points,
    miter=1000,
    distance=0.01,
    rlim=[0, 0.2],
):
    indices, model = seg_tree.fit_cylinder_ransac(
        filtered_points,
        max_iterations=miter,
        distance_threshold=distance,
        rlim=rlim,
    )
    return indices, model


def trunk_center(filtered_points):
    # half_height = (np.max(filtered_points[:, 2]) - np.min(filtered_points[:, 2])) / 2
    # filtered_points = filtered_points[filtered_points[:, 2] < half_height]
    indices, _ = seg_normals(filtered_points)
    temp_object_center_xyz = np.mean(filtered_points[indices], 0)
    if len(filtered_points[indices]) < len(filtered_points) * 0.1:
        return []
    return temp_object_center_xyz


def group_trees(
    filtered_points, tolerance=0.1, min_cluster_size=20, max_cluster_size=25000
):
    cluster_list = seg_tree.dbscan_cluster_extract(
        filtered_points,
        eps=tolerance,
        min_points=min_cluster_size,
    )
    return cluster_list


def get_tree_center(
    temp_xyz,
    downsample_leaf=0.01,
    search_radius=0.1,
    verticality_threshold=0.4,
    curvature_threshold=0.1,
    return_filtered=False,
    non_if_no_seg_center=False,
):

    down_points = downsample(temp_xyz, downsample_leaf, False)

    no_outlier_tree = outliers(down_points, 100, 10)

    filtered_points = normal_filter(
        no_outlier_tree,
        search_radius,
        verticality_threshold,
        curvature_threshold,
        return_indexes=False,
    )

    if len(filtered_points) > len(down_points) * 0.1:
        temp_object_center_xyz = trunk_center(filtered_points)
        if len(temp_object_center_xyz) == 0:
            if non_if_no_seg_center:
                temp_object_center_xyz = None
            else:
                temp_object_center_xyz = np.mean(filtered_points, 0)
    else:
        if non_if_no_seg_center:
            temp_object_center_xyz = None
        else:
            temp_object_center_xyz = np.mean(filtered_points, 0)



    if return_filtered:
        return (
            temp_object_center_xyz,
            filtered_points,
        )
    return temp_object_center_xyz
