import numpy as np
import random
import math
import os
import sys
import pclpy
import open3d as o3d

sys.path.append("/home/omar/Documents/Mine/Git/TreeTool")
import TreeTool.seg_tree as seg_tree


def downsample(point_cloud, leaf_size=0.005, return_idx=False):
    if return_idx:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        rest = pcd.voxel_down_sample_and_trace(
            voxel_size=leaf_size,
            min_bound=np.array([-10, -10, -10]),
            max_bound=np.array([10, 10, 10]),
        )
        indexes = rest[1][rest[1] != -1]
        return indexes
    else:
        return seg_tree.voxelize(point_cloud, leaf_size)


def combine_IOU(vis_dict):
    vis_dict_ = {n: i for n, i in enumerate(vis_dict)}
    discard_list = []
    for key1, _points in vis_dict_.items():
        if key1 in discard_list:
            continue

        for key2, _points2 in vis_dict_.items():
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
            if iou > 0.8:
                vis_dict_[key1] = np.vstack([vis_dict_[key1], vis_dict_[key2]])
                discard_list.append(key2)
    return vis_dict_


def data_preprocess(temp_point_set):
    temp_xyz = temp_point_set[:, :3]
    temp_xyz = normalize(temp_xyz)
    return temp_xyz


def load_data(path):
    try:
        return np.load(path)
    except:
        return np.loadtxt(path)


def get_data_set(data_path):
    files_set = os.listdir(data_path)
    random.shuffle(files_set)
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
    min_xyz = np.min(sample_xyz, axis=0)
    max_xyz = np.max(sample_xyz, axis=0)
    deta_central_xyz = (max_xyz - min_xyz) / 2.0
    central_xyz = deta_central_xyz + min_xyz
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
    non_ground_normals = seg_tree.extract_normals(subject_cloud, search_radius)

    # remove Nan points
    non_nan_mask = np.bitwise_not(np.isnan(non_ground_normals.normals[:, 0]))
    non_nan_cloud = subject_cloud[non_nan_mask]
    non_nan_normals = non_ground_normals.normals[non_nan_mask]
    non_nan_curvature = non_ground_normals.curvature[non_nan_mask]

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


def trunk_center(filtered_points):
    half_height = (np.max(filtered_points[:, 2]) - np.min(filtered_points[:, 2])) / 2
    filtered_points = filtered_points[filtered_points[:, 2] < half_height]
    indices, model = seg_tree.segment_normals(
        filtered_points,
        search_radius=0.05,
        model=pclpy.pcl.sample_consensus.SACMODEL_CYLINDER,
        method=pclpy.pcl.sample_consensus.SAC_RANSAC,
        normalweight=0.01,
        miter=1000,
        distance=0.01,
        rlim=[0, 0.2],
    )
    temp_object_center_xyz = np.mean(filtered_points[indices], 0)
    if len(filtered_points[indices]) < len(filtered_points) * 0.1:
        return []
    return temp_object_center_xyz
