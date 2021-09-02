"""
Created on Mon July 11 18:50:39 2020

@author: Haifeng Luo
"""
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'voxel_traversal'))
sys.path.append(os.path.join(BASE_DIR, 'accessible_region'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append('Libraries')
sys.path.append('utils')
sys.path.append('..')
import py_util
#import AccessibleRegionGrowing as ARG
import PointwiseDirectionPrediction_torch as PDE_net
from BatchSampleGenerator_torch import tree_dataset
from torch.utils.data import DataLoader
from center_detection.center_detection import center_detection
from open3dvis import open3dpaint, o3d_pointSetClass
import BatchSampleGenerator as BSG
import Loss_torch
import torch


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

def show_AR_RG(voxels1, voxels2):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ####accessible region
    ax.voxels(voxels2, facecolors='red', edgecolor='k', alpha=0.9)
    ####region growing results
    ax.voxels(voxels1, facecolors='green', edgecolor='k')
    plt.show()

############################################################
def compute_object_center(sample_xyz):
    min_xyz = np.min(sample_xyz, axis=0)
    max_xyz = np.max(sample_xyz, axis=0)
    deta_central_xyz = (max_xyz - min_xyz) / 2.0
    central_xyz = deta_central_xyz + min_xyz
    return central_xyz

############################################################
def object_xoy_bounding(xyz, object_xyz, sphere_level, bounding_order=1):

    min_xy = np.min(object_xyz[:, :2], axis=0)
    max_xy = np.max(object_xyz[:, :2], axis=0)
    delta_xy = (max_xy - min_xy) / sphere_level
    min_xy += bounding_order * delta_xy
    max_xy -= bounding_order * delta_xy
    modify_object_index_x = np.where((xyz[:, 0] >= min_xy[0]) == (xyz[:, 0] < max_xy[0]))
    modify_object_index_y = np.where((xyz[:, 1] >= min_xy[1]) == (xyz[:, 1] < max_xy[1]))
    modify_object_index_xy = np.intersect1d(modify_object_index_x[0], modify_object_index_y[0])
    modify_object_index_xy = list(modify_object_index_xy)
    return modify_object_index_xy

############################################################
def individual_tree_separation(xyz, directions, center_xyz, voxel_size, min_xyz, num_voxel_xyz,
                               angle_threshold, visulization=False):

    #####generate accessible region
    accessible_region, accessible_index = ARG.detect_accessible_region(xyz, directions, center_xyz,
                                                                       voxel_size, angle_threshold)
    #####
    #####voxelize accessible region
    accessible_region_voxels, seed_voxel, valid_voxels, voxel2point_index_list = ARG.voxelization(accessible_region,
                                                                                              accessible_index,
                                                                                              voxel_size,
                                                                                              center_xyz,
                                                                                              min_xyz,
                                                                                              num_voxel_xyz)
    ###########
    output_voxels_v2 = np.array(accessible_region_voxels)
    output_voxels_v2 = output_voxels_v2.astype(bool)

    ####voxel-based region growing
    objcetMask, objcetMaskVoxelIndex = ARG.voxel_region_grow(accessible_region_voxels, seed_voxel)

    ###########visualization
    objcetMask = np.array(objcetMask)
    objcetMask = objcetMask.astype(bool)
    if visulization == True:
        show_AR_RG(objcetMask, output_voxels_v2)

    ######refine seed voxels
    index_voxel2point = [valid_voxels.index(tempMaskIndex) for tempMaskIndex in objcetMaskVoxelIndex]
    ######
    temp_object_xyz_index = []
    for temp_index_voxel2point in index_voxel2point:
        temp_object_xyz_index += voxel2point_index_list[temp_index_voxel2point]
    #####
    object_result = xyz[temp_object_xyz_index, :]
    return object_result, temp_object_xyz_index, objcetMask

def individual_tree_extraction(PDE_net_model_path, test_data_path, result_path, voxel_size, Nd, ARe):
    '''Individual Tree Extraction'''
    ####restore trained PDE-net
    PDE_net_model_path
    model = PDE_net.restore_trained_model(NUM_POINT, PDE_net_model_path).cuda()
    val_set = py_util.get_data_set(test_data_path)
    generator_val = BSG.minibatch_generator(test_data_path, 1, val_set, NUM_POINT)
    generator_val = tree_dataset(
            test_data_path, NUM_POINT
        )
    test_loader = DataLoader(generator_val, 1, shuffle=True, num_workers=0)
    ####
    file_list = os.listdir(test_data_path)
    for i in range(len(file_list[:10])):
        tree_index = 0
        filename, _ = os.path.splitext(file_list[i])
        print('Separating ' + filename + '...')
        #### data[x, y, z] original coordinates
        testdata, directions, labels = next(iter(test_loader))
        testdata, directions, labels = testdata.squeeze().numpy(), directions.squeeze().numpy(), labels.squeeze().numpy()
        ind_trees = [testdata[labels==i] for i in np.unique(labels)]
        object_centers = [py_util.compute_object_center(i) for i in ind_trees]
        ####normalized coordinates
        nor_testdata = torch.tensor(testdata, device='cuda').squeeze()
        ####Pointwise direction prediction
        xyz_direction = PDE_net.prediction(model, nor_testdata)
        ####tree center detection
        if False:
            xyz = xyz_direction[:,:3]
            angles = np.rad2deg(np.arctan2(directions[:,1], directions[:,0]))
            ps = o3d_pointSetClass(xyz, angles)
            open3dpaint([ps]+[makesphere(i, 0.1) for i in object_centers], pointsize=5, axis=True)

            angles = np.rad2deg(np.arctan2(directions[:,2], directions[:,0]))
            ps = o3d_pointSetClass(xyz, angles)
            open3dpaint([ps]+[makesphere(i, 0.1) for i in object_centers], pointsize=5, axis=True)

            angles = np.rad2deg(np.arctan2(directions[:,2], directions[:,1]))
            ps = o3d_pointSetClass(xyz, angles)
            open3dpaint([ps]+[makesphere(i, 0.1) for i in object_centers], pointsize=5, axis=True)

            object_center_list = center_detection(xyz_direction, voxel_size, ARe, Nd)
            loss_esd_ = Loss_torch.slack_based_direction_loss(torch.tensor(xyz_direction.T[np.newaxis,3:6,:].astype(np.float32)) ,torch.tensor(directions[np.newaxis,:].astype(np.float32)))
            print(loss_esd_)
            dirsxyz = xyz_direction[:,3:]
            
            angles = np.rad2deg(np.arctan2(dirsxyz[:,1], dirsxyz[:,0]))
            ps = o3d_pointSetClass(xyz, angles)
            open3dpaint([ps]+[makesphere(i, 0.1) for i in object_center_list], pointsize=5, axis=True)

            angles = np.rad2deg(np.arctan2(dirsxyz[:,2], dirsxyz[:,0]))
            ps = o3d_pointSetClass(xyz, angles)
            open3dpaint([ps]+[makesphere(i, 0.1) for i in object_center_list], pointsize=5, axis=True)

            angles = np.rad2deg(np.arctan2(dirsxyz[:,2], dirsxyz[:,1]))
            ps = o3d_pointSetClass(xyz, angles)
            open3dpaint([ps]+[makesphere(i, 0.1) for i in object_center_list], pointsize=5, axis=True)
        else:
            xyz = xyz_direction[:,:3]
            ps = o3d_pointSetClass(xyz)
            open3dpaint([ps]+[makesphere(i, 0.1) for i in object_centers], pointsize=5, axis=True)

            object_center_list = center_detection(xyz_direction, voxel_size, ARe, Nd)
            loss_esd_ = Loss_torch.slack_based_direction_loss(torch.tensor(xyz_direction.T[np.newaxis,3:6,:].astype(np.float32)) ,torch.tensor(directions[np.newaxis,:].astype(np.float32)))
            print(loss_esd_)
            open3dpaint([ps]+[makesphere(i, 0.1) for i in object_center_list], pointsize=5, axis=True)

        continue

        ####for single tree clusters
        if np.size(object_center_list, axis=0) <= 1:
            ####random colors
            num_pointIntree = np.size(xyz_direction, axis=0)
            color = np.random.randint(0, 255, size=3)
            ####assign tree labels
            temp_tree_label = np.ones([num_pointIntree, 1]) * tree_index
            color = np.ones([num_pointIntree, 3]) * color
            ######
            individualtree = np.concatenate([testdata[:, :3], color, temp_tree_label], axis=-1)
            np.savetxt(result_path + file_list[i], individualtree, fmt='%.4f')
            tree_index += 1
            continue

        ####for multi tree clusters
        extracted_object_list = []
        object_color_list = []
        temp_tree_id = 0
        for j in range(np.size(object_center_list, 0)):

            xyz = xyz_direction[:, :3]
            directions = xyz_direction[:, 3:]
            ####
            min_xyz = np.min(xyz, axis=0)
            max_xyz = np.max(xyz, axis=0)
            delta_xyz = max_xyz - min_xyz
            num_voxel_xyz = np.ceil(delta_xyz / voxel_size)
            ####
            center_xyz = object_center_list[j, :]
            ####use padding to fix the situation where the tree center voxel is empty
            center_xyz_padding = np.array([[center_xyz[0], center_xyz[1], center_xyz[2]],
                                           [center_xyz[0], center_xyz[1], center_xyz[2] - voxel_size],
                                           [center_xyz[0], center_xyz[1], center_xyz[2] + voxel_size]])
            directions_padding = np.array([[0.0, 0.0, 0.0],
                                           [0.0, 0.0, 1.0],
                                           [0.0, 0.0, -1.0]])
            center_direction_padding = np.concatenate([center_xyz_padding, directions_padding], axis=-1)

            xyz = np.concatenate([center_xyz_padding, xyz], axis=0)
            directions = np.concatenate([directions_padding, directions], axis=0)
            xyz_direction = np.concatenate([center_direction_padding, xyz_direction], axis=0)

            ####only for align the indexes
            testdata = np.concatenate([testdata[:3, :], testdata], axis=0)
            ####
            object_result, temp_object_xyz_index, _ = individual_tree_separation(xyz,
                                                                                 directions,
                                                                                 center_xyz,
                                                                                 voxel_size,
                                                                                 min_xyz,
                                                                                 num_voxel_xyz,
                                                                                 ARe,
                                                                                 visulization=False)
            ####refine the NULL growing results
            if np.size(object_result, 0) == 0: continue
            ###fix the discontinuity of the voxel in the vertical direction of tree centers
            modify_object_index_xy = object_xoy_bounding(xyz, object_result, 8, bounding_order=1)
            temp_object_xyz_index += modify_object_index_xy
            temp_object_xyz_index = list(set(temp_object_xyz_index))

            #####remove padding points
            real_object_xyz_index = [i for i in temp_object_xyz_index if i > 2]
            object_result = testdata[real_object_xyz_index, :3]
            ####generate random color for extracted individual tree points
            num_pointInObject = np.size(object_result, axis=0)
            color = np.random.randint(0, 255, size=3)
            object_color_list.append(color)
            ####assign a tree label for each individual tree
            temp_object_label = np.ones([num_pointInObject, 1]) * temp_tree_id
            color = np.ones([num_pointInObject, 3]) * color
            extracted_object_list.append(np.concatenate([object_result, color, temp_object_label], axis=-1))
            ####
            temp_tree_id += 1
            ####delete the extracted individual tree points
            testdata = np.delete(testdata, temp_object_xyz_index, axis=0)
            xyz_direction = np.delete(xyz_direction, temp_object_xyz_index, axis=0)

        ####use the nearest neighbor assignment to refine those points with large errors
        for k in range(np.size(xyz_direction, 0)):
            temp_remain_xyz_nor = xyz_direction[k, :3]
            temp_remain_xyz = testdata[k, :3]
            temp_distances = np.sqrt(np.sum(np.asarray(temp_remain_xyz_nor - object_center_list) ** 2, axis=1))
            nearestObjectCenter = np.where(temp_distances == np.min(temp_distances))
            color = object_color_list[int(nearestObjectCenter[0])]
            temp_remain_xyz_label = np.expand_dims(np.concatenate([temp_remain_xyz, color, nearestObjectCenter[0]], axis=-1), axis=0)
            extracted_object_list.append(temp_remain_xyz_label)
        ####output the final results
        np.savetxt(result_path + filename + '.txt', np.vstack(extracted_object_list), fmt='%.4f')


if __name__ == '__main__':

    NUM_POINT = 4096
    Nd = 80
    ARe = np.pi / 9.0
    voxel_size = 0.08
    #######
    PDE_net_model_path ='IndividualTreeExtraction/backbone_network/pre_trained_PDE_net/'
    test_data_path = 'datasets/custom_data/PDE/validating_data/'
    result_path = './result/'
    if not os.path.exists(result_path): os.mkdir(result_path)

    #######extract individual trees from tree clusters
    individual_tree_extraction(PDE_net_model_path, test_data_path, result_path, voxel_size, Nd, ARe)
