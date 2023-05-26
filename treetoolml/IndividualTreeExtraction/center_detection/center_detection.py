import numpy as np
import treetoolml.IndividualTreeExtraction.voxel_traversal.VoxelTraversalAlgorithm as VTA
import treetoolml.IndividualTreeExtraction.accessible_region.AccessibleRegionGrowing as ARG
from multiprocessing import Pool
from itertools import repeat
import matplotlib.pyplot as plt


def show_AR_RG(voxels1, voxels2):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ####accessible region
    ax.voxels(voxels2, facecolors='red', edgecolor='k', alpha=0.9)
    ####region growing results
    ax.voxels(voxels1, facecolors='green', edgecolor='k')
    plt.show()

def direction_vote_voxels(points, directions, voxel_size, num_voxel_xyz, min_xyz, return_start_points=False):
    # accumulate count of visited voxels, and accumulate which point traverses each voxel
    # setup
    numpoints = np.size(points, 0)
    output_voxel_direction_count = np.zeros(
        (int(num_voxel_xyz[0]), int(num_voxel_xyz[1]), int(num_voxel_xyz[2])), dtype=int
    )

    ######
    per_voxel_direction_start_points = [
        [
            [[] for _ in range(int(num_voxel_xyz[2]))]
            for _ in range(int(num_voxel_xyz[1]))
        ]
        for _ in range(int(num_voxel_xyz[0]))
    ]
    ####
    if True:
        for i in range(numpoints):
            # visit voxels based on directions
            visited_voxels = VTA.voxel_traversal(
                points[i, :], directions[i, :], min_xyz, num_voxel_xyz, voxel_size
            )
            try:
                for j in range(len(visited_voxels)):
                    output_voxel_direction_count[
                        int(visited_voxels[j][0]),
                        int(visited_voxels[j][1]),
                        int(visited_voxels[j][2]),
                    ] += 1
                    per_voxel_direction_start_points[int(visited_voxels[j][0])][
                        int(visited_voxels[j][1])
                    ][int(visited_voxels[j][2])].append(i)
            except:
                visited_voxels = VTA.voxel_traversal(
                points[i, :], directions[i, :], min_xyz, num_voxel_xyz, voxel_size
            )

    return output_voxel_direction_count, per_voxel_direction_start_points


def center_detection_xoy(
    voxel_direction_count, num_voxel_xyz, center_direction_count_th
):

    numVoxel_x = num_voxel_xyz[0]
    numVoxel_y = num_voxel_xyz[1]
    object_center_voxel_list = []

    for i in range(int(numVoxel_x - 2)):
        for j in range(int(numVoxel_y - 2)):
            temp_object_voxel_dir_count = voxel_direction_count[i + 1, j + 1]

            if temp_object_voxel_dir_count < center_direction_count_th:
                continue

            temp_neighbors = [
                voxel_direction_count[i, j],
                voxel_direction_count[i + 1, j],
                voxel_direction_count[i + 2, j],
                voxel_direction_count[i, j + 1],
                voxel_direction_count[i + 2, j + 1],
                voxel_direction_count[i, j + 2],
                voxel_direction_count[i + 1, j + 2],
                voxel_direction_count[i + 2, j + 2],
            ]
            max_neighbors = np.max(np.array(temp_neighbors))

            if temp_object_voxel_dir_count > max_neighbors:
                object_center_voxel_list.append([i + 1, j + 1])
    if len(object_center_voxel_list) > 0:
        return np.vstack(object_center_voxel_list)
    else:
        return []

def vis_xy_votes(output_voxel_direction_count_xoy, plt_pointcloud, delta_xyz, min_xyz):
    output_voxel_direction_count_xoy
    x_d,y_d = output_voxel_direction_count_xoy.shape
    x, y = np.meshgrid(np.arange(x_d),np.arange(y_d))
    x=x*delta_xyz + min_xyz[0]
    y=y*delta_xyz + min_xyz[1]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    dx=dy=np.ones_like(output_voxel_direction_count_xoy.flatten())/2
    dx=dx*delta_xyz
    dy=dy*delta_xyz
    ax.bar3d(y.flatten(), x.flatten(), np.zeros_like(output_voxel_direction_count_xoy.flatten()), dx,dy, output_voxel_direction_count_xoy.flatten(), zsort='average')

    ax.scatter3D(plt_pointcloud[:,0], plt_pointcloud[:,1], plt_pointcloud[:,2]*np.max(output_voxel_direction_count_xoy)+np.max(output_voxel_direction_count_xoy), c=[0,0,0], s=2)
    plt.show()
    print('f')

############################################################
def center_detection(data, voxel_size, angle_threshold, center_direction_count_th=20, generate=False):
    """detect the tree centers"""
    object_xyz_list = []
    xyz = data[:, :3]
    directions = data[:, 3:6]
    min_xyz = np.min(xyz, axis=0)
    max_xyz = np.max(xyz, axis=0) + 0.000001
    delta_xyz = max_xyz - min_xyz
    num_voxel_xyz = np.ceil(delta_xyz / voxel_size)

    #######################################################################
    ############################Center Detection###########################
    #######################################################################
    (
        output_voxel_direction_count,
        per_voxel_direction_start_points,
    ) = direction_vote_voxels(xyz, directions, voxel_size, num_voxel_xyz, min_xyz)
    #####centers in xoy plane
    output_voxel_direction_count_xoy = np.sum(output_voxel_direction_count, axis=2)
    object_centers_xoy = center_detection_xoy(
        output_voxel_direction_count_xoy, num_voxel_xyz[:2], center_direction_count_th
    )
    if generate and False:
        vis_xy_votes(output_voxel_direction_count_xoy, xyz, voxel_size, min_xyz)
    ####centers in z-axis
    for i in range(np.size(object_centers_xoy, 0)):
        temp_object_center_xoy = object_centers_xoy[i, :]
        ####
        temp_centre_xyz = np.array(
            [temp_object_center_xoy[0], temp_object_center_xoy[1]]
        )
        temp_centre_xyz = (
            temp_centre_xyz * voxel_size + min_xyz[:2]
        )  # + voxel_size / 2.0
        ####
        center_xbottom = temp_centre_xyz[0] - voxel_size / 2.0
        center_xup = temp_centre_xyz[0] + voxel_size / 2.0
        center_ybottom = temp_centre_xyz[1] - voxel_size / 2.0
        center_yup = temp_centre_xyz[1] + voxel_size / 2.0
        x_vaild_range = np.where(
            (xyz[:, 0] > center_xbottom) == (xyz[:, 0] < center_xup)
        )
        y_vaild_range = np.where(
            (xyz[:, 1] > center_ybottom) == (xyz[:, 1] < center_yup)
        )
        xy_intersection_index = list(
            set(x_vaild_range[0]).intersection(set(y_vaild_range[0]))
        )

        ####discard the fake centers
        if False:
            if len(xy_intersection_index) == 0:
                continue
        #####
        output_voxel_direction_count_z = output_voxel_direction_count[
            temp_object_center_xoy[0], temp_object_center_xoy[1], :
        ]
        temp_index = np.where(
            output_voxel_direction_count_z == np.max(output_voxel_direction_count_z)
        )
        object_xyz_list.append(
            [temp_object_center_xoy[0], temp_object_center_xoy[1], temp_index[0][0]]
        )


    if len(object_xyz_list) > 0:
        object_xyz_list = np.vstack(object_xyz_list)
        object_xyz_list = object_xyz_list * voxel_size + min_xyz  # + voxel_size / 2.0
        print("Num of Tree Centers: %d" % int(np.size(object_xyz_list, 0)))
        ####### further refine detected centers using intersection directions
    ####### Note that the following steps have not been discussed in our paper #############
    ####### If higher efficiency is required, these steps can be discarded ###############
        objectVoxelMask_list = []
        sep_points_list = []
        for i in range(np.size(object_xyz_list, 0)):

            center_xyz = object_xyz_list[i, :]
            sep_points, _, objectVoxelMask = individual_tree_separation(
                xyz,
                directions,
                center_xyz,
                voxel_size,
                min_xyz,
                num_voxel_xyz,
                angle_threshold,
            )
            sep_points_list.append(sep_points)

            objectVoxelMask_index = np.where(objectVoxelMask == True)
            if np.size(objectVoxelMask_index[0], 0) == 0:
                continue
            temp_objectvoxels = []
            for j in range(np.size(objectVoxelMask_index[0], 0)):
                temp_objectvoxel_index = [
                    objectVoxelMask_index[0][j],
                    objectVoxelMask_index[1][j],
                    objectVoxelMask_index[2][j],
                ]
                temp_objectvoxels.append(temp_objectvoxel_index)
            objectVoxelMask_list.append(temp_objectvoxels)

        #######
        final_object_center_index = []
        for i in range(len(objectVoxelMask_list)):
            #####
            temp_object_voxels = np.vstack(objectVoxelMask_list[i])
            #####copy array
            temp_all_object_voxels = objectVoxelMask_list[:]
            del temp_all_object_voxels[i]

            #######
            for j in range(len(temp_all_object_voxels)):

                temp_remain_object_voxels = np.vstack(temp_all_object_voxels[j])
                temp_intersection = np.array(
                    [
                        x
                        for x in set(tuple(x) for x in temp_object_voxels)
                        & set(tuple(x) for x in temp_remain_object_voxels)
                    ]
                )

                if np.size(temp_intersection, 0) > 0:
                    temp_object_voxels = set(
                        tuple(x) for x in temp_object_voxels
                    ).difference(set(tuple(x) for x in temp_intersection))
                    temp_object_voxels = np.array([list(x) for x in temp_object_voxels])

                    if np.size(temp_object_voxels, 0) == 0:
                        break  #
            if np.size(temp_object_voxels, 0) >= 3:
                final_object_center_index.append(i)

        object_xyz_list = object_xyz_list[final_object_center_index, :]
    else:
        object_xyz_list = np.array([[999, 999, 999]])
        sep_points_list = []
    
    return object_xyz_list, sep_points_list


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
    try:
        objcetMask, objcetMaskVoxelIndex = ARG.voxel_region_grow(accessible_region_voxels, seed_voxel)
    except:
        print('')

    ###########visualization
    objcetMask = np.array(objcetMask)
    objcetMask = objcetMask.astype(bool)
    if visulization:
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