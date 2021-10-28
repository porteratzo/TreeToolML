import numpy as np
import IndividualTreeExtraction.voxel_traversal.VoxelTraversalAlgorithm as VTA

def direction_vote_voxels(points, directions, voxel_size, num_voxel_xyz, min_xyz):
    # accumulate count of visited voxels, and accumulate which point traverses each voxel
    # setup
    numpints = np.size(points, 0)
    output_voxel_direction_count = np.zeros((int(num_voxel_xyz[0]), int(num_voxel_xyz[1]), int(num_voxel_xyz[2])), dtype=int)

    ######
    per_voxel_direction_start_points = [[[[] for _ in range(int(num_voxel_xyz[2]))] for _ in range(int(num_voxel_xyz[1]))] for _ in range(int(num_voxel_xyz[0]))]
    ####
    for i in range(numpints):
        # visit voxels based on directions
        visited_voxels = VTA.voxel_traversal(points[i, :], directions[i, :], min_xyz, num_voxel_xyz, voxel_size)
        for j in range(len(visited_voxels)):
            output_voxel_direction_count[int(visited_voxels[j][0]), int(visited_voxels[j][1]), int(visited_voxels[j][2])] += 1
            per_voxel_direction_start_points[int(visited_voxels[j][0])][int(visited_voxels[j][1])][int(visited_voxels[j][2])].append(i)

    return output_voxel_direction_count, per_voxel_direction_start_points


def center_detection_xoy(voxel_direction_count, num_voxel_xyz, center_direction_count_th):

    numVoxel_x = num_voxel_xyz[0]
    numVoxel_y = num_voxel_xyz[1]
    object_center_voxel_list = []

    for i in range(int(numVoxel_x - 2)):
        for j in range(int(numVoxel_y - 2)):
            temp_object_voxel_dir_count = voxel_direction_count[i + 1, j + 1]

            if temp_object_voxel_dir_count < center_direction_count_th:
                continue

            temp_neighbors = [voxel_direction_count[i, j], voxel_direction_count[i + 1, j],
                              voxel_direction_count[i + 2, j],
                              voxel_direction_count[i, j + 1], voxel_direction_count[i + 2, j + 1],
                              voxel_direction_count[i, j + 2], voxel_direction_count[i + 1, j + 2],
                              voxel_direction_count[i + 2, j + 2]]
            max_neighbors = np.max(np.array(temp_neighbors))

            if temp_object_voxel_dir_count > max_neighbors:
                object_center_voxel_list.append([i + 1, j + 1])
    if len(object_center_voxel_list) > 0:
        return np.vstack(object_center_voxel_list)
    else:
        return []


############################################################
def center_detection(data, voxel_size, angle_threshold, center_direction_count_th=20):
    '''detect the tree centers'''

    object_xyz_list = []
    xyz = data[:, :3]
    directions = data[:, 3:]
    min_xyz = np.min(xyz, axis=0)
    max_xyz = np.max(xyz, axis=0)
    delta_xyz = max_xyz - min_xyz
    num_voxel_xyz = np.ceil(delta_xyz / voxel_size)

    #######################################################################
    ############################Center Detection###########################
    #######################################################################
    output_voxel_direction_count, per_voxel_direction_start_points = direction_vote_voxels(xyz,
                                                                                           directions,
                                                                                           voxel_size,
                                                                                           num_voxel_xyz,
                                                                                           min_xyz)
    #####centers in xoy plane
    output_voxel_direction_count_xoy = np.sum(output_voxel_direction_count, axis=2)
    object_centers_xoy = center_detection_xoy(output_voxel_direction_count_xoy,
                                                     num_voxel_xyz[:2],
                                                     center_direction_count_th)

    ####centers in z-axis
    for i in range(np.size(object_centers_xoy, 0)):
        temp_object_center_xoy = object_centers_xoy[i, :]
        ####
        temp_centre_xyz = np.array([temp_object_center_xoy[0], temp_object_center_xoy[1]])
        temp_centre_xyz = temp_centre_xyz * voxel_size + min_xyz[:2] # + voxel_size / 2.0
        ####
        center_xbottom = temp_centre_xyz[0] - voxel_size / 2.0
        center_xup = temp_centre_xyz[0] + voxel_size / 2.0
        center_ybottom = temp_centre_xyz[1] - voxel_size / 2.0
        center_yup = temp_centre_xyz[1] + voxel_size / 2.0
        x_vaild_range = np.where((xyz[:, 0] > center_xbottom) == (xyz[:, 0] < center_xup))
        y_vaild_range = np.where((xyz[:, 1] > center_ybottom) == (xyz[:, 1] < center_yup))
        xy_intersection_index = list(set(x_vaild_range[0]).intersection(set(y_vaild_range[0])))

        ####discard the fake centers
        if len(xy_intersection_index) == 0:
            continue
        #####
        output_voxel_direction_count_z = output_voxel_direction_count[temp_object_center_xoy[0], temp_object_center_xoy[1], :]
        temp_index = np.where(output_voxel_direction_count_z == np.max(output_voxel_direction_count_z))
        object_xyz_list.append([temp_object_center_xoy[0], temp_object_center_xoy[1], temp_index[0][0]])

    if len(object_xyz_list)>0:
        object_xyz_list = np.vstack(object_xyz_list)
        object_xyz_list = object_xyz_list * voxel_size + min_xyz # + voxel_size / 2.0
        print('Num of Tree Centers: %d'%int(np.size(object_xyz_list, 0)))
    else:
        object_xyz_list=np.array([[999,999,999]])
    ####### further refine detected centers using intersection directions
    ####### Note that the following steps have not been discussed in our paper #############
    ####### If higher efficiency is required, these steps can be discarded ###############
    if False:
        objectVoxelMask_list = []
        for i in range(np.size(object_xyz_list, 0)):

            center_xyz = object_xyz_list[i, :]
            _, _, objectVoxelMask = individual_tree_separation(xyz,
                                                            directions,
                                                            center_xyz,
                                                            voxel_size,
                                                            min_xyz,
                                                            num_voxel_xyz,
                                                            angle_threshold)

            objectVoxelMask_index = np.where(objectVoxelMask == True)
            if np.size(objectVoxelMask_index[0], 0) == 0:
                continue
            temp_objectvoxels = []
            for j in range(np.size(objectVoxelMask_index[0], 0)):
                temp_objectvoxel_index = [objectVoxelMask_index[0][j], objectVoxelMask_index[1][j], objectVoxelMask_index[2][j]]
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
                temp_intersection = np.array([x for x in set(tuple(x) for x in temp_object_voxels) & set(tuple(x) for x in temp_remain_object_voxels)])

                if np.size(temp_intersection, 0) > 0:
                    temp_object_voxels = set(tuple(x) for x in temp_object_voxels).difference(set(tuple(x) for x in temp_intersection))
                    temp_object_voxels = np.array([list(x) for x in temp_object_voxels])

                    if np.size(temp_object_voxels, 0) == 0:
                        break        #
            if np.size(temp_object_voxels, 0) >= 3:
                final_object_center_index.append(i)

        object_xyz_list = object_xyz_list[final_object_center_index, :]
    
    return object_xyz_list