#%%
import os
import argparse
import torch
from torch.utils.data import DataLoader
from treetoolml.config.config import combine_cfgs
from treetoolml.data.BatchSampleGenerator_torch import tree_dataset_cloud
from treetoolml.model.build_model import build_model
from scipy.optimize import linear_sum_assignment
from treetoolml.IndividualTreeExtraction_utils.center_detection.center_detection_vis import (
    center_detection, direction_vote_voxels, center_detection_xoy, center_z, segmentation
)
import numpy as np
from treetoolml.utils.torch_utils import device_configs, load_checkpoint, find_training_dir, make_xyz_mat


torch.backends.cudnn.benchmark = True

torch.manual_seed(123)


#%%
args = argparse.Namespace
args.device = 0
args.cfg = "configs/datasets/subconfigs/trunks.yaml"
args.gpu_number = 0
args.amp = 1
args.opts = []

cfg_path = args.cfg
cfg = combine_cfgs(cfg_path, args.opts)


model_dir = find_training_dir(cfg)
print(model_dir)
model = build_model(cfg)
load_checkpoint(model_dir, model)

device_configs(model, args)

test_path = cfg.VALIDATION.PATH
generator_val = tree_dataset_cloud(
    test_path,
    cfg.TRAIN.N_POINTS,
    normal_filter=True,
    distances=1,
    return_centers=True,
    center_collection_size=16,
    return_scale=True
)

device = "cuda"
test_loader = DataLoader(
    generator_val, cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=1
)
Nd = cfg.BENCHMARKING.XY_THRESHOLD
ARe = np.deg2rad(cfg.BENCHMARKING.ANGLE_THRESHOLD)
voxel_size = cfg.BENCHMARKING.VOXEL_SIZE
gen = iter(test_loader)
(
            batch_test_data,
            batch_direction_label_data,
            batch_object_label,
            batch_centers,
            batch_scales
) = next(gen)
batch_test_data = batch_test_data.half().to(device)
batch_direction_label_data = batch_direction_label_data.half().to(device)
###
with torch.no_grad():
    with torch.cuda.amp.autocast(enabled=args.amp):
        y = model(batch_test_data)
    
batch_centers_np = np.array(
                [i.numpy() for i in batch_centers]
            ).swapaxes(0, 1)
for n_center, dirs in enumerate(y):
    
    xyz_direction = make_xyz_mat(batch_test_data, n_center, dirs)
    
    scale = batch_scales[n_center]
    true_centers = batch_centers_np[n_center]
    true_centers = [
        i for i in true_centers if np.all(i != np.array([-1, -1, -1]))
    ]
    if cfg.DATA_PREPROCESSING.DISTANCE_FILTER == 0.0:
        _xyz_direction = xyz_direction
    else:
        _xyz_direction = xyz_direction[
            xyz_direction[:, 6] < cfg.DATA_PREPROCESSING.DISTANCE_FILTER
        ]

voxel_size = 0.04

# center detection init
object_xyz_list = []
data = _xyz_direction
angle_threshold = ARe
center_direction_count_th = Nd

xyz = data[:, :3]
directions = data[:, 3:6]
min_xyz = np.min(xyz, axis=0)
max_xyz = np.max(xyz, axis=0) + 0.000001
delta_xyz = max_xyz - min_xyz
num_voxel_xyz = np.ceil(delta_xyz / voxel_size)

(
    output_voxel_direction_count,
    per_voxel_direction_start_points,
) = direction_vote_voxels(
    xyz, directions, voxel_size, num_voxel_xyz, min_xyz
) 
output_voxel_direction_count_xoy = np.sum(output_voxel_direction_count, axis=2)
object_centers_xoy = center_detection_xoy(
        output_voxel_direction_count_xoy, num_voxel_xyz[:2], center_direction_count_th
    )
object_xyz_list = center_z(voxel_size, xyz, min_xyz, output_voxel_direction_count, object_centers_xoy)

object_xyz_list, sep_points_list = segmentation(voxel_size, angle_threshold, xyz, directions, min_xyz, num_voxel_xyz, object_xyz_list)


object_center_list, seppoints = center_detection(
    _xyz_direction, voxel_size, ARe, Nd
)
CostMat = np.ones([len(object_center_list), len(true_centers)])
for X, datatree in enumerate(object_center_list):
    for Y, foundtree in enumerate(true_centers):
        CostMat[X, Y] = np.linalg.norm(
            [datatree[0:2]*scale.numpy() - foundtree[0:2]*scale.numpy()]
        )

dataindex, foundindex = linear_sum_assignment(
    CostMat, maximize=False
)

