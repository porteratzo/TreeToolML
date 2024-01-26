# %%

import argparse
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from treetoolml.config.config import combine_cfgs
from treetoolml.IndividualTreeExtraction_utils.center_detection.center_detection import (
    center_detection,
)
from treetoolml.model.build_model import build_model
from treetoolml.utils.torch_utils import (
    device_configs,
    find_training_dir,
    load_checkpoint,
)
from treetoolml.utils.vis_utils import tree_vis_tool
from treetoolml.data.BatchSampleGenerator_torch import tree_dataset_cloud
from treetoolml.utils.benchmark_util import geometrics
from torch.utils.data import DataLoader
from treetoolml.utils.torch_utils import (
    device_configs,
    load_checkpoint,
    find_training_dir,
    make_xyz_mat,
)
import matplotlib.pyplot as plt
import sys
import os


# %%
# %%
args = argparse.Namespace
args.device = "cuda"
#args.cfg = "configs/experimentos_model/subconfigs/distance_out_loss.yaml"
args.cfg = "configs/datasets/subconfigs/center_filtered.yaml"
#args.cfg = "configs/datasets/subconfigs/trunks_new.yaml"
args.gpu_number = 0
args.amp = True
args.opts = []

cfg_path = args.cfg
cfg = combine_cfgs(cfg_path, args.opts)
use_amp = args.amp

Nd = cfg.BENCHMARKING.XY_THRESHOLD
ARe = np.deg2rad(cfg.BENCHMARKING.ANGLE_THRESHOLD)
voxel_size = cfg.BENCHMARKING.VOXEL_SIZE
sample_side_size = cfg.BENCHMARKING.WINDOW_STRIDE
overlap = cfg.BENCHMARKING.OVERLAP

model_dir = find_training_dir(cfg)
model = build_model(cfg)
load_checkpoint(model_dir, model)
device_configs(model, args)


# %%
#test_path = cfg.VALIDATION.PATH
test_path = 'datasets/custom_data/center_filtered/testing_data'
generator_val = tree_dataset_cloud(
    test_path,
    cfg.TRAIN.N_POINTS,
    normal_filter=True,
    distances=1,
    return_centers=True,
    center_collection_size=16,
    return_scale=True,
)

test_loader = DataLoader(generator_val, 1, shuffle=False, num_workers=0)
gen = iter(test_loader)
# %%
if next(model.parameters()).is_cuda:
    device = "cuda"
next(gen)
next(gen)
next(gen)
next(gen)
(
    batch_test_data,
    batch_direction_label_data,
    batch_object_label,
    batch_centers,
    batch_scales,
) = next(gen)


batch_test_data = batch_test_data.half().to(device)
batch_direction_label_data = batch_direction_label_data.half().to(device)
###
model.eval()
with torch.no_grad():
    with torch.cuda.amp.autocast(enabled=use_amp):
        y = model(batch_test_data)

batch_centers_np = np.array([i.numpy() for i in batch_centers]).swapaxes(0, 1)
dirs = y[0]
n_center = 0
xyz_direction = make_xyz_mat(batch_test_data, n_center, dirs)

scale = batch_scales[n_center]
true_centers = batch_centers_np[n_center]
true_centers = [i for i in true_centers if np.all(i != np.array([-1, -1, -1]))]

if cfg.DATA_PREPROCESSING.DISTANCE_FILTER == 0.0:
    _xyz_direction = xyz_direction
else:
    f_distances = xyz_direction[:, 6]
    if cfg.MODEL.CLASS_SIGMOID:
        _xyz_direction = xyz_direction[ f_distances >  cfg.DATA_PREPROCESSING.DISTANCE_FILTER ]
    else:
        _xyz_direction = xyz_direction[ (f_distances - np.min(f_distances))
            / (np.max(f_distances) - np.min(f_distances))
            < cfg.DATA_PREPROCESSING.DISTANCE_FILTER ]
#%%
#tree_vis_tool(xyz_direction[:, 0:3],true_centers, for_thesis=True, pointsize=5)
# %%
filtered_points = []
for n_c_, center in enumerate(true_centers):
    idx_s = batch_object_label[n_center] == n_c_
    xyz_ = xyz_direction[:, 0:3][idx_s]
    directions_ = xyz_direction[:, 3:6][idx_s]
    
    distances_ = np.linalg.norm(center - xyz_, axis=-1)
    out = [xyz_,]
    if False:
        distances_ = xyz_direction[:, 6][idx_s]
        filtered_points.append(xyz_[
                        (distances_ - np.min(distances_))
                        / (np.max(distances_) - np.min(distances_))
                        < 0.2
                    ])
        out.append(
                xyz_[
                    (distances_ - np.min(distances_))
                    / (np.max(distances_) - np.min(distances_))
                    < 0.2
                ]
                + [0.5, 0, 0])

    if True:
        tree_vis_tool(
            out,
            center,
            vectors=np.hstack([directions_ / 5, xyz_])[0:-1:10],
            vector_scale=0.2,
            axis=0,
            sphere_rad=0.01,
            for_thesis=True,
            pointsize=10,
        )
tree_vis_tool(filtered_points,true_centers, for_thesis=True, pointsize=5, sphere_rad=0.02)
# %%
output_dict = geometrics(
    batch_object_label,
    n_center,
    xyz_direction,
    true_centers,
    batch_direction_label_data,
)


voxel_size = 0.04
object_center_list, seppoints = center_detection(_xyz_direction, voxel_size, ARe, Nd)

CostMat = np.ones([len(object_center_list), len(true_centers)])
for X, datatree in enumerate(object_center_list):
    for Y, foundtree in enumerate(true_centers):
        CostMat[X, Y] = np.linalg.norm(
            [datatree[0:2] * scale.numpy() - foundtree[0:2] * scale.numpy()]
        )

dataindex, foundindex = linear_sum_assignment(CostMat, maximize=False)
# %%
