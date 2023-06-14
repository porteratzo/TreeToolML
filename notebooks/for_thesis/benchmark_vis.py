# %%

import sys
from typing import DefaultDict

import os

sys.path.append('/home/omar/Documents/mine/TreeTool')
sys.path.append('/home/omar/Documents/mine/TreeToolML')
sys.path.append("/home/omar/Documents/mine/IndividualTreeExtraction/voxel_region_grow/")
import argparse
from collections import defaultdict

from TreeTool.tree_tool import TreeTool
import numpy as np
import pclpy
import torch
from tqdm import tqdm
from treetoolml.config.config import combine_cfgs
from treetoolml.IndividualTreeExtraction.center_detection.center_detection import (
    center_detection,
)
from treetoolml.IndividualTreeExtraction.PointwiseDirectionPrediction_torch import (
    prediction,
)
from treetoolml.model.build_model import build_model
from treetoolml.utils.file_tracability import find_model_dir, get_checkpoint_file, get_model_dir
from treetoolml.utils.py_util import (
    combine_IOU,
    data_preprocess,
    get_center_scale,
    shuffle_data,
)
from porteratzolibs.visualization_o3d.open3dvis import open3dpaint
from treetoolml.benchmark.benchmark_utils import make_metrics_dict

# %%
args = argparse.Namespace
args.device = 0
args.cfg = "configs/datasets/trunks.yaml"
args.gpu_number = 0
args.amp = 1
args.opts = []

cfg_path = args.cfg
cfg = combine_cfgs(cfg_path, args.opts)

device = args.device
device = "cuda" if device == "gpu" else device
device = device if torch.cuda.is_available() else "cpu"

if device == "cuda":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_number)

Nd = cfg.BENCHMARKING.XY_THRESHOLD
ARe = np.deg2rad(cfg.BENCHMARKING.ANGLE_THRESHOLD)
voxel_size = cfg.BENCHMARKING.VOXEL_SIZE

sample_side_size = cfg.BENCHMARKING.WINDOW_STRIDE
overlap = cfg.BENCHMARKING.OVERLAP

model_name = cfg.TRAIN.MODEL_NAME
model_dir = os.path.join("results_training", model_name)
model_dir = find_model_dir(model_dir)
checkpoint_file = os.path.join(model_dir, "trained_model", "checkpoints")
checkpoint_path = get_checkpoint_file(checkpoint_file)
model = build_model(cfg).cuda()
if device == "cuda":
    model.cuda()
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint["model_state_dict"])

EvaluationMetrics = make_metrics_dict()

#%%
step_vis = []
# %%
number = 1
cloud_file = f"benchmark/subsampled_data/TLS_Benchmarking_Plot_{number}_MS.pcd"
PointCloud = pclpy.pcl.PointCloud.PointXYZ()
pclpy.pcl.io.loadPCDFile(cloud_file, PointCloud)

#open3dpaint(PointCloud.xyz)

#%%
treetool = TreeTool(PointCloud)
treetool.step_1_remove_floor()
#open3dpaint([treetool.non_ground_cloud.xyz,treetool.ground_cloud.xyz],reduce_for_vis = True  , voxel_size = 0.5, for_thesis=True)

#%%
treetool.step_2_normal_filtering(
    verticality_threshold=0.08, curvature_threshold=0.12, search_radius=0.1
)
#open3dpaint(treetool.filtered_points.xyz, reduce_for_vis = True , voxel_size = 0.1, for_thesis=True)

#%%
x1, y1, z1 = np.min(treetool.filtered_points.xyz, 0)
x2, y2, z2 = np.max(treetool.filtered_points.xyz, 0)
cropfilter = pclpy.pcl.filters.CropBox.PointXYZ()
results_dict = defaultdict(dict)
vis_dict = []
vis_dict2 = []
for x_start in tqdm(np.arange(x1, x2, sample_side_size - sample_side_size * overlap)):
    for y_start in np.arange(y1, y2, sample_side_size - sample_side_size * overlap):
        cropfilter.setMin(np.array([x_start, y_start, -100, 1.0]))
        cropfilter.setMax(
            np.array([x_start + sample_side_size, y_start + sample_side_size, 100, 1.0])
        )
        cropfilter.setInputCloud(treetool.filtered_points)
        sub_pcd = pclpy.pcl.PointCloud.PointXYZ()
        cropfilter.filter(sub_pcd)
        cropfilter.setInputCloud(treetool.non_filtered_points)
        sub_pcd_nf = pclpy.pcl.PointCloud.PointXYZ()
        cropfilter.filter(sub_pcd_nf)
        if np.shape(sub_pcd.xyz)[0] > 0:
            Odata_xyz = data_preprocess(sub_pcd_nf.xyz)
            data_xyz = data_preprocess(sub_pcd.xyz)
            data_xyz = shuffle_data(data_xyz)
            _data_xyz = data_xyz[:4096]
            center, scale = get_center_scale(sub_pcd.xyz)

            if np.shape(_data_xyz)[0] >= 4096:
                nor_testdata = torch.tensor(_data_xyz, device="cuda").squeeze()
                xyz_direction = prediction(model, nor_testdata, args)

                object_center_list, seppoints = center_detection(
                    xyz_direction, voxel_size, ARe, Nd
                )

                if len(seppoints) > 0:
                    seppoints = [i for i in seppoints if np.size(i, 0)]
                    results_dict[x_start][y_start] = {
                        "x": x_start,
                        "y": y_start,
                        "points": data_xyz,
                        "centers": object_center_list,
                        "segmentation": seppoints,
                    }
                    result_points = [(i * scale) + center for i in seppoints]
                    vis_dict.extend(result_points)
                    vis_dict2.append(result_points)

if cfg.BENCHMARKING.COMBINE_IOU:
    vis_dict_ = combine_IOU(vis_dict)
else:
    vis_dict_ = vis_dict

treetool.complete_Stems = vis_dict_
treetool.step_5_get_ground_level_trees()
treetool.step_6_get_cylinder_tree_models()
if cfg.BENCHMARKING.COMBINE_STEMS:
    from sklearn.cluster import dbscan
    models = [i['model'] for i in treetool.finalstems]
    vis = [i['tree'] for i in treetool.finalstems]
    dp = dbscan(np.array(models), eps=1, min_samples=2)[1]

    _models = np.array(models)[dp == -1].tolist()
    _vis = np.array(vis)[dp == -1].tolist()
    for clust in np.unique(dp):
        if clust == -1:
            continue
        _models.append(
            np.array(models)[dp == clust].tolist()[np.argmax([len(i) for i in np.array(models)[dp == clust]])])
        _vis.append(np.vstack(np.array(vis)[dp == clust]).tolist())
    treetool.finalstems = [{'tree': np.array(v), 'model': np.array(m)} for m, v in zip(_models, _vis)]
treetool.step_7_ellipse_fit()

# %%
open3dpaint([np.vstack(i) for i in vis_dict2], pointsize=2, for_thesis=True)
# %%
cloud_match = [i['tree'] for i in treetool.finalstems] + [i for i in treetool.visualization_cylinders]
open3dpaint(cloud_match + [PointCloud.xyz], pointsize=2, for_thesis=True)
# %%
