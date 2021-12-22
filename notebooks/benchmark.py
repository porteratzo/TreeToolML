#%%

import sys
from typing import DefaultDict

#sys.path.append("..")
import os

#os.chdir("..")
#%%
sys.path.append("/home/omar/Documents/Mine/Git/TreeTool")
import argparse
from collections import defaultdict

import libraries.tree_tool as treeTool
import numpy as np
import pandas as pd
import pclpy
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from TreeToolML.benchmark.benchmark_utils import load_gt, store_metrics, save_eval_results
from TreeToolML.config.config import combine_cfgs
from TreeToolML.IndividualTreeExtraction.center_detection.center_detection import (
    center_detection,
)
from TreeToolML.IndividualTreeExtraction.PointwiseDirectionPrediction_torch import (
    prediction,
)
from TreeToolML.model.build_model import build_model
from TreeToolML.utils.file_tracability import find_model_dir, get_checkpoint_file
from TreeToolML.utils.py_util import (
    bb_intersection_over_union,
    combine_IOU,
    data_preprocess,
    get_center_scale,
    makesphere,
    shuffle_data,
)

#%%
args = argparse.Namespace
args.device = 0
args.cfg = "configs/RRFSegNet.yaml"
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

NUM_POINT = cfg.TRAIN.N_POINTS
Nd = 80
ARe = np.pi / 9.0
voxel_size = 0.08

sample_side_size = 8
overlap = 0.2

model_name = cfg.TRAIN.MODEL_NAME
result_dir = os.path.join("results", model_name)
result_dir = find_model_dir(result_dir)
checkpoint_file = os.path.join(result_dir, "trained_model", "checkpoints")
checkpoint_path = get_checkpoint_file(checkpoint_file)
model = build_model(cfg).cuda()
if device == "cuda":
    model.cuda()
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint["model_state_dict"])

EvaluationMetrics = {}

EvaluationMetrics["Completeness"] = []
EvaluationMetrics["Correctness"] = []
EvaluationMetrics["Mean_AoD"] = []
EvaluationMetrics["Diameter_RMSE"] = []
EvaluationMetrics["Diameter_RMSE_E"] = []
EvaluationMetrics["Diameter_RMSE_C"] = []
EvaluationMetrics["Diameter_bias"] = []
EvaluationMetrics["Location_RMSE"] = []
EvaluationMetrics["Location_bias"] = []

EvaluationMetrics["Relative_Diameter_RMSE"] = []
EvaluationMetrics["Relative_Diameter_bias"] = []
EvaluationMetrics["Relative_Location_RMSE"] = []
EvaluationMetrics["Relative_Location_bias"] = []

EvaluationMetrics["n_ref"] = []
EvaluationMetrics["n_match"] = []
EvaluationMetrics["n_extr"] = []
EvaluationMetrics["location_y"] = []
EvaluationMetrics["diameter_y"] = []

for number in tqdm(range(1,7,1)):
    cloud_file = f"benchmark/subsampled_data/TLS_Benchmarking_Plot_{number}_MS.pcd"
    PointCloud = pclpy.pcl.PointCloud.PointXYZ()
    pclpy.pcl.io.loadPCDFile(cloud_file, PointCloud)

    treetool = treeTool.TreeTool(PointCloud)
    treetool.step_1_remove_floor()
    treetool.step_2_normal_filtering(
        verticality_threshold=0.08, curvature_threshold=0.12, search_radius=0.1
    )

    x1, y1, z1 = np.min(treetool.filtered_points.xyz, 0)
    x2, y2, z2 = np.max(treetool.filtered_points.xyz, 0)
    x_range, y_range, z_range = x2 - x1, y2 - y1, z2 - z1

    cropfilter = pclpy.pcl.filters.CropBox.PointXYZ()
    results_dict = defaultdict(dict)
    vis_dict = []
    for x_start in tqdm(np.arange(x1, x2, sample_side_size - sample_side_size * overlap)):
        for y_start in np.arange(y1, y2, sample_side_size - sample_side_size * overlap):
            cropfilter.setMin(np.array([x_start, y_start, -100, 1.0]))
            cropfilter.setMax(
                np.array([x_start + sample_side_size, y_start + sample_side_size, 100, 1.0])
            )
            cropfilter.setInputCloud(treetool.filtered_points)
            sub_pcd = pclpy.pcl.PointCloud.PointXYZ()
            cropfilter.filter(sub_pcd)
            if np.shape(sub_pcd.xyz)[0] > 0:
                data_xyz = data_preprocess(sub_pcd.xyz)
                data_xyz = shuffle_data(data_xyz)
                data_xyz = data_xyz[:4096]
                center, scale = get_center_scale(sub_pcd.xyz)

                if np.shape(data_xyz)[0] >= 4096:
                    nor_testdata = torch.tensor(data_xyz, device="cuda").squeeze()
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

    vis_dict_ = combine_IOU(vis_dict)

    treetool.complete_Stems = list(vis_dict_.values())
    treetool.step_5_get_ground_level_trees()
    treetool.step_6_get_cylinder_tree_models()
    treetool.step_7_ellipse_fit()

    TreeDict = load_gt(path=f"benchmark/annotations/TLS_Benchmarking_Plot_{number}_LHD.txt")

    CostMat = np.ones([len(TreeDict), len(treetool.visualization_cylinders)])
    for X, datatree in enumerate(TreeDict):
        for Y, foundtree in enumerate(treetool.finalstems):
            CostMat[X, Y] = np.linalg.norm([datatree[0:2] - foundtree["model"][0:2]])

    dataindex, foundindex = linear_sum_assignment(CostMat, maximize=False)

    store_metrics(EvaluationMetrics, treetool, TreeDict, dataindex, foundindex)
save_eval_results(EvaluationMetrics=EvaluationMetrics)