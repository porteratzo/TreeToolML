# %%

import sys
from typing import DefaultDict

# sys.path.append("../..")
import os

# os.chdir("../..")
# %%
sys.path.append("/home/omar/Documents/mine/IndividualTreeExtraction/voxel_region_grow/")
import argparse
from collections import defaultdict

from TreeTool.tree_tool import TreeTool
import TreeTool.seg_tree as seg_tree
import numpy as np
import pandas as pd
import pclpy
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from treetoolml.benchmark.benchmark_utils import load_gt, store_metrics, save_eval_results
from treetoolml.config.config import combine_cfgs
from treetoolml.IndividualTreeExtraction.center_detection.center_detection import (
    center_detection,
)
from treetoolml.IndividualTreeExtraction.PointwiseDirectionPrediction_torch import (
    prediction,
)
from treetoolml.model.build_model import build_model
from treetoolml.utils.file_tracability import find_model_dir, get_checkpoint_file
from treetoolml.utils.py_util import (
    bb_intersection_over_union,
    combine_IOU,
    data_preprocess,
    get_center_scale,
    makesphere,
    shuffle_data,
)
from treetoolml.Libraries.open3dvis import open3dpaint

# %%
cfg = combine_cfgs("configs/datasets/trunks.yaml", [])
model_name = cfg.TRAIN.MODEL_NAME
result_dir = os.path.join("results", model_name)
result_dir = find_model_dir(result_dir)
data = np.load(f'{result_dir}/confusion_results.npz', allow_pickle=True)
confMat_list = data['confMat_list']
# %%
plotnumber = 1

cloud_file = f"benchmark/subsampled_data/TLS_Benchmarking_Plot_{plotnumber + 1}_MS.pcd"
PointCloud = pclpy.pcl.PointCloud.PointXYZ()
pclpy.pcl.io.loadPCDFile(cloud_file, PointCloud)
forrest = seg_tree.voxelize(seg_tree.floor_remove(PointCloud)[0])

# %%
print('see true positives')
tp_gt_trees = []
tp_found_trees = []
for record in confMat_list[plotnumber][0]:
    if record['true_tree'] is not None:
        tp_gt_trees.append(record['true_tree'])
    if record['found_tree'] is not None:
        tp_found_trees.append(record['found_tree'])
open3dpaint(tp_gt_trees + tp_found_trees, pointsize=2)
open3dpaint([forrest + [0.2, 0, 0]] + tp_found_trees, pointsize=2)
# %%
print('see false fp')
fp_found_trees = []
for record in confMat_list[plotnumber][1]:
    if record['found_tree'] is not None:
        fp_found_trees.append(record['found_tree'])
# open3dpaint([forrest + [0.2, 0, 0]] + fp_found_trees, pointsize=2)
open3dpaint(fp_found_trees, pointsize=2)

# %%
print('see false negatives')
fn_found_trees = []
for record in confMat_list[plotnumber][2]:
    if record['true_tree'] is not None:
        fn_found_trees.append(record['true_tree'])
# open3dpaint([forrest + [0.2, 0, 0]] + fn_found_trees, pointsize=2)
open3dpaint(fn_found_trees, pointsize=2)

# %%
open3dpaint(
    [forrest + [0.3, 0, 0]] + [np.vstack(tp_found_trees) + [0.2, 0, 0]] + [np.vstack(fp_found_trees) + [0.1, 0, 0]] + [
        np.vstack(fn_found_trees)], pointsize=2, axis=1)
# %%
[len(i) for i in confMat_list[1]]
