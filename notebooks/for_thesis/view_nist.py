#%%
import sys
import os

from collections import defaultdict

print(os.getcwd())
import treetool.tree_tool as treeTool
from treetool.seg_tree import box_crop
from treetoolml.utils.py_util import downsample
from porteratzolibs.visualization_o3d.open3d_pointsetClass_new import o3d_pointSetClass
import numpy as np
import pclpy
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from treetoolml.benchmark.benchmark_utils import load_gt, store_metrics, save_eval_results, confusion_metrics
from treetoolml.benchmark.benchmark_utils import make_metrics_dict
import pickle
from treetoolml.utils.vis_utils import tree_vis_tool
import open3d
#%%
#cloud_file = '/home/omar/Downloads/downsampledcloud.pcd'
if True:
    cloud_file = '/home/omar/Downloads/fullCloudSFM.pcd'
    pcd = open3d.io.read_point_cloud(cloud_file)
    #%%
    #%%
    if False:
        down = pcd.voxel_down_sample(0.02)
        open3d.io.write_point_cloud("downsampled25M.pcd", down)
    else:
        down = pcd.voxel_down_sample(0.04)
        open3d.io.write_point_cloud("downsampled9M.pcd", down)
    # %%
pcd = open3d.io.read_point_cloud("downsampled25M.pcd")
points = o3d_pointSetClass(np.asarray(pcd.points),np.asarray(pcd.colors))
tree_vis_tool(points)
# %%
