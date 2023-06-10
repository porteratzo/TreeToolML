#%%
import os
from collections import defaultdict

import treetool.tree_tool as treeTool
import numpy as np
import pclpy
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from treetoolml.utils.default_parser import default_argument_parser
from treetoolml.benchmark.benchmark_utils import load_gt, store_metrics, save_eval_results, confusion_metrics
from treetoolml.config.config import combine_cfgs
from treetoolml.IndividualTreeExtraction.center_detection.center_detection_vis import (
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
from treetoolml.utils.benchmark_util import make_metrics_dict
import pickle
from treetoolml.Libraries.open3dvis import open3dpaint

#%%
number=5
TreeDict = []
for number in range(1,7):
    TreeDict.append(load_gt(path=f"benchmark/annotations/TLS_Benchmarking_Plot_{number}_LHD.txt"))
#%%
import matplotlib.pyplot as plt
for n, points in enumerate(TreeDict):
    points = np.array(points)
    plt.scatter(points[:,0]+40*n, points[:,1])

# Add labels and title
plt.title("Scatter Plot")
plt.xlabel("X")
plt.ylabel("Y")

# Display the plot
plt.show()
#%%
densities = []
for n, points in enumerate(TreeDict):
    points = np.array(points)
    mins = np.min(points, 0)[:2]
    maxs = np.max(points, 0)[:2]
    dists = maxs - mins
    print('########### plot numer ',n,'###########')
    for size in [10,20,30]:
        minxy = (dists - size)/2 + mins
        maxxy = maxs - (dists - size)/2 
        dist_xy = maxxy - minxy
        bools = (points[:,:2] > minxy.reshape(-1,2)) & (points[:,:2] < maxxy.reshape(-1,2))
        bools = np.all(bools, axis=1)
        print('trees found ',sum(bools))
        #print('dims ',dist_xy)
        print('area ', np.prod(dist_xy))
        print('density ', sum(bools)/np.prod(dist_xy))
        densities.append(sum(bools)/np.prod(dist_xy))

# %%
x_positions = ['Easy', 'Medium', 'Hard']

# Tree density values
densities_ = [np.mean(densities[i:i+6]) for i in range(0,18,6)]

# Create the bar plot
plt.bar(x_positions, densities_)

# Add labels and title
plt.title("Tree Density")
plt.xlabel("Difficulty Level")
plt.ylabel("Tree Density")

# Display the plot
plt.show()
# %%
side = 6
print('for side size ', side, 'm')
print('expected trees are ', densities_[-1] * side * side)
print('with an average points per tree of ', 4016/(densities_[-1] * side * side))

print('if you get 3 trees', densities_[-1] * side * side +3)
print('with an average points per tree of ', 4016/(densities_[-1] * side * side + 3))
# %%
from scipy.spatial.distance import cdist
# %%
for n, points in enumerate(TreeDict):
    print('')
    print('########### plot numer ',n,'###########')
    points = np.array(points)[:,:2]
    distance_matrix = cdist(points, points)
    distance_matrix[distance_matrix==0] = 1000
    print('distance between trees')
    print('min', np.min(distance_matrix))
    print('mean', np.mean(np.min(distance_matrix, axis=0)))
    print('percentile', np.percentile(np.min(distance_matrix, axis=0), 5))
# %%
