#%%
import sys
from typing import DefaultDict

sys.path.append("..")
import os

os.chdir("..")
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
from TreeToolML.benchmark.benchmark_utils import load_gt, store_metrics, save_eval_results, load_eval_results, make_benchmark_metrics
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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# %%
EvaluationMetrics = load_eval_results()
BenchmarkMetrics, Methods = make_benchmark_metrics()
# %%
alldata = ["Completeness", "Correctness"]
dificulties = ["easy", "medium", "hard"]
plt.figure(figsize=(16, 46))
for n, i in enumerate(alldata):
    for n2 in range(3):
        plt.subplot(12, 1, n * 3 + n2 + 1)
        plt.title(i + " " + dificulties[n2])
        mine = np.mean(EvaluationMetrics[i][slice(n2, n2 + 2)]) * 100
        colors = [np.array(cm.gist_rainbow(i))*0.5 if i!=1 else cm.gist_rainbow(i) for i in np.linspace(0,1,len(Methods)+1)]
        sortstuff = sorted(
            zip(
                Methods + ["our_imp"],
                BenchmarkMetrics[i][n2] + [mine],
                colors,
            ),
            key=lambda x: x[1],
        )
        sortmethods = [i[0] for i in sortstuff]
        sortnum = [i[1] for i in sortstuff]
        sortcol = [i[2] for i in sortstuff]
        # plt.bar(np.arange(len(BenchmarkMetrics[i][n2])+1),BenchmarkMetrics[i][n2]+[mine])
        plt.bar(sortmethods, sortnum, color=sortcol, width=0.2)
        plt.tight_layout(pad=3.0)
        plt.xticks(rotation=30, fontsize=18)
        plt.grid(axis="y")
# %%
alldata = ["Location_RMSE", "Diameter_RMSE"]
dificulties = ["easy", "medium", "hard"]
plt.figure(figsize=(16, 46))
for n, i in enumerate(alldata):
    for n2 in range(3):
        plt.subplot(12, 1, n * 3 + n2 + 1)
        plt.title(i + " " + dificulties[n2])
        mine = np.mean(EvaluationMetrics[i][slice(n2, n2 + 2)]) * 100
        sortstuff = sorted(
            zip(
                Methods + ["our_imp"],
                BenchmarkMetrics[i][n2] + [mine],
                colors,
            ),
            key=lambda x: x[1],
        )
        sortmethods = [i[0] for i in sortstuff]
        sortnum = [i[1] for i in sortstuff]
        sortcol = [i[2] for i in sortstuff]
        # plt.bar(np.arange(len(BenchmarkMetrics[i][n2])+1),BenchmarkMetrics[i][n2]+[mine])
        plt.bar(sortmethods, sortnum, color=sortcol, width=0.2)
        plt.tight_layout(pad=3.0)
        plt.grid(axis="y")
        plt.xticks(rotation=30, fontsize=18)
# %%
