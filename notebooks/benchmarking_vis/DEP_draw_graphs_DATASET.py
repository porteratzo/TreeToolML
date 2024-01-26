# %%
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
from treetoolml.IndividualTreeExtraction_utils.center_detection.center_detection_vis import (
    center_detection,
)
from treetoolml.IndividualTreeExtraction_utils.PointwiseDirectionPrediction_torch import (
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
from treetoolml.benchmark.benchmark_utils import make_metrics_dict
import pickle
from porteratzolibs.visualization_o3d.open3dvis import open3dpaint
import pandas as pd
# %%
models_multi = ['trunks','original','center_filtered']
metrics = {}
for model in list(models_multi):
    file_name = model
    result_dir = os.path.join("results_training", file_name)
    result_dir = os.path.join(find_model_dir(result_dir), 'test_results.csv')
    if os.path.isfile(result_dir):
        EvaluationMetrics = pd.read_csv(result_dir)
        metrics[model] = EvaluationMetrics
    else:
        print(model, 'not found')

#%%
# Get the metric values for Original and Trunks
original_completeness = metrics['original']['n_match']
original_correctness = metrics['original']['n_extr']
trunks_completeness = metrics['trunks']['n_match']
trunks_correctness = metrics['trunks']['n_extr']

# X-axis labels
labels = ['Completeness', 'Correctness']

# X-axis positions
x = np.arange(len(labels))

# Bar heights
heights_original = np.array([original_completeness, original_correctness]).flatten()
heights_trunks = np.array([trunks_completeness, trunks_correctness]).flatten()

# Width of the bars
width = 0.35

# Create the plot
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, heights_original, width, label='Original')
rects2 = ax.bar(x + width/2, heights_trunks, width, label='Trunks')
ax.axhline(y=np.array(metrics['trunks']['n_ref']), color='black', linestyle='dotted', label='Total Trees')

for i, rect in enumerate(rects1):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height, f'{height}', ha='center', va='bottom')

for i, rect in enumerate(rects2):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height, f'{height}', ha='center', va='bottom')

h = int(metrics['trunks']['n_ref'])
# Add label to the horizontal line
ax.text(0.5, h, f'{h}', ha='right', va='center', color='black', bbox=dict(facecolor='white', edgecolor='none', pad=3))

# Set plot title and labels
ax.set_title('Metrics Comparison')
ax.set_xlabel('Metrics')
ax.set_ylabel('Values')

# Set x-axis tick labels
ax.set_xticks(x)
ax.set_xticklabels(labels)

# Add legend
ax.legend(loc='upper right')

# Display the plot
plt.savefig('misc_data/found_trees_dataset')
#%%

# %%

centered = pd.read_csv('/home/omar/Downloads/center_filtered_28_05_2023_15_45_trained_model.csv')
orig = pd.read_csv('/home/omar/Downloads/original_28_05_2023_12_40_trained_model.csv')


# %%
import matplotlib.pyplot as plt

original_loss = orig['Value']
trunks_loss = centered['Value']
# Create the plot
plt.figure()
plt.plot(original_loss, label='Original')
plt.plot(trunks_loss, label='Center Filtered')

# Set plot title and labels
plt.title('Total Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Add legend
plt.legend()

# Display the plot
plt.savefig('misc_data/loss_dataset')
# %%
