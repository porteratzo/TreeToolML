#%%
import sys

sys.path.append("..")
import os
import socket
from argparse import ArgumentParser

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import treetoolml.utils.py_util as py_util
from collections import defaultdict
from scipy.spatial import distance_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from treetoolml.config.config import combine_cfgs
from treetoolml.data.BatchSampleGenerator_torch import tree_dataset
from treetoolml.IndividualTreeExtraction_utils.center_detection.center_detection import (
    center_detection,
)
from torch.multiprocessing import Pool
from itertools import repeat
from treetoolml.IndividualTreeExtraction_utils.PointwiseDirectionPrediction_torch import (
    prediction,
)
from treetoolml.model.build_model import build_model
from treetoolml.utils.default_parser import default_argument_parser
from treetoolml.utils.py_util import compute_object_center
from treetoolml.utils.file_tracability import get_model_dir, get_checkpoint_file, find_model_dir
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment

print(socket.gethostname())
if socket.gethostname() == "omar-G5-KC":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

result_dir = '/home/omar/Documents/Mine/Git/TreeToolML/results/RRFSegNet_24:11:2021:22:47'
test_path = os.path.join(result_dir,'test')
with  open(os.path.join(test_path, 'results.pkl'), 'rb') as f:
    data = pickle.load(f)

# %%
data[0]
# %%
MIN_DIST = 0.2
g_tp = defaultdict(int)
g_fp = defaultdict(int)
g_fn = defaultdict(int)
g_acc = defaultdict(list)

for DIST in tqdm(np.linspace(0,1,50)):
    g_tp[DIST] = 0
    g_fp[DIST] = 0
    g_fn[DIST] = 0
    for instance in data:
        dm = distance_matrix(instance[0],instance[1])

        dm[dm>DIST] = 100

        gt_n, det_n = dm.shape
        assigned = linear_sum_assignment(dm)[1]
        True_pos = np.sum(dm[np.arange(len(assigned)),assigned] < DIST)
        False_negative = gt_n - True_pos
        False_pos = det_n - True_pos
        
        good_assigned = np.vstack([np.arange(len(assigned))[dm[np.arange(len(assigned)),assigned] < DIST],assigned[dm[np.arange(len(assigned)),assigned] < DIST]])
        if len(good_assigned) > 0:
            g_acc[DIST].append(dm[good_assigned[0,:],good_assigned[1,:]])
        
        False_pos += len(assigned) - True_pos 
        g_tp[DIST] += True_pos
        g_fp[DIST] += False_pos
        g_fn[DIST] += False_negative
        
#%%
final_acc = {}
for key, i in g_acc.items():
    all_vals = [p for p in i if len(p) > 0]
    if len(all_vals) > 0:
        final_acc[key] = np.mean(np.concatenate(all_vals))
    else:
        final_acc[key] = 0
# %%
thresh = np.array(list(g_tp.keys()))
tp = np.array(list(g_tp.values()))
fp = np.array(list(g_fp.values()))
fn = np.array(list(g_fn.values()))
acc = np.array(list(final_acc.values()))
precision = tp/(tp+fp)
recall = tp/(tp+fn)

tpr = tp/(tp+fn)

# %%
plt.figure(figsize=(28,18))
plt.plot(recall, precision)
# %%
fig, ax_left = plt.subplots(figsize=(28,18))
ax_right = ax_left.twinx()

ax_left.plot(thresh,tp, label='tp')
ax_left.plot(thresh,fp, label='fp')
ax_left.plot(thresh,fn, label='fn')
ax_right.plot(thresh,acc, 'x',label='error')
ax_left.legend(loc="upper left")
ax_right.legend(loc="upper right")
# %%
fig, ax_left = plt.subplots(figsize=(28,18))
ax_right = ax_left.twinx()

ax_left.plot(thresh,tpr, label='tpr')
ax_right.plot(thresh,acc, 'x',label='error')
ax_left.legend(loc="upper left")
ax_right.legend(loc="upper right")
# %%
