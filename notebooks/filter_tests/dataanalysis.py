#%%
import enum
import sys
import os

from matplotlib.pyplot import axis

sys.path.append("../..")
os.chdir("../..")
import treetoolml.utils.py_util as py_util

sys.path.append("/home/omar/Documents/Mine/Git/TreeTool")
import TreeTool.utils as utils
from treetoolml.config.config import combine_cfgs
from treetoolml.utils.default_parser import default_argument_parser
from treetoolml.data.data_gen_utils.all_dataloader import all_data_loader
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from treetoolml.utils.tictoc import bench_dict

from treetoolml.utils.tictoc import bench_dict
from treetoolml.utils.py_util import outliers, normalize, downsample, normalize_2
from torch.utils.data import Dataset
from treetoolml.Libraries.open3dvis import open3dpaint, sidexsidepaint

######################
# Extra processing for Paris_lille dataset to remove bad trees
######################

#%%
loader = all_data_loader(
    onlyTrees=False, preprocess=True, default=False, train_split=True, new_paris=False
)

loader.load_all("datasets/custom_data/preprocessed")
trees_dicts = defaultdict(dict)
#%%
for _loader, key in tqdm(zip(loader.dataset_list, loader.tree_list_names)):
    print(key)
    all_trees = _loader.get_trees()
    trees_dicts[key]['trees'] = all_trees
    no_outlier_tree = [outliers(down_points, 100, 10) for down_points in all_trees]
    trees_dicts[key]['outliers'] = no_outlier_tree
    norm_tree = [normalize_2(down_points) for down_points in no_outlier_tree]
    trees_dicts[key]['norm'] = norm_tree

    subpoints = [downsample(down_points,0.005) for down_points in norm_tree]
    #subpoints = all_trees
    trees_dicts[key]['down'] = subpoints
    

    print("number of trees ", len(subpoints))
    dims = np.mean([np.max(i, 0) - np.min(i, 0) for i in subpoints], 0)
    volume = dims[0] * dims[1] * dims[2]
    n_points = np.mean([len(i) for i in subpoints])
    print("mean Dims ", dims)
    print("mean points ", n_points)
    print("point density", n_points / volume)
    # test search space
# %%
for _loader, key in tqdm(zip(loader.dataset_list, loader.tree_list_names)):
    print(key)
    subpoints = trees_dicts[key]['trees']
    print("number of trees ", len(subpoints))
    dims = np.mean([np.max(i, 0) - np.min(i, 0) for i in subpoints], 0)
    volume = dims[0] * dims[1] * dims[2]
    n_points = np.mean([len(i) for i in subpoints])
    print("mean Dims ", dims)
    print("mean points ", n_points)
    print("point density", n_points / volume)
# %%
for _loader, key in zip(loader.dataset_list, loader.tree_list_names):
    print(key)
    norm_tree = trees_dicts[key]['norm']
    subpoints = [downsample(down_points,0.005) for down_points in norm_tree]
    trees_dicts[key]['down'] = subpoints

    print("number of trees ", len(subpoints))
    dims = np.mean([np.max(i, 0) - np.min(i, 0) for i in subpoints], 0)
    volume = dims[0] * dims[1] * dims[2]
    n_points = np.mean([len(i[i[:,2]<0.5]) for i in subpoints])
    #n_points = np.mean([len(i) for i in subpoints])
    print("mean Dims ", dims)
    print("mean points ", n_points)
    print("point density", n_points / volume)
# %%
disp = []
for _loader, key in tqdm(zip(loader.dataset_list, loader.tree_list_names)):
    disp.extend(trees_dicts[key]['down'][:20])

sidexsidepaint(disp, pointsize=2)
# %%
disp = []
for _loader, key in tqdm(zip(loader.dataset_list, loader.tree_list_names)):
    pointslists = [i[i[:,2]<0] for i in trees_dicts[key]['down']]
    disp.extend(pointslists[:20])

sidexsidepaint(disp, pointsize=2)
# %%
