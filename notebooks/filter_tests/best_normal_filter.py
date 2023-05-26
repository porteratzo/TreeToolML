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
from tqdm import tqdm

from treetoolml.utils.tictoc import bench_dict

from treetoolml.utils.tictoc import bench_dict
from torch.utils.data import Dataset
from treetoolml.Libraries.open3dvis import open3dpaint, sidexsidepaint

######################
# Extra processing for Paris_lille dataset to remove bad trees
######################

#%%
loader = all_data_loader(
    onlyTrees=False, preprocess=False, default=False, train_split=True, new_paris=False
)

loader.load_all("datasets/custom_data/preprocessed")
all_trees = loader.paris.get_trees()
saved_trees = []
#%%
temp_index_object_xyz = all_trees[56]
out_tree = py_util.outliers(temp_index_object_xyz, 100, 10)
norm_tree = py_util.normalize_2(out_tree)
down_points = py_util.downsample(norm_tree, 0.005)
open3dpaint(down_points, pointsize=2, axis=1)
# test search space
#%%
saved_trees = []
for i in np.linspace(0.001, 0.2, 10):
    print(i)
    filtered_points = py_util.normal_filter(down_points, i, 0.3, 0.1)
    saved_trees.append(filtered_points)
sidexsidepaint(saved_trees, pointsize=2, axis=1)

#%%
saved_trees = []
for i in np.linspace(0.01, 0.9, 10):
    print(i)
    filtered_points = py_util.normal_filter(down_points, 0.04, i, 0.4)
    saved_trees.append(filtered_points)
sidexsidepaint(saved_trees, pointsize=2, axis=1)


# %%
saved_trees = []
for i in np.linspace(0.01, 0.2, 10):
    
    filtered_points = py_util.normal_filter(down_points, 0.04, 0.3, i)
    saved_trees.append(filtered_points)
    if len(filtered_points) > 100:
        print(i)
sidexsidepaint(saved_trees, pointsize=2, axis=1)
# %%
saved_trees = []
fi = []
for temp_index_object_xyz in tqdm(all_trees[20:40]):
    out_tree = py_util.outliers(temp_index_object_xyz, 100, 10)
    centerd_tree = py_util.normalize_2(out_tree)
    down_points = py_util.downsample(centerd_tree, 0.005)
    filtered_points = py_util.normal_filter(down_points, 0.1, 0.3, 0.1)
    saved_trees.append(down_points)
    saved_trees.append(filtered_points)
    fi.append(filtered_points)
sidexsidepaint(saved_trees, pointsize=2, axis=1)

# %%
sidexsidepaint(fi, pointsize=2, axis=1)
# %%
