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
    onlyTrees=False, preprocess=False, default=False, train_split=True, new_paris=True
)

loader.load_all("datasets/custom_data/preprocessed")
all_trees = loader.paris.get_trees()
saved_trees = []
temp_index_object_xyz = all_trees[2]
out_tree = py_util.outliers(temp_index_object_xyz, 100, 10)
norm_tree = py_util.normalize_2(out_tree)
down_points = py_util.downsample(norm_tree, 0.005)
centerd_tree = down_points
filtered_points = py_util.normal_filter(centerd_tree, 0.04, 0.4, 0.1)
# test search space
#%%
saved_trees = []
visualization_cylinders = []
for i in np.linspace(0.01, 1, 20):
    print(i)
    index, model = py_util.seg_normals(
                filtered_points,
                i,
                0.4,
                distance=0.01,
                rlim=[0, 0.1],
            )
    model = np.array(model)
    Z = 0.5
    Y = model[1] + model[4] * (Z - model[2]) / model[5]
    X = model[0] + model[3] * (Z - model[2]) / model[5]
    model[0:3] = np.array([X, Y, Z])
    # make sure the vector is pointing upward

    model[3:6] = utils.similarize(model[3:6], [0, 0, 1])
    visualization_cylinders.append(
                    utils.makecylinder(model=model, height=1, density=30)
                )
    saved_trees.append(filtered_points)
sidexsidepaint(saved_trees, visualization_cylinders, pointsize=2, axis=1)

#%%
saved_trees = []
visualization_cylinders = []
for i in np.linspace(0.01, 1, 20):
    print(i)
    index, model = py_util.seg_normals(
                filtered_points,
                0.1,
                i,
                distance=0.01,
                rlim=[0, 0.1],
            )
    model = np.array(model)
    Z = 0.5
    Y = model[1] + model[4] * (Z - model[2]) / model[5]
    X = model[0] + model[3] * (Z - model[2]) / model[5]
    model[0:3] = np.array([X, Y, Z])
    # make sure the vector is pointing upward

    model[3:6] = utils.similarize(model[3:6], [0, 0, 1])
    visualization_cylinders.append(
                    utils.makecylinder(model=model, height=1, density=30)
                )
    saved_trees.append(filtered_points)
sidexsidepaint(saved_trees, visualization_cylinders, pointsize=2, axis=1)
# %%
saved_trees = []
visualization_cylinders = []
for temp_index_object_xyz in all_trees:
    out_tree = py_util.outliers(temp_index_object_xyz, 100, 10)
    norm_tree = py_util.normalize_2(out_tree)
    down_points = py_util.downsample(norm_tree, 0.005)
    centerd_tree = down_points
    filtered_points = py_util.normal_filter(centerd_tree, 0.04, 0.4, 0.1)
    
    index, model = py_util.seg_normals(
                filtered_points,
                0.1,
                0.4,
                distance=0.01,
                rlim=[0, 0.2],
            )
    model = np.array(model)
    Z = 0.5
    Y = model[1] + model[4] * (Z - model[2]) / model[5]
    X = model[0] + model[3] * (Z - model[2]) / model[5]
    model[0:3] = np.array([X, Y, Z])
    # make sure the vector is pointing upward

    model[3:6] = utils.similarize(model[3:6], [0, 0, 1])
    visualization_cylinders.append(
                    utils.makecylinder(model=model, height=1, density=30)
                )
    saved_trees.append(filtered_points)
sidexsidepaint(saved_trees, visualization_cylinders, pointsize=2, axis=1)

# %%
