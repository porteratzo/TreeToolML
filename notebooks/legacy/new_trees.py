#%%
import enum
import sys
import os

from matplotlib.pyplot import axis

import treetoolml.utils.py_util as py_util
from Legacy.Plane import similarize 
from porteratzolibs.Misc.geometry import angle_b_vectors 
from treetoolml.config.config import combine_cfgs
from treetoolml.utils.default_parser import default_argument_parser
from treetoolml.data.data_gen_utils.all_dataloader import all_data_loader
import numpy as np
from tqdm import tqdm

from treetoolml.utils.tictoc import bench_dict

from treetoolml.utils.tictoc import bench_dict
from torch.utils.data import Dataset
from porteratzolibs.visualization_o3d.open3dvis import open3dpaint, sidexsidepaint

#%%
loader = all_data_loader(
    onlyTrees=False, preprocess=False, default=False, train_split=True, new_paris=True
)

loader.load_all("datasets/custom_data/preprocessed")
# %%
all_trees = loader.paris.get_trees()
# %%
saved_trees = []
saved_n = []
other_saved_trees = []
for n, tree in enumerate(all_trees[1:4]):
    any_tree = False
    temp_index_object_xyz = tree
    down_points = py_util.downsample(temp_index_object_xyz, 0.01)
    tree = down_points - np.multiply(np.min(down_points, 0), [0, 0, 1])
    tree = tree - np.multiply(np.mean(tree, axis=0), [1, 1, 0])
    out_tree = py_util.outliers(tree, 100, 6)
    filtered_points = py_util.normal_filter(out_tree, 0.6, 0.4, 0.1)
    tree_groups = py_util.group_trees(filtered_points, 0.4)
    tree_trunks = []
    tree_models = []
    visualization_cylinders = []
    if len(tree_groups) == 0:
        continue
    for single_tree in tree_groups:
        index, model = py_util.seg_normals(
            single_tree,
            0.1,
            0.00001,
            distance=0.1,
            rlim=[0, 0.6],
        )
        if len(model) == 0:
            continue
        if abs(np.dot(model[3:6], [0, 0, 1]) / np.linalg.norm(model[3:6])) > 0.5:
            tree_trunks.append(single_tree[index])
            tree_models.append(model)
            model = np.array(model)
            Z = 1.3
            Y = model[1] + model[4] * (Z - model[2]) / model[5]
            X = model[0] + model[3] * (Z - model[2]) / model[5]
            model[0:3] = np.array([X, Y, Z])
            # make sure the vector is pointing upward

            model[3:6] = utils.similarize(model[3:6], [0, 0, 1])
            visualization_cylinders.append(
                utils.makecylinder(model=model, height=3, density=30)
            )
        if len(single_tree[index]) / len(filtered_points) > 0.3:
            any_tree = True
            goodmodel=model
            # saved_trees.append(np.vstack([single_tree[index],utils.makecylinder(model=model, height=3, density=30)]))
            saved_trees.append(
                np.vstack(
                    [
                        out_tree,
                        utils.makecylinder(model=model, height=3, density=30),
                    ]
                )
            )
    if not any_tree:
        other_saved_trees.append(
            np.vstack(
                    out_tree
            )
        )

print(len(saved_trees)), print(len(other_saved_trees))
#%%
def makesphere(centroid=[0, 0, 0], radius=1, dense=90):
    n = np.arange(0, 360, int(360 / dense))
    n = np.deg2rad(n)
    x, y = np.meshgrid(n, n)
    x = x.flatten()
    y = y.flatten()
    sphere = np.vstack(
        [
            centroid[0] + np.sin(x) * np.cos(y) * radius,
            centroid[1] + np.sin(x) * np.sin(y) * radius,
            centroid[2] + np.cos(x) * radius,
        ]
    ).T
    return sphere
#%%
#disptrees = []
#for n, p in enumerate(saved_trees):

sidexsidepaint([out_tree, filtered_points, utils.makecylinder(model=goodmodel, height=3, density=30),], pointsize=2)
#%%
disptrees = []
for n, p in enumerate(saved_trees):
    disptrees.append(p + np.array([5 * n, 0.0, 0]))
print(len(disptrees))
if len(disptrees) > 0:
    open3dpaint(
        disptrees,
        pointsize=2,
    )
#%%
disptrees = []
pfd = 0
for n, p in enumerate(other_saved_trees):
    if len(p) > 39900:
        disptrees.append(p + np.array([10 * pfd, 0.0, 0]))
        pfd += 1
print(len(disptrees))
if len(disptrees) > 0:
    open3dpaint(
        disptrees,
        pointsize=2,
    )
# %%
