#%%
import enum
import sys
import os

from matplotlib.pyplot import axis

sys.path.append("..")
os.chdir("..")
sys.path.append("/home/omar/Documents/mine/TreeTool")
import treetoolml.utils.py_util as py_util

import TreeTool.utils as utils
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
        onlyTrees=False, preprocess=False, default=False, train_split=True
    )

loader.load_all("datasets/custom_data/preprocessed")
for dataset_loader in tqdm(loader.dataset_list):
    all_trees = dataset_loader.get_trees()
    out_trees = []
    for temp_index_object_xyz in all_trees:
        out_tree = py_util.outliers(temp_index_object_xyz, 100, 10)
        norm_tree = py_util.normalize_2(out_tree)
        down_points = py_util.downsample(norm_tree, 0.005)
        out_trees.append(down_points)


loader.load_all("datasets/custom_data/preprocessed")