#%%
import pclpy
import open3d as o3d
from json import load

from treetoolml.config.config import combine_cfgs
from treetoolml.utils.default_parser import default_argument_parser
from treetoolml.data.data_gen_utils.all_dataloader import all_data_loader
from treetoolml.data.data_gen_utils.all_dataloader_fullcloud import all_data_loader_cloud
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from treetoolml.utils.tictoc import bench_dict
from treetoolml.data.BatchSampleGenerator_torch import tree_dataset_cloud
from treetoolml.Libraries.open3dvis import open3dpaint
#%%
#original datasets
loader = all_data_loader(
        onlyTrees=False, preprocess=False, default=False, train_split=True
)

loader.load_all("datasets/custom_data/preprocessed")
all_trees = []
for dataset_loader in tqdm(loader.dataset_list):
    all_trees.extend(dataset_loader.get_trees())
centered_trees = []
for n, tree in enumerate(all_trees):
    centerd_tree = tree - np.multiply(np.min(tree, 0), [0, 0, 1])
    centerd_tree = centerd_tree - np.multiply(np.mean(centerd_tree, axis=0), [1, 1, 0])
    centerd_tree = centerd_tree/np.max(np.linalg.norm(centerd_tree, axis=1))
    centerd_tree = centerd_tree + [n*0.1,0,0]
    centered_trees.append(centerd_tree) 
    
open3dpaint(centered_trees, axis=1)

# %%

loader = all_data_loader_cloud(
        onlyTrees=False, preprocess=False, default=False, train_split=True
    )
loader.load_all("datasets/custom_data/full_cloud")
centered_trees = []
for n, tree in enumerate(loader.full_cloud.trees):
    centerd_tree = tree + [n*0.5,0,0]
    centered_trees.append(centerd_tree) 
open3dpaint(centered_trees, axis=1)
[len(i) for i in loader.full_cloud.trees]
# %%
val_path = 'datasets/custom_data/trunks/validating_data'
loader = tree_dataset_cloud(val_path, 4096, normal_filter=True, distances=1, return_centers=True)
# %%
from treetoolml.utils.vis_utils import vis_trees_centers
n = 21
vis_trees_centers(loader[n][0], loader[n][3])
# %%
