# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from collections import defaultdict
import open3d as o3d
from Libraries.Visualization import open3dpaint
from data_gen_utils.all_dataloader import all_data_loader
from data_gen_utils.custom_loaders import data_loader
import numpy as np
from plyfile import PlyData, PlyElement, PlyProperty
import glob
import zmq
from Libraries.time_utils import TicTocClass
from tqdm import tqdm

data = data_loader()
data.load_data("datasets/custom_data/preprocessed/paris.ply")
data.get_trees(unique=True)
open3dpaint(data.get_trees(unique=True))
print('l')