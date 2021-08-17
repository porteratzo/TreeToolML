#%%
from collections import defaultdict
import open3d as o3d
from Libraries.Visualization import open3dpaint, open3dpaint_non_block
from data_gen_utils.all_dataloader import all_data_loader
from data_gen_utils.dataloaders import save_cloud, load_cloud
from data_gen_utils.custom_loaders import data_loader
import numpy as np
from plyfile import PlyData, PlyElement, PlyProperty
import glob
import zmq
from Libraries.time_utils import TicTocClass
from tqdm import tqdm
import time
from pygifsicle import optimize
import imageio
import os
from shutil import rmtree
import cv2

##################IQMULUS
loader = np.load('data/custom_data/PDE/validating_data/3.npy')
np.random.shuffle(loader)
loader  = loader[:1024*4]
clouds = [loader[loader[:,3]==i, :3] for i in np.unique(loader[:,3])]
open3dpaint(clouds, pointsize=2)


# %%
