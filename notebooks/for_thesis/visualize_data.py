#%%
import sys
from typing import DefaultDict

sys.path.append("../..")
import os
sys.path.append('/home/omar/Documents/mine/TreeTool')
os.chdir('/home/omar/Documents/mine/TreeToolML')

import pclpy
import open3d
from treetoolml.Libraries.open3dvis import o3d_pointSetClass, open3dpaint
from treetoolml.config.config import combine_cfgs
from treetoolml.utils.default_parser import default_argument_parser
from treetoolml.data.data_gen_utils.all_dataloader import all_data_loader
from treetoolml.data.BatchSampleGenerator_torch import tree_dataset, tree_dataset_cloud
import numpy as np
import treetoolml.utils.py_util as py_util
from tqdm import tqdm
import os
from argparse import Namespace
from treetoolml.utils.tictoc import bench_dict


#cfg_path = os.path.join('configs','datasets','original')
cfg_path = os.path.join('configs','datasets','trunks.yaml')
cfg = combine_cfgs(cfg_path, [])
#%%
loader = all_data_loader(
    onlyTrees=False, preprocess=False, default=False, train_split=True
)

loader.load_all("datasets/custom_data/preprocessed")

#%%
open3dpaint(loader.open.get_trees(), for_thesis=True)
open3dpaint(loader.paris.get_trees(), for_thesis=True)
open3dpaint(loader.tropical.get_trees(), for_thesis=True)
#%%
open3dpaint(loader.open.point_cloud, pointsize=3, for_thesis=True)
open3dpaint(loader.paris.point_cloud, pointsize=3, for_thesis=True)
open3dpaint(loader.tropical.point_cloud, pointsize=3, for_thesis=True)
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
savepath = os.path.join(cfg.FILES.DATA_SET, cfg.FILES.DATA_WORK_FOLDER)

train_path = os.path.join(savepath, "training_data")
generator_training = tree_dataset_cloud(train_path, cfg.TRAIN.N_POINTS, return_centers=True, normal_filter=True)
for i in np.random.choice(20,10):
    cloud,_,labels, centers = generator_training[i]
    spheres = [makesphere(p,0.05) for p in centers]
    open3dpaint([cloud] + spheres, pointsize=2, axis=0.2)
    #open3dpaint([cloud], pointsize=2, axis=0.1)
    break
#%%
import numpy as np
#generator_training = tree_dataset(train_path, cfg.TRAIN.N_POINTS, return_centers=True, normal_filter=False)
for i in np.random.choice(100,10):
    #cloud,_,labels, centers = generator_training[i]
    spheres = [makesphere(p,0.05) for p in centers]
    open3dpaint([cloud] + spheres, pointsize=3, axis=0.2)
    #open3dpaint([cloud], pointsize=2, axis=0.1)
    fcloud = py_util.normal_filter(cloud, 0.1, 0.6, 0.06)
    open3dpaint([fcloud] + spheres, pointsize=3, axis=0.2)
    break
#%%
import open3d as o3d
# %%
generator_training = tree_dataset(train_path, cfg.TRAIN.N_POINTS, return_centers=True)
for i in tqdm(np.random.choice(100,10)):
    cloud,_,labels, centers = generator_training[i]
bench_dict.save()
# %%

# %%
ixd = py_util.downsample(cloud, return_idx=True)
# %%
ixd

# %%
