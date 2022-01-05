#%%
import sys
from typing import DefaultDict

sys.path.append("..")
import os

os.chdir("..")

#%%
import pclpy
import open3d
from TreeToolML.Libraries.open3dvis import o3d_pointSetClass, open3dpaint
from TreeToolML.config.config import combine_cfgs
from TreeToolML.utils.default_parser import default_argument_parser
from TreeToolML.data.data_gen_utils.all_dataloader import all_data_loader
from TreeToolML.data.BatchSampleGenerator_torch import tree_dataset
import numpy as np
import TreeToolML.utils.py_util as py_util
from tqdm import tqdm
import os
from argparse import Namespace
from TreeToolML.utils.tictoc import bench_dict

#cfg_path = os.path.join('configs','datasets','original')
cfg_path = os.path.join('configs','datasets','trunks.yaml')
cfg = combine_cfgs(cfg_path, [])
#%%
if False:
    loader = all_data_loader(
        onlyTrees=False, preprocess=False, default=False, train_split=True
    )

    loader.load_all("datasets/custom_data/preprocessed")
    #%%
    cluster, labels = loader.get_tree_cluster(
        train=True,
        max_trees=cfg.DATA_CREATION.AUGMENTATION.MAX_TREES,
        translation_xy=cfg.DATA_CREATION.AUGMENTATION.TRANSLATION_XY,
        translation_z=cfg.DATA_CREATION.AUGMENTATION.TRANSLATION_Z,
        scale=cfg.DATA_CREATION.AUGMENTATION.SCALE,
        xy_rotation=cfg.DATA_CREATION.AUGMENTATION.XY_ROTATION,
        dist_between=cfg.DATA_CREATION.AUGMENTATION.MIN_DIST_BETWEEN,
        do_normalize=cfg.DATA_CREATION.AUGMENTATION.DO_NORMALIZE,
        zero_floor=cfg.DATA_CREATION.AUGMENTATION.ZERO_FLOOR
    )
    open3dpaint(cluster, pointsize=2, axis=1)


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
generator_training = tree_dataset(train_path, cfg.TRAIN.N_POINTS, return_centers=True, normal_filter=True)
for i in np.random.choice(100,10):
    cloud,_,labels, centers = generator_training[i]
    spheres = [makesphere(p,0.05) for p in centers]
    open3dpaint([cloud] + spheres, pointsize=2, axis=0.2)
    #open3dpaint([cloud], pointsize=2, axis=0.1)
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
