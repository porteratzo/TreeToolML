#%%
import sys
from typing import DefaultDict

sys.path.append("..")
import os
sys.path.append('/home/omar/Documents/mine/TreeTool')
#os.chdir("..")

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
from treetoolml.data.data_gen_utils.all_dataloader_fullcloud import all_data_loader_cloud
loader = all_data_loader_cloud(
        onlyTrees=False, preprocess=False, default=False, train_split=True
    )
loader.load_all("datasets/custom_data/full_cloud")

#%%
cluster, labels, centers = loader.get_tree_cluster(
                split="train",
                max_trees=cfg.DATA_CREATION.AUGMENTATION.MAX_TREES,
                translation_xy=cfg.DATA_CREATION.AUGMENTATION.TRANSLATION_XY,
                translation_z=cfg.DATA_CREATION.AUGMENTATION.TRANSLATION_Z,
                scale=cfg.DATA_CREATION.AUGMENTATION.SCALE,
                xy_rotation=cfg.DATA_CREATION.AUGMENTATION.XY_ROTATION,
                dist_between=cfg.DATA_CREATION.AUGMENTATION.MIN_DIST_BETWEEN,
                do_normalize=cfg.DATA_CREATION.AUGMENTATION.DO_NORMALIZE,
            )
open3dpaint(cluster + [makesphere(i[0], 0.1) for i in centers], pointsize=2, for_thesis=True)
# %%
