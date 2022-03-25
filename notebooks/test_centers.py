#%%
import os
import sys

from matplotlib.pyplot import axis

sys.path.append(".")
sys.path.append("/home/omar/Documents/Mine/Git/TreeTool")
sys.path.append("/home/omar/Documents/Mine/Git/TreeToolML")
import os

os.chdir("/home/omar/Documents/Mine/Git/TreeToolML")
import sys

import torch
from TreeToolML.utils.tictoc import bench_dict
import TreeToolML.layers.Loss_torch as Loss_torch
import TreeToolML.utils.py_util as py_util
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm import tqdm
from TreeToolML.config.config import combine_cfgs
from TreeToolML.data.BatchSampleGenerator_torch import tree_dataset
from TreeToolML.model.build_model import build_model
from TreeToolML.utils.default_parser import default_argument_parser
from TreeToolML.utils.file_tracability import get_model_dir
from TreeToolML.data.data_gen_utils.all_dataloader import all_data_loader
from TreeToolML.Libraries.open3dvis import o3d_pointSetClass, open3dpaint
import numpy as np

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
cfg_path = "configs/datasets/trunks.yaml"
cfg = combine_cfgs(cfg_path, [])
savepath = os.path.join(cfg.FILES.DATA_SET, cfg.FILES.DATA_WORK_FOLDER)

train_path = os.path.join(savepath, "training_data")
loader = all_data_loader(
    onlyTrees=False, preprocess=False, default=False, train_split=False, new_paris=True
)
loader.load_all("datasets/custom_data/preprocessed")
# %%
# p =loader.tropical.get_trees()
# p =loader.open.get_trees()
p = loader.paris.get_trees_centers()

# %%
quit()
open3dpaint(
    [i[0] + np.array([[n * 2, 0, 0]]) for n, i in enumerate(p)]
    + [makesphere(i[1]) + np.array([[n * 2, 0, 0]]) for n, i in enumerate(p)],
    pointsize=1,
)
# %%
sorts = np.argsort([len(i) for i in p])
# %%
open3dpaint(p[sorts[-20]], pointsize=1)
# %%
