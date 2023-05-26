#%%
import os
import sys

from matplotlib.pyplot import axis

sys.path.append(".")
sys.path.append('/home/omar/Documents/mine/TreeTool')

import os

os.chdir('/home/omar/Documents/mine/TreeToolML')
from treetoolml.config.config import combine_cfgs
from treetoolml.data.data_gen_utils.all_dataloader import all_data_loader
from treetoolml.Libraries.open3dvis import o3d_pointSetClass, open3dpaint
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
    onlyTrees=False, preprocess=False, default=False, train_split=False, new_paris=False
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
