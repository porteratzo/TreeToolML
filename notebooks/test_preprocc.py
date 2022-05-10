#%%
import os
import sys

from matplotlib.pyplot import axis

sys.path.append(".")
sys.path.append('/home/omar/Documents/mine/TreeTool')
#sys.path.append("/home/omar/Documents/Mine/Git/TreeToolML")
import os
os.chdir('/home/omar/Documents/mine/TreeToolML')
from TreeToolML.config.config import combine_cfgs
from TreeToolML.data.data_gen_utils.all_dataloader import all_data_loader
from TreeToolML.Libraries.open3dvis import o3d_pointSetClass, open3dpaint
import numpy as np
#%%
cfg_path = 'configs/datasets/trunks.yaml'
cfg = combine_cfgs(cfg_path, [])
savepath = os.path.join(cfg.FILES.DATA_SET, cfg.FILES.DATA_WORK_FOLDER)

train_path = os.path.join(savepath, "training_data")
loader = all_data_loader(
        onlyTrees=False, preprocess=False, default=False, train_split=True
    )
loader.load_all("datasets/custom_data/preprocessed")
# %%
#p =loader.tropical.get_trees()
#p =loader.open.get_trees()
p =loader.paris.get_trees()

# %%
open3dpaint(p,pointsize=1)
# %%
sorts = np.argsort([len(i) for i in p])
# %%
open3dpaint(p[sorts[-20]],pointsize=1)
# %%
