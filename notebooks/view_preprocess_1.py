#%%
import enum
import sys
import os

from matplotlib.pyplot import axis

sys.path.append("..")
os.chdir("..")
import TreeToolML.utils.py_util as py_util

sys.path.append("/home/omar/Documents/Mine/Git/TreeTool")
import TreeTool.utils as utils
from TreeToolML.config.config import combine_cfgs
from TreeToolML.utils.default_parser import default_argument_parser
from TreeToolML.data.data_gen_utils.all_dataloader import all_data_loader
import numpy as np
from tqdm import tqdm

from TreeToolML.utils.tictoc import bench_dict

from TreeToolML.utils.tictoc import bench_dict
from torch.utils.data import Dataset
from TreeToolML.Libraries.open3dvis import open3dpaint, sidexsidepaint

#%%
cfg_path = args.cfg
cfg = combine_cfgs(cfg_path, args.opts)

loader = all_data_loader(
    onlyTrees=False, preprocess=False, default=False, train_split=True
)

all_datasets = defaultdict(list)

loader.load_all("datasets/custom_data/preprocessed")