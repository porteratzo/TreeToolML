# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from collections import defaultdict
import open3d as o3d
from Libraries.Visualization import open3dpaint
from data_gen_utils.all_dataloader import all_data_loader
import numpy as np
from plyfile import PlyData, PlyElement, PlyProperty
import glob
import zmq
from Libraries.time_utils import TicTocClass
from tqdm import tqdm


##################IQMULUS
timer = TicTocClass()
timer.tic()
loader = all_data_loader(onlyTrees=False)
timer.pttoc('data_loader')



#%%
if True:
    loader.iqumulus.load_data()
    timer.pttoc('iqumulus')
    #%%
    points = loader.iqumulus.get_trees()
    loader.iqumulus.save_cloud('')
    timer.pttoc('iqumulus trees')
    open3dpaint(points)
    timer.tic()

    #%%

    loader.tropical.load_data()
    timer.pttoc('tropical')
    #%%
    points = loader.tropical.get_trees()
    timer.pttoc('tropical trees')
    open3dpaint(points)
    timer.tic()

    #%%

    loader.open.load_data()
    timer.pttoc('open')
    #%%
    points = loader.open.get_trees()
    timer.pttoc('open trees')
    open3dpaint(points)
    timer.tic()

#%%

loader.paris.load_data()
timer.pttoc('paris')
#%%
points = loader.paris.get_trees()
timer.pttoc('paris trees')
open3dpaint(points)
timer.tic()