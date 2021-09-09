#%%
from collections import defaultdict
import open3d as o3d
from Libraries.Visualization import open3dpaint, open3dpaint_non_block
from datapreparation.data_gen_utils.all_dataloader import all_data_loader
from datapreparation.data_gen_utils.dataloaders import save_cloud, load_cloud
from datapreparation.data_gen_utils.custom_loaders import data_loader
import numpy as np
import glob

loader = all_data_loader(onlyTrees=False, preprocess=False, default=False, train_split=True)
loader.load_all('datasets/custom_data/preprocessed')
#%%
trees = loader.paris.get_trees()
print('unfiltered paris: ',len([len(i) for i in trees]))
print('filtered paris: ',len([len(i) for i in trees if len(i)>500]))
print('average paris: ',np.mean([len(i) for i in trees if len(i)>500]))
print("agerage max mins", np.mean([np.min(i-np.mean(i, axis=0), axis=0) for i in trees if len(i)>500], axis=0), np.mean([np.max(i-np.mean(i, axis=0), axis=0) for i in trees if len(i)>500], axis=0))
#%%
trees = loader.open.get_trees()
print('unfiltered open: ',len([len(i) for i in trees]))
print('filtered open: ',len([len(i) for i in trees if len(i)>500]))
print('average open: ',np.mean([len(i) for i in trees if len(i)>500]))
print("agerage max mins", np.mean([np.min(i-np.mean(i, axis=0), axis=0) for i in trees if len(i)>500], axis=0), np.mean([np.max(i-np.mean(i, axis=0), axis=0) for i in trees if len(i)>500], axis=0))
#%%
trees = loader.tropical.get_trees()
print('unfiltered tropical: ',len([len(i) for i in trees]))
print('filtered tropical: ',len([len(i) for i in trees if len(i)>500]))
print('average tropical: ',np.mean([len(i) for i in trees if len(i)>500]))
print("agerage max mins", np.mean([np.min(i-np.mean(i, axis=0), axis=0) for i in trees if len(i)>500], axis=0), np.mean([np.max(i-np.mean(i, axis=0), axis=0) for i in trees if len(i)>500], axis=0))
#%%

files = glob.glob('datasets/custom_data/PDE/validating_data/*.npy')
ch = np.random.choice(len(files))
loader = np.load(files[ch])
np.random.shuffle(loader)
loader  = loader[:1024*4]
clouds = [loader[loader[:,3]==i, :3] for i in np.unique(loader[:,3])]
open3dpaint(clouds, pointsize=2)


# %%
