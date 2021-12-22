#%%

import sys
sys.path.append("..")
import os
os.chdir('..')
#%%
sys.path.append('/home/omar/Documents/Mine/Git/TreeTool')
import open3d as o3d
import laspy
import numpy as np
from glob import glob
from tqdm import tqdm
from TreeToolML.Libraries.open3dvis import open3dpaint
# %%

#cloud_file = 'datasets/bogdan/LabeledTLS/Plot3Group.las'
cloud_file = 'datasets/bogdan/LabeledTLS/TreeFive.las'
pcd = laspy.read(cloud_file)
points = np.vstack((pcd.x, pcd.y, pcd.z)).transpose()
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
downpcd = pcd.voxel_down_sample(voxel_size=0.04)
# %%
open3dpaint(np.asarray(downpcd.points), pointsize=2)
# %%
