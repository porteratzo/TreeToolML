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
# %%
os.makedirs('benchmark/subsampled_data', exist_ok=True)
for cloud_file in tqdm(glob('benchmark/original/*.las')):
    file_name = os.path.splitext(os.path.basename(cloud_file))[0]
    pcd = laspy.read(cloud_file)
    points = np.vstack((pcd.x, pcd.y, pcd.z)).transpose()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    downpcd = pcd.voxel_down_sample(voxel_size=0.04)
    o3d.io.write_point_cloud(f'benchmark/subsampled_data/{file_name}.pcd',downpcd)
# %%
