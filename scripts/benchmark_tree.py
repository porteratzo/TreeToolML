#%%
import open3d
import sys
import numpy as np
import os
from open3dvis import open3dpaint, o3d_pointSetClass
import pdal
import os
import IndividualTreeExtraction.PointwiseDirectionPrediction_torch as PDE_net
import py_util
from IndividualTreeExtraction.center_detection.center_detection import center_detection
import torch

def FloorRemove(
    points, scalar=0.2, slope=0.2, threshold=0.45, window=16.0, RGB=False
):
    open3d.io.write_point_cloud("LIDARRF.ply", PointCloud)
    json = """
    [
        "LIDARRF.ply",
        {
            "type":"filters.smrf",
            "scalar":1.25,
            "slope":0.15,
            "threshold":0.5,
            "window":18.0
        }
    ]
    """
    pipeline = pdal.Pipeline(json)
    pipeline.validate()
    pipeline.execute()
    arrays = pipeline.arrays
    points1 = arrays[0][arrays[0]["Classification"] == 1]
    points2 = arrays[0][arrays[0]["Classification"] == 2]

    Nogroundpoints = np.array(points1[["X", "Y", "Z"]].tolist())
    ground = np.array(points2[["X", "Y", "Z"]].tolist())
    os.remove("LIDARRF.ply")

    return Nogroundpoints, ground


def preprocess_data(points):
    temp_xyz = py_util.normalize(points)
    return temp_xyz


PointCloud = open3d.io.read_point_cloud('/home/omar/Documents/Mine/tests/NistClouds/downsampledlesscloudEURO4.pcd')
nonground, ground = FloorRemove(PointCloud)
nongroundPC = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(nonground))

#%%
print(np.array(nongroundPC.points).shape)
downpcd = nongroundPC.voxel_down_sample(voxel_size=0.1)
print(np.array(downpcd.points).shape)
#open3dpaint(np.array(downpcd.points))
#%%

labels = np.array(downpcd.cluster_dbscan(0.4, 100, True))

clusters = []
points = np.array(downpcd.points)
for i in np.unique(labels):
    clusters.append(points[i==labels])
print(len(clusters))
#%%
open3dpaint(clusters)

# %%
PDE_net_model_path ='IndividualTreeExtraction/pre_trained_PDE_net/'
NUM_POINT = 4092
model = PDE_net.restore_trained_model(PDE_net_model_path).cuda()
# %%
np.random.shuffle(clusters[2])
allpoints = clusters[2][:NUM_POINT]
testdata = torch.tensor(preprocess_data(allpoints))
with torch.cuda.amp.autocast():
    out = model(torch.unsqueeze(testdata.cuda(),0))
# %%
object_center_list = center_detection(xyz_direction, voxel_size, ARe, Nd)