#%%
import sys
import os

from collections import defaultdict

print(os.getcwd())
import treetool.tree_tool as treeTool
import numpy as np
import pclpy
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from treetoolml.benchmark.benchmark_utils import load_gt, store_metrics, save_eval_results, confusion_metrics
from treetoolml.benchmark.benchmark_utils import make_metrics_dict
import pickle
from treetool.seg_tree import radius_outlier_removal
from porteratzolibs.visualization_o3d.open3dvis import open3dpaint
from porteratzolibs.visualization_o3d.open3d_pointsetClass import o3d_pointSetClass
from porteratzolibs.visualization_o3d.create_geometries import make_arrow
from porteratzolibs.Misc.geometry import getPrincipalVectors
from scipy.linalg import eigh
import open3d as o3d
#%%
result_dir = os.path.join("results_benchmark", 'treetool')
os.makedirs(result_dir, exist_ok=True)
EvaluationMetrics = make_metrics_dict()
visualize = False

confMat_list = []
result_list = []
process_dict = []

for number in tqdm(range(1, 2, 1), desc='forest plot'):
    cloud_file = f"benchmark/subsampled_data/TLS_Benchmarking_Plot_{number}_MS.pcd"
    PointCloud = pclpy.pcl.PointCloud.PointXYZ()
    results_dict = defaultdict(dict)

    pclpy.pcl.io.loadPCDFile(cloud_file, PointCloud)
    treetool = treeTool.treetool(PointCloud)
    treetool.step_1_remove_floor()
    treetool.step_2_normal_filtering(
        verticality_threshold=0.04, curvature_threshold=0.06, search_radius=0.08, min_points=8
    )

# %%
import treetool.seg_tree as seg_tree
non_filtered_points = treetool.non_filtered_points.xyz
filtered_points = treetool.filtered_points.xyz


# %%
import open3d as o3d
sample_pcd_data = o3d.geometry.PointCloud()
sample_pcd_data.points = o3d.utility.Vector3dVector(non_filtered_points)

pcd_tree = o3d.geometry.KDTreeFlann(sample_pcd_data)
# %%

def extract_principal_vectors(point_cloud):
    # Center the point cloud
    centroid = np.mean(point_cloud, axis=0)
    centered_points = point_cloud - centroid

    # Compute the covariance matrix
    covariance_matrix = np.cov(centered_points.T)

    # Perform eigen decomposition
    eigenvalues, eigenvectors = eigh(covariance_matrix)

    # Sort the eigenvalues and eigenvectors in descending order
    indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]

    return eigenvectors, eigenvalues
# %%
p_ = 1900
[k, idx, _] = pcd_tree.search_radius_vector_3d(sample_pcd_data.points[p_], 0.08)
bad_neibors = np.array(sample_pcd_data.points)[idx]

[k, idx2, _] = pcd_tree.search_radius_vector_3d(sample_pcd_data.points[p_], 0.4)
idx_ = np.array(idx2)[~np.in1d(idx2,idx)]
more_bad_neigb = np.array(sample_pcd_data.points)[idx_]
arrows = []
colors = [[1,0,0],[0,1,0],[0,0,1]]
for n, (vec, val) in enumerate(zip(*extract_principal_vectors(bad_neibors))):
    arrow = make_arrow(vec*val*500, sample_pcd_data.points[p_],1)
    arrow.paint_uniform_color(colors[n])
    arrows.append(arrow)
    #arrow.paint_uniform_color([0,1,1])
bad= o3d_pointSetClass(more_bad_neigb)
bad_nei= o3d_pointSetClass(bad_neibors, [0,0,0])
open3dpaint([bad, bad_nei,[sample_pcd_data.points[p_]]] + arrows, for_thesis=True, pointsize=15)
# %%
p = 160
sample_pcd_data1 = o3d.geometry.PointCloud()
sample_pcd_data1.points = o3d.utility.Vector3dVector(filtered_points)

pcd_tree = o3d.geometry.KDTreeFlann(sample_pcd_data1)
[k, idx, _] = pcd_tree.search_radius_vector_3d(sample_pcd_data1.points[p], 0.08)
good_neibors = np.array(sample_pcd_data1.points)[idx]

[k, idx2, _] = pcd_tree.search_radius_vector_3d(sample_pcd_data1.points[p], 1)
idx_ = np.array(idx2)[~np.in1d(idx2,idx)]
more_good_neigb = np.array(sample_pcd_data1.points)[idx_]
arrows = []
colors = [[1,0,0],[0,1,0],[0,0,1]]
for n, (vec, val) in enumerate(zip(*getPrincipalVectors(good_neibors))):
    arrow = make_arrow(np.array(vec)*np.array(val)*20, sample_pcd_data1.points[p],1)
    arrow.paint_uniform_color(colors[n])
    arrows.append(arrow)
    #arrow.paint_uniform_color([0,1,1])
bad= o3d_pointSetClass(more_good_neigb)
more_db= o3d_pointSetClass(good_neibors, [0,0,0])
open3dpaint([bad, more_db,[sample_pcd_data1.points[p]]] + arrows, for_thesis=True, pointsize=15)
# %%
