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
from porteratzolibs.visualization_o3d.open3dvis import open3dpaint


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
        verticality_threshold=0.04, curvature_threshold=0.06, search_radius=0.08
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
[k, idx, _] = pcd_tree.search_radius_vector_3d(sample_pcd_data.points[1000], 0.08)
sub_array = np.delete(np.array(sample_pcd_data.points), idx, axis=0)
bad_neibors = np.array(sample_pcd_data.points)[idx]

import open3d as o3d

# %%
def calculate_zy_rotation_for_arrow(vec):
    gamma = np.arctan2(vec[1], vec[0])
    Rz = np.array([
                    [np.cos(gamma), -np.sin(gamma), 0],
                    [np.sin(gamma), np.cos(gamma), 0],
                    [0, 0, 1]
                ])

    vec = Rz.T @ vec

    beta = np.arctan2(vec[0], vec[2])
    Ry = np.array([
                    [np.cos(beta), 0, np.sin(beta)],
                    [0, 1, 0],
                    [-np.sin(beta), 0, np.cos(beta)]
                ])
    return Rz, Ry

def get_arrow(end, origin=np.array([0, 0, 0]), scale=1):
    assert(not np.all(end == origin))
    #vec = end - origin
    vec = np.array(end)
    size = np.sqrt(np.sum(vec**2))

    Rz, Ry = calculate_zy_rotation_for_arrow(vec)
    mesh = o3d.geometry.TriangleMesh.create_arrow(cone_radius=size/17.5 * scale,
        cone_height=size*0.2 * scale,
        cylinder_radius=size/30 * scale,
        cylinder_height=size*(1 - 0.2*scale))
    mesh.rotate(Ry, center=np.array([0, 0, 0]))
    mesh.rotate(Rz, center=np.array([0, 0, 0]))
    mesh.translate(origin)
    return(mesh)

# %%
from scipy.linalg import eigh
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

arrows = []
colors = [[1,0,0],[0,1,0],[0,0,1]]
for n, (vec, val) in enumerate(zip(*extract_principal_vectors(bad_neibors))):
    arrow = get_arrow(vec*val*10, sample_pcd_data.points[1000],1)
    arrow.paint_uniform_color(colors[n])
    arrows.append(arrow)
    #arrow.paint_uniform_color([0,1,1])
from treetoolml.Libraries.open3dvis import o3d_pointSetClass
bad= o3d_pointSetClass(bad_neibors)
open3dpaint([bad,[sample_pcd_data.points[1000]]] + arrows, for_thesis=True, pointsize=15)
# %%
p = 1501
sample_pcd_data1 = o3d.geometry.PointCloud()
sample_pcd_data1.points = o3d.utility.Vector3dVector(filtered_points)

pcd_tree = o3d.geometry.KDTreeFlann(sample_pcd_data1)
[k, idx, _] = pcd_tree.search_radius_vector_3d(sample_pcd_data1.points[p], 0.08)
sub_array = np.delete(np.array(sample_pcd_data1.points), idx, axis=0)
good_neibors = np.array(sample_pcd_data1.points)[idx]
arrows = []
colors = [[1,0,0],[0,1,0],[0,0,1]]
for n, (vec, val) in enumerate(zip(*extract_principal_vectors(good_neibors))):
    arrow = get_arrow(vec*val*10, sample_pcd_data1.points[p],1)
    arrow.paint_uniform_color(colors[n])
    arrows.append(arrow)
    #arrow.paint_uniform_color([0,1,1])
from treetoolml.Libraries.open3dvis import o3d_pointSetClass
bad= o3d_pointSetClass(good_neibors)
open3dpaint([bad,[sample_pcd_data1.points[p]]] + arrows, for_thesis=True, pointsize=15)
# %%
