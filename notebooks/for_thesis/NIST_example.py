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
from treetoolml.utils.vis_utils import tree_vis_tool
import open3d
import sys
import os

from collections import defaultdict

print(os.getcwd())
import treetool.tree_tool as treeTool
from treetoolml.utils.py_util import downsample
from treetool.seg_tree import box_crop
from treetoolml.utils.py_util import downsample, outliers
from porteratzolibs.visualization_o3d.open3d_pointsetClass_new import o3d_pointSetClass
import numpy as np
import pclpy
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from treetoolml.benchmark.benchmark_utils import load_gt, store_metrics, save_eval_results, confusion_metrics
from treetoolml.benchmark.benchmark_utils import make_metrics_dict
import pickle
from treetoolml.utils.vis_utils import tree_vis_tool
import open3d

#%%
result_dir = os.path.join("results_benchmark", 'NIST_treetool')
os.makedirs(result_dir, exist_ok=True)

cloud_file = "downsampled9M.pcd"
PointCloud = pclpy.pcl.PointCloud.PointXYZ()
visPointCloud = pclpy.pcl.PointCloud.PointXYZRGB()
results_dict = defaultdict(dict)

pclpy.pcl.io.loadPCDFile(cloud_file, PointCloud)
pclpy.pcl.io.loadPCDFile(cloud_file, visPointCloud)

#_PointCloud = downsample(PointCloud.xyz, 0.05)
_PointCloud = PointCloud.xyz
_PointCloud = outliers(_PointCloud, 60)
treetool = treeTool.treetool(_PointCloud)
treetool.step_1_remove_floor(set_max_window_size=5, set_initial_distance=0.8)
if False:
    tree_vis_tool([treetool.ground_cloud.xyz,treetool.non_ground_cloud.xyz])
treetool.step_2_normal_filtering(
    verticality_threshold=0.10, curvature_threshold=0.12, search_radius=0.08
)
if False:
    tree_vis_tool([treetool.filtered_points.xyz,treetool.non_filtered_points.xyz])
treetool.step_3_euclidean_clustering(tolerance=0.1, min_cluster_size=40, max_cluster_size=6000000)
if False:
    tree_vis_tool(treetool.cluster_list)
treetool.step_4_group_stems(max_distance=0.4)
if False:
    tree_vis_tool(treetool.complete_Stems)
treetool.step_5_get_ground_level_trees(lowstems_height=50, cutstems_height=50)
treetool.step_6_get_cylinder_tree_models(search_radius=0.1)
if True:
    vis = o3d_pointSetClass(visPointCloud.xyz, visPointCloud.rgb)
    #tree_vis_tool({'trees':[i['tree'] for i in treetool.finalstems],'cyls':treetool.visualization_cylinders,'vis':vis})
    tree_vis_tool({'trees':[i['tree'] for i in treetool.finalstems],'vis':vis})
treetool.step_7_ellipse_fit()

results_dict['non_filtered_points'] = treetool.non_filtered_points.xyz
results_dict['filtered_points'] = treetool.filtered_points.xyz
results_dict['clustering'] = treetool.cluster_list
results_dict['complete_Stems'] = treetool.complete_Stems
results_dict['finalstems'] = treetool.finalstems
results_dict['visualization_cylinders'] = treetool.visualization_cylinders

TreeDict = load_gt(path=f"benchmark/annotations/TLS_Benchmarking_Plot_{number}_LHD.txt")

CostMat = np.ones([len(TreeDict), len(treetool.finalstems)])
for X, datatree in enumerate(TreeDict):
    for Y, foundtree in enumerate(treetool.finalstems):
        CostMat[X, Y] = np.linalg.norm([datatree[0:2] - foundtree["model"][0:2]])

dataindex, foundindex = linear_sum_assignment(CostMat, maximize=False)
store_metrics(EvaluationMetrics, treetool, TreeDict, dataindex, foundindex)
confMat_list.append(confusion_metrics(treetool, TreeDict, dataindex, foundindex))
result_list.append(results_dict)
save_eval_results(path=f'{result_dir}/results.npz', EvaluationMetrics=EvaluationMetrics)
np.savez(f'{result_dir}/confusion_results.npz', confMat_list=confMat_list)
with open(f'{result_dir}/results_dict.pk', 'wb') as f:
    pickle.dump(result_list, f)
with open(f'{result_dir}/results_dict_iou.pk', 'wb') as f:
    pickle.dump(process_dict, f)

