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

import os
import pickle
from collections import defaultdict

import numpy as np
import pclpy
import torch
import treetool.tree_tool as treeTool
from tqdm import tqdm

from treetoolml.benchmark.benchmark_utils import (
    confusion_metrics,
    load_gt,
    make_metrics_dict,
    run_combine_stems,
    run_detection,
    sample_generator,
    save_eval_results,
    store_metrics,
    matching,
)
from treetoolml.config.config import combine_cfgs
from treetoolml.model.build_model import build_model
from treetoolml.utils.default_parser import default_argument_parser
from treetoolml.utils.file_tracability import (
    find_model_dir,
    get_checkpoint_file,
    get_model_dir,
)
from treetoolml.utils.py_util import combine_IOU
import argparse
import sys
from typing import DefaultDict

import os
import argparse
from collections import defaultdict
from porteratzolibs.visualization_o3d.create_geometries import make_arrow, make_cylinder
from treetool.seg_tree import box_crop
import numpy as np
import pclpy.pcl as pcl
import torch
import pclpy
import treetool.seg_tree as seg_tree

from tqdm import tqdm
from treetoolml.config.config import combine_cfgs
from treetoolml.IndividualTreeExtraction_utils.center_detection.center_detection import (
    center_detection,
)
from scipy.optimize import linear_sum_assignment
from treetoolml.IndividualTreeExtraction_utils.PointwiseDirectionPrediction_torch import (
    prediction,
)
from treetoolml.model.build_model import build_model
from treetoolml.utils.file_tracability import (
    find_model_dir,
    get_checkpoint_file,
    get_model_dir,
)
from treetoolml.utils.py_util import (
    combine_IOU,
    data_preprocess,
    get_center_scale,
    shuffle_data,
)

# from porteratzolibs.visualization_o3d.open3dvis import open3dpaint, sidexsidepaint
from porteratzolibs.visualization_o3d.open3dvis_new import open3dpaint, sidexsidepaint
from porteratzolibs.visualization_o3d.create_geometries import make_plane
from treetoolml.benchmark.benchmark_utils import make_metrics_dict, load_gt
from treetoolml.utils.torch_utils import (
    device_configs,
    find_training_dir,
    load_checkpoint,
)
from treetoolml.benchmark.benchmark_utils import (
    sample_generator,
    run_detection,
    run_combine_stems,
    cloud_match,
    cloud_match_clusters,
    metrics_2_clouds,
    print_metrics,
    make_cost_mat,
    close_points_ratio_merge
)
from treetoolml.utils.vis_utils import tree_vis_tool
from scipy.spatial import distance
#%%
args = argparse.Namespace
args.device = "cuda"
#args.cfg = "configs/experimentos_model/subconfigs/distance_out_loss.yaml"
args.cfg = "configs/datasets/subconfigs/trunks_new.yaml"
args.gpu_number = 0
args.amp = True
args.opts = []

cfg_path = args.cfg
cfg = combine_cfgs(cfg_path, args.opts)
use_amp = args.amp

Nd = cfg.BENCHMARKING.XY_THRESHOLD
ARe = np.deg2rad(cfg.BENCHMARKING.ANGLE_THRESHOLD)
voxel_size = cfg.BENCHMARKING.VOXEL_SIZE
sample_side_size = cfg.BENCHMARKING.WINDOW_STRIDE
overlap = cfg.BENCHMARKING.OVERLAP

model_dir = find_training_dir(cfg)
model = build_model(cfg)
load_checkpoint(model_dir, model)
device_configs(model, args)

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
    verticality_threshold=0.12, curvature_threshold=0.14, search_radius=0.08
)
if False:
    tree_vis_tool([treetool.filtered_points.xyz,treetool.non_filtered_points.xyz])

results_dict = defaultdict(dict)
vis_dict = []

sample_side_size = 8
overlap = cfg.BENCHMARKING.OVERLAP
cfg.defrost()
cfg.DATA_PREPROCESSING.DISTANCE_FILTER = 0.8
generator = sample_generator(sample_side_size, overlap, treetool)

run_detection(args, cfg, model, results_dict, vis_dict, generator, tolerence=0.05)

if cfg.BENCHMARKING.GROUP_STEMS:
    treetool.cluster_list = vis_dict

if False:
    generator = sample_generator(sample_side_size, overlap, treetool)
    poi = [i[-2].xyz for i in list(generator)]
    _poi = [i for i in poi if len(i) > 0]
    tree_vis_tool(_poi)

if False:
    tree_vis_tool({'cluster':treetool.cluster_list, 'filter':treetool.filtered_points.xyz})
    prd = np.vstack([(results_dict[key][key2]['prediction'][:,:3] * results_dict[key][key2]['scale']) + results_dict[key][key2]['center']  for key in results_dict.keys() for key2 in results_dict[key].keys()])
    #prd = np.vstack([(results_dict[key][key2]['Ipoints'][:,:3] * results_dict[key][key2]['scale']) + results_dict[key][key2]['center']  for key in results_dict.keys() for key2 in results_dict[key].keys()])
    tree_vis_tool(prd)

if cfg.BENCHMARKING.GROUP_STEMS:
    treetool.cluster_list = vis_dict
    treetool.step_4_group_stems()

if False:
    tree_vis_tool(treetool.complete_Stems)


if cfg.BENCHMARKING.COMBINE_IOU:
    vis_dict_ = combine_IOU(vis_dict)
else:
    vis_dict_ = vis_dict

treetool.complete_Stems = vis_dict_
treetool.step_5_get_ground_level_trees(20,20)
treetool.step_6_get_cylinder_tree_models(stick=False)

run_combine_stems(cfg, treetool)
if True:
    vis = o3d_pointSetClass(visPointCloud.xyz, visPointCloud.rgb)
    #tree_vis_tool({'trees':[i['tree'] for i in treetool.finalstems],'cyls':treetool.visualization_cylinders,'vis':vis})
    tree_vis_tool({'trees':[i['tree'] for i in treetool.finalstems],'vis':vis})
# open3dpaint([np.vstack(_vis) + [0.1, 0, 0]] + vis, pointsize=2)
treetool.finalstems = [i for i in treetool.finalstems if len(i['tree'])>30]
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

