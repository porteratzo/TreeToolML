# %%
import os
from collections import defaultdict

import treetool.tree_tool as treeTool
import numpy as np
import pclpy
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from treetoolml.utils.default_parser import default_argument_parser
from treetoolml.benchmark.benchmark_utils import load_gt, store_metrics, save_eval_results, confusion_metrics
from treetoolml.config.config import combine_cfgs
from treetoolml.IndividualTreeExtraction.center_detection.center_detection_vis import (
    center_detection,
)
from treetoolml.IndividualTreeExtraction.PointwiseDirectionPrediction_torch import (
    prediction,
)
from treetoolml.model.build_model import build_model
from treetoolml.utils.file_tracability import find_model_dir, get_checkpoint_file, get_model_dir
from treetoolml.utils.py_util import (
    combine_IOU,
    data_preprocess,
    get_center_scale,
    shuffle_data,
)
from treetoolml.utils.benchmark_util import make_metrics_dict
import pickle
from treetoolml.Libraries.open3dvis import open3dpaint
import ray
ray.init()
# %%
    
@ray.remote
def getMetrics_remote(Nd):
    return getMetrics(Nd)

def getMetrics(Nd):
    print('using ',Nd)
    cfg = combine_cfgs("configs/experimentos_distancefilter/subconfigs/distance_loss.yaml", [])

    model_name = cfg.FILES.RESULT_FOLDER
    model_name = 'distance_loss'
    if model_name != 'treetool':
        result_dir = os.path.join("results_benchmark", model_name)
        result_dir = find_model_dir(result_dir)
    else:
        result_dir = os.path.join('results_benchmark','treetool')
    with open(f'{result_dir}/results_dict.pk','rb') as f:
        dataresult_list = pickle.load(f)

    voxel_size = cfg.BENCHMARKING.VOXEL_SIZE
    #Nd = cfg.BENCHMARKING.XY_THRESHOLD
    ARe = np.deg2rad(cfg.BENCHMARKING.ANGLE_THRESHOLD)

    metrics_dict = {}
    EvaluationMetrics = make_metrics_dict()
    for number in range(1, 7, 1):
        cloud_file = f"benchmark/subsampled_data/TLS_Benchmarking_Plot_{number}_MS.pcd"
        PointCloud = pclpy.pcl.PointCloud.PointXYZ()
        pclpy.pcl.io.loadPCDFile(cloud_file, PointCloud)
    
        treetool = treeTool.treetool(PointCloud)
        vis_dict = []
        treetool.step_1_remove_floor()
        x_positions = [i for i in dataresult_list[number-1].keys() if isinstance(i,int)]
        for n_x in x_positions:
            y_positions = [i for i in dataresult_list[number-1][n_x].keys() if isinstance(i,int)]
            for n_y in y_positions:
                prediction = dataresult_list[number-1][n_x][n_y]['prediction']

                object_center_list, seppoints = center_detection(
                            prediction, voxel_size, ARe, Nd
                        )
            
                if len(seppoints) > 0:
                    seppoints = [i for i in seppoints if np.size(i, 0)]
                    scale = dataresult_list[number-1][n_x][n_y]['scale']
                    center = dataresult_list[number-1][n_x][n_y]['center']
                    result_points = [(i * scale) + center for i in seppoints]
                    vis_dict.extend(result_points)

        if cfg.BENCHMARKING.GROUP_STEMS:
            treetool.cluster_list = vis_dict
            treetool.step_4_group_stems()
        
        if cfg.BENCHMARKING.COMBINE_IOU:
            vis_dict_ = combine_IOU(vis_dict)
        else:
            vis_dict_ = vis_dict

        treetool.complete_Stems = vis_dict_
        treetool.step_5_get_ground_level_trees()
        treetool.step_6_get_cylinder_tree_models()

        if cfg.BENCHMARKING.COMBINE_STEMS:
            from sklearn.cluster import dbscan
            models = [i['model'] for i in treetool.finalstems]
            vis = [i['tree'] for i in treetool.finalstems]
            if len(models) > 0:
                dp = dbscan(np.array(models), eps=1, min_samples=2)[1]

                _models = np.array(models)[dp == -1].tolist()
                _vis = np.array(vis, dtype=object)[dp == -1].tolist()
                for clust in np.unique(dp):
                    if clust == -1:
                        continue
                    _models.append(
                    np.array(models)[dp == clust].tolist()[np.argmax([len(i) for i in np.array(vis, dtype=object)[dp == clust]])])
                    _vis.append(np.vstack(np.array(vis, dtype=object)[dp == clust]).tolist())
                treetool.finalstems = [{'tree': np.array(v), 'model': np.array(m)} for m, v in zip(_models, _vis)]
            else:
                treetool.finalstems
        # open3dpaint([np.vstack(_vis) + [0.1, 0, 0]] + vis, pointsize=2)
        treetool.step_7_ellipse_fit()

        TreeDict = load_gt(path=f"benchmark/annotations/TLS_Benchmarking_Plot_{number}_LHD.txt")

        CostMat = np.ones([len(TreeDict), len(treetool.finalstems)])
        for X, datatree in enumerate(TreeDict):
            for Y, foundtree in enumerate(treetool.finalstems):
                CostMat[X, Y] = np.linalg.norm([datatree[0:2] - foundtree["model"][0:2]])

        dataindex, foundindex = linear_sum_assignment(CostMat, maximize=False)
        store_metrics(EvaluationMetrics, treetool, TreeDict, dataindex, foundindex)
    metrics_dict[Nd] = EvaluationMetrics
    return metrics_dict

#getMetrics(32000)
#quit()
Nd_list = [1, 20, 50, 100, 1000,2000, 4000, 8000, 12000, 16000,32000]
metrics_dict = {}
refs = []
for nd in tqdm(Nd_list, desc='nd list'):
    refs.append(getMetrics_remote.remote(nd))

#%%
for i in tqdm(refs):
    metrics_dict.update(ray.get(i))

# %%
#completeness
#this_slice = slice(0,2)
#this_slice = slice(2,4)
Nd_list = [1, 20, 50, 100, 1000,2000, 4000, 8000, 12000, 16000,32000]
values_p = {}
values_r = {}
for n,level in enumerate(['easy','mid','hard']):
    this_slice = [slice(0,2),slice(2,4),slice(4,6)][n]
    values_p[level] = []
    values_r[level] = []
    for nd in Nd_list:
        val = [i for i in metrics_dict[nd]['Completeness'][this_slice] if i != 0]
        mean_val = np.mean(val)
        values_r[level].append(mean_val)

        val = [i for i in metrics_dict[nd]['Correctness'][this_slice] if i != 0]
        mean_val = np.mean(val)
        values_p[level].append(mean_val)
        

# %%

import matplotlib.pyplot as plt

for n,level in enumerate(['easy','mid','hard']):
    fig, axs = plt.subplots(1, 1, figsize=(15, 5))
    p = values_p[level]
    r = values_r[level]
    axs.plot(r, p, marker='o')
    axs.set_xlabel('Recall')
    axs.set_ylabel('Precision')
    axs.set_title(level + ' precision-recall')
    axs.set_xlim([0, 1])
    axs.set_ylim([0, 1])
    axs.grid(True)
    plt.savefig(f"{level + ' precision-recall'}.png")
# %%
if False:
    import pickle
    with open(f'buff_results_dict_p.pk', 'wb') as f:
        pickle.dump(values_p, f)

    with open(f'buff_results_dict_r.pk', 'wb') as f:
        pickle.dump(values_r, f)
# %%
import pickle
with open(f'misc_data/buff_results_dict_p.pk', 'rb') as f:
    values_p = pickle.load(f)

with open(f'misc_data/buff_results_dict_r.pk', 'rb') as f:
    values_r = pickle.load(f)

# %%
