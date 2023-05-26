# %%
import sys
import os
sys.path.append("/home/omar/Documents/mine/IndividualTreeExtraction/voxel_region_grow/")
import TreeTool.seg_tree as seg_tree
import numpy as np
import pclpy
from treetoolml.config.config import combine_cfgs
from treetoolml.utils.file_tracability import find_model_dir
from treetoolml.Libraries.open3dvis import open3dpaint
from treetoolml.utils.benchmark_util import make_metrics_dict
import pickle
import matplotlib.pyplot as plt

# %%
cfg = combine_cfgs("configs/experimentos_combine2/subconfigs/baseline.yaml", [])
model_name = cfg.FILES.RESULT_FOLDER
#model_name = 'treetool'
model_name = 'distance_loss'
if model_name != 'treetool':
    result_dir = os.path.join("results_benchmark", model_name)
    result_dir = find_model_dir(result_dir)
else:
    result_dir = os.path.join('results_benchmark','treetool')
data = np.load(f'{result_dir}/confusion_results.npz', allow_pickle=True)
result_data = np.load(f'{result_dir}/results.npz', allow_pickle=True)
confMat_list = data['confMat_list']
EvaluationMetrics = make_metrics_dict()
for i in EvaluationMetrics.keys():
    EvaluationMetrics[i] = result_data[i]
with open(f'{result_dir}/results_dict.pk','rb') as f:
    dataresult_list = pickle.load(f)
with open(f'{result_dir}/results_dict_iou.pk','rb') as f:
    IOU_data = pickle.load(f)
# %%
plotnumber = 1

cloud_file = f"benchmark/subsampled_data/TLS_Benchmarking_Plot_{plotnumber + 1}_MS.pcd"
PointCloud = pclpy.pcl.PointCloud.PointXYZ()
pclpy.pcl.io.loadPCDFile(cloud_file, PointCloud)
forrest = seg_tree.voxelize(seg_tree.floor_remove(PointCloud)[0])

#%%
for i,j in zip(confMat_list[plotnumber],['tps','fps','fns']):
    print(f'{j}: ', len(i))

#%%
Opoint_result_points = []
Fpoint_result_points = []
Ipoint_result_points = []
segmentation_result_points = []
for plotx in dataresult_list[plotnumber].keys():
    if type(plotx) is int:
        for ploty in dataresult_list[plotnumber][plotx].keys():
            center = dataresult_list[plotnumber][plotx][ploty]['center'] + np.random.random([1,3])/10
            scale = dataresult_list[plotnumber][plotx][ploty]['scale']
            OriginalPoints = dataresult_list[plotnumber][plotx][ploty]['Opoints']
            FilterPoints = dataresult_list[plotnumber][plotx][ploty]["Fpoints"]
            InputPoints = dataresult_list[plotnumber][plotx][ploty]["Ipoints"]
            SegPoints = np.concatenate(dataresult_list[plotnumber][plotx][ploty]["segmentation"])
            Opoint_result_points.append((OriginalPoints * scale) + center)
            Fpoint_result_points.append((FilterPoints * scale) + center)
            Ipoint_result_points.append((InputPoints * scale) + center)
            segmentation_result_points.append((SegPoints * scale) + center)
# %%
print('see true positives')
tp_gt_trees = []
tp_found_trees = []
tp_found_trees_diams = []
tp_found_trees_pos = []
_tp_found_trees = []
for n,record in enumerate(confMat_list[plotnumber][0]):
    if record['true_tree'] is not None:
        tp_gt_trees.append(record['true_tree'])
    if record['found_tree'] is not None:
        tp_found_trees.append(record['found_tree'] )
        _tp_found_trees.append(record['found_tree'])
        tp_found_trees_diams.append(record['true_model'][-1])
        tp_found_trees_pos.append(record['true_model'][:2])

print('see false fp')
fp_found_trees = []
fp_found_trees_diams = []
for record in confMat_list[plotnumber][1]:
    if record['found_tree'] is not None:
        fp_found_trees.append(record['found_tree'])
        fp_found_trees_diams.append(record['found_model'][-1])

print('see false negatives')
fn_found_trees = []
fn_found_trees_diams = []
fn_found_trees_pos = []
for record in confMat_list[plotnumber][2]:
    if record['true_tree'] is not None:
        fn_found_trees.append(record['true_tree'])
        fn_found_trees_diams.append(record['true_model'][-1])
        fn_found_trees_pos.append(record['true_model'][:2])
#%%

open3dpaint(Opoint_result_points, pointsize=2) #how they enter
open3dpaint(Fpoint_result_points, pointsize=2) #after filtering
open3dpaint(Ipoint_result_points, pointsize=2) #input
open3dpaint(segmentation_result_points + tp_gt_trees, pointsize=2) #output
#%%
open3dpaint(Opoint_result_points, pointsize=2) #how they enter
open3dpaint(Fpoint_result_points + fn_found_trees, pointsize=2) #after filtering
open3dpaint(Ipoint_result_points + fn_found_trees, pointsize=2) #input
open3dpaint(segmentation_result_points + fn_found_trees, pointsize=2) #output

#%%
open3dpaint([np.vstack(Fpoint_result_points)] + fp_found_trees, pointsize=2) #after filtering
open3dpaint([np.vstack(Ipoint_result_points)] + fp_found_trees, pointsize=2) #input
open3dpaint([np.vstack(segmentation_result_points)] + fp_found_trees, pointsize=2) #output
#%%
open3dpaint([np.vstack(tp_gt_trees + fn_found_trees)] + [np.vstack(Opoint_result_points)] + fp_found_trees, pointsize=2) #after filtering
#%%
plt.title('hist diameter')
plt.subplot(3,1,1)
plt.title('fn')
plt.hist(fn_found_trees_diams,50)
plt.subplot(3,1,2)
plt.title('tp')
plt.hist(tp_found_trees_diams,50)
plt.subplot(3,1,3)
plt.title('fp')
plt.hist(fp_found_trees_diams,50)
plt.show()
#%%
plt.title('positions')
plt.scatter(np.array(fn_found_trees_pos)[:,0],np.array(fn_found_trees_pos)[:,1])
plt.scatter(np.array(tp_found_trees_pos)[:,0],np.array(tp_found_trees_pos)[:,1])
plt.legend(['FNs','TP'])
plt.show()
#%%

open3dpaint(tp_gt_trees + tp_found_trees, pointsize=2)
#open3dpaint([forrest + [0.2, 0, 0]] + tp_found_trees, pointsize=2)
# open3dpaint([forrest + [0.2, 0, 0]] + fp_found_trees, pointsize=2)
open3dpaint(fp_found_trees, pointsize=2)
open3dpaint(tp_found_trees+fp_found_trees, pointsize=2)
open3dpaint([forrest + [0.2, 0, 0]] + fn_found_trees, pointsize=2)
#open3dpaint(fn_found_trees, pointsize=2)
open3dpaint(tp_found_trees+fn_found_trees, pointsize=2)
open3dpaint(
    [forrest + [0.3, 0, 0]] + [np.vstack(tp_found_trees) + [0.2, 0, 0]] + [np.vstack(fp_found_trees) + [0.1, 0, 0]] + [
        np.vstack(fn_found_trees)], pointsize=2, axis=1)
# %%
