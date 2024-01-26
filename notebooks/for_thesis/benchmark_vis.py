# %%

import sys
from typing import DefaultDict

import os
import argparse
from collections import defaultdict
from porteratzolibs.visualization_o3d.create_geometries import make_arrow, make_cylinder
from treetool.tree_tool import treetool
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
from treetoolml.benchmark.benchmark_utils import get_close_points

# from porteratzolibs.visualization_o3d.open3dvis import open3dpaint, sidexsidepaint
from porteratzolibs.visualization_o3d.open3dvis_new import open3dpaint, sidexsidepaint
from porteratzolibs.visualization_o3d.open3d_pointsetClass import o3d_pointSetClass
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
    close_points_ratio_merge,
    get_close_point_ratio
)
from treetoolml.utils.vis_utils import tree_vis_tool
from scipy.spatial import distance
import timeit
# %%
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

continue_stuff = True
# %%
step_vis = []
# %%
number = 3
gt_trees = load_gt(path=f"benchmark/annotations/TLS_Benchmarking_Plot_{number}_LHD.txt")
cloud_file = f"benchmark/subsampled_data/TLS_Benchmarking_Plot_{number}_MS.pcd"
PointCloud = pcl.PointCloud.PointXYZ()
pcl.io.loadPCDFile(cloud_file, PointCloud)

# %%
my_treetool = treetool(PointCloud)


my_treetool.step_1_remove_floor()
if False:
    tree_vis_tool(
        [my_treetool.non_ground_cloud.xyz, my_treetool.ground_cloud.xyz],
        gt_cylinders=gt_trees,
        for_thesis=True,
        pointsize=4,
    )
if not continue_stuff:
    # %%
    if False:
        good, bad = cloud_match(gt_trees, my_treetool.non_ground_cloud.xyz)
        p = tree_vis_tool(
            {
                "non_ground": my_treetool.non_ground_cloud.xyz,
                "bad": np.vstack(bad.values()),
                "good": np.vstack(good.values()),
            },
            gt_cylinders=gt_trees,
            for_thesis=True,
            pointsize=4,
        )
    # %%

    my_treetool.step_2_normal_filtering(
        #verticality_threshold=0.08, curvature_threshold=0.12,
        #verticality_threshold=0.12, curvature_threshold=0.16,
        verticality_threshold=0.16, curvature_threshold=0.24,
        #verticality_threshold=0.20, curvature_threshold=0.30,
        #verticality_threshold=0.30, curvature_threshold=0.40,
        search_radius=0.1,
    )
    if False:
        tree_vis_tool(
            {
                "non filtered": my_treetool.non_filtered_points.xyz,
                "filtered": my_treetool.filtered_points.xyz,
            },
            gt_cylinders=gt_trees,
            for_thesis=True,
            pointsize=4,
        )
    if False:
        good, bad = cloud_match(gt_trees, my_treetool.filtered_points.xyz, 0.2, 50)
        tree_vis_tool(
            {
                "all_points": my_treetool.filtered_points.xyz,
                "all_points2": my_treetool.non_filtered_points.xyz,
                "bad": np.vstack(bad.values()),
                "good": np.vstack(good.values()),
            },
            gt_cylinders=gt_trees,
            for_thesis=True,
            pointsize=4,
        )
    # %%

    results_dict = defaultdict(dict)
    pointcloud_cluster = []
    if True:
        cfg.defrost()
        #cfg.DATA_PREPROCESSING.DISTANCE_FILTER = 0.1
        cfg.DATA_PREPROCESSING.DISTANCE_FILTER = 0.6
        cfg.BENCHMARKING.CENTER_DETECTION_ENABLE = 0
    sample_side_size = 4
    overlap = 0.4

    generator = sample_generator(sample_side_size, overlap, my_treetool)
    run_detection(args, cfg, model, results_dict, pointcloud_cluster, generator, use_non_filtered=False, tolerence=0.25)
    tp, fp, fn = cloud_match_clusters(gt_trees, pointcloud_cluster, mean=False, tp_criteria=10)
    print_metrics('initial detection',tp, fp, fn)
    if False:
        tp_clouds, fp_clouds, fn_clouds = metrics_2_clouds(gt_trees, pointcloud_cluster, tp, fp, fn)
        tree_vis_tool(
            {
                "filtred_points": my_treetool.filtered_points.xyz,
                "cluster_points": pointcloud_cluster,
                "tp": tp_clouds,
                "fp": fp_clouds,
                "fn": fn_clouds,
            },
            gt_cylinders=gt_trees,
            for_thesis=False,
            pointsize=6,
        )

    if cfg.BENCHMARKING.COMBINE_IOU:
        vis_dict_ = combine_IOU(pointcloud_cluster)
    else:
        vis_dict_ = pointcloud_cluster

    def combine_close_points(vis_dict_, distance=0.05, ratio=0.6):
        new_clust = vis_dict_.copy()
        centers = [np.mean(i,0)[:2] for i in new_clust]
        for treenumber1 in reversed(range(0, len(new_clust))):
            for treenumber2 in reversed(range(0, treenumber1)):
                center1 = centers[treenumber1]
                center2 = centers[treenumber2]
                if np.linalg.norm(center1-center2) < 1:
                    points1 = vis_dict_[treenumber1]
                    points2 = vis_dict_[treenumber2]
                    if get_close_point_ratio(points1,points2, distance) > ratio:
                        new_clust[treenumber2] = np.vstack(
                            [new_clust[treenumber2], new_clust.pop(treenumber1)]
                        )
                        break
        return new_clust

    close_in = vis_dict_
    while True:
        close_out = combine_close_points(close_in, 0.1, 0.4)
        print('closepoints input:',len(close_in),' out:', len(close_out))
        if len(close_in) == len(close_out):
            break
        close_in = close_out
        tp, fp, fn = cloud_match_clusters(gt_trees, close_in, mean=False, tp_criteria=10)
        print_metrics('combine close',tp, fp, fn)
        

    if False:
        tp_clouds, fp_clouds, fn_clouds = metrics_2_clouds(gt_trees, pointcloud_cluster, tp, fp, fn)

        tree_vis_tool(
            {
                "filtred_points": my_treetool.filtered_points.xyz,
                "cluster_points": new_clust,
                "tp": tp_clouds,
                "fp": fp_clouds,
                "fn": fn_clouds,
            },
            gt_cylinders=gt_trees,
            for_thesis=False,
            pointsize=6,
        )

    if True:
        g_stem_in = close_out
        while True:
            my_treetool.cluster_list = g_stem_in
            my_treetool.step_4_group_stems(0.2)
            print('stem groups input:',len(g_stem_in),' out:', len(my_treetool.complete_Stems))
            if len(g_stem_in) == len(my_treetool.complete_Stems):
                break
            g_stem_in = my_treetool.complete_Stems
            tp, fp, fn = cloud_match_clusters(gt_trees, my_treetool.complete_Stems, mean=False, tp_criteria=10)
            print_metrics('group stems',tp, fp, fn)
    else:
        my_treetool.complete_Stems = vis_dict_

    tp, fp, fn = cloud_match_clusters(gt_trees, my_treetool.complete_Stems, mean=False, tp_criteria=10)
    print_metrics('group stems',tp, fp, fn)
    if False:
        tp_clouds, fp_clouds, fn_clouds = metrics_2_clouds(gt_trees, pointcloud_cluster, tp, fp, fn)

        tree_vis_tool(
            {
                "filtred_points": my_treetool.filtered_points.xyz,
                "cluster_points": complete_2,
                "tp": tp_clouds,
                "fp": fp_clouds,
                "fn": fn_clouds,
            },
            gt_cylinders=gt_trees,
            for_thesis=False,
            pointsize=6,
        )

    if False:
        _cloud = [i['cloud'] for i in my_treetool.stem_groups]
        _straightness = [i['straightness'] for i in my_treetool.stem_groups]
        _center = [i['center'] for i in my_treetool.stem_groups]
        _direction = [i['direction'][0] for i in my_treetool.stem_groups]
        tree_vis_tool(_cloud,centers=_center,vectors=np.hstack([_direction,_center]))
        

    new_pointcloud_cluster = []
    sub_pcd = pclpy.pcl.PointCloud.PointXYZ()
    cropfilter = pclpy.pcl.filters.CropBox.PointXYZ()
    cropfilter.setInputCloud(my_treetool.non_filtered_points)
    for clust in tqdm(my_treetool.complete_Stems):
        _min, _max =  np.min(clust,axis=0)-1, np.max(clust,axis=0)+1
        cropfilter.setMin(np.hstack([_min,1]))
        cropfilter.setMax(np.hstack([_max,1]))
        cropfilter.filter(sub_pcd)
        _idx = get_close_points(clust,sub_pcd.xyz, 0.02)
        new_pointcloud_cluster.append(sub_pcd.xyz[_idx])

    if False:
        tp, fp, fn = cloud_match_clusters(gt_trees, new_pointcloud_cluster, mean=False, tp_criteria=10)
        print_metrics(tp, fp, fn)
        tp_clouds, fp_clouds, fn_clouds = metrics_2_clouds(gt_trees, pointcloud_cluster, tp, fp, fn)

        tree_vis_tool(
            {
                "filtred_points": my_treetool.filtered_points.xyz,
                "cluster_points": complete_2,
                "tp": tp_clouds,
                "fp": fp_clouds,
                "fn": fn_clouds,
            },
            gt_cylinders=gt_trees,
            for_thesis=False,
            pointsize=6,
        )
if False:
    np.save('misc_data/vis_Data.npy',np.array(new_pointcloud_cluster,dtype=object))
if True:
    new_pointcloud_cluster = np.load('misc_data/vis_Data.npy', allow_pickle=True)
my_treetool.complete_Stems = new_pointcloud_cluster
my_treetool.step_5_get_ground_level_trees(2,4,True,True) ### find better way
tp, fp, fn = cloud_match_clusters(gt_trees, my_treetool.low_stems, mean=False, tp_criteria=10)
print_metrics('cut trees',tp, fp, fn)
if False:
    tp_clouds, fp_clouds, fn_clouds = metrics_2_clouds(gt_trees, my_treetool.low_stems, tp, fp, fn)

    tree_vis_tool(
        {
            "cluster_points": my_treetool.complete_Stems,
            "low_stems": my_treetool.low_stems,
            "tp": tp_clouds,
            "fp": fp_clouds,
            "fn": fn_clouds,
        },
        gt_cylinders=gt_trees,
        for_thesis=False,
        pointsize=6,
    )
my_treetool.step_6_get_cylinder_tree_models(stick=False,distance=0.06)
stems_ = [i["tree"] for i in my_treetool.finalstems if len(i['tree'])>0]
tp, fp, fn = cloud_match_clusters(gt_trees, stems_, mean=False, tp_criteria=10)
print_metrics('cylinder models',tp, fp, fn)
if False:
    tp_clouds, fp_clouds, fn_clouds = metrics_2_clouds(gt_trees, stems_, tp, fp, fn)

    tree_vis_tool(
        {
            "filtred_points": new_pointcloud_cluster,
            'low_stems':my_treetool.low_stems,
            "cluster_points": stems_,
            "tp": tp_clouds,
            "fp": fp_clouds,
            "fn": fn_clouds,
        },
        gt_cylinders=gt_trees,
        for_thesis=False,
        pointsize=6,
    )

# remove singles
if False:
    cstems = [i["tree"] for i in my_treetool.finalstems if len(i['tree'])>0]
    singles_point_cluster = []
    for clust in my_treetool.finalstems:
        _min, _max =  np.min(clust['tree'],axis=0), np.max(clust['tree'],axis=0)
        if max(_max - _min) > 1:
            singles_point_cluster.append(clust)
    cstems2 = [i['tree'] for i in singles_point_cluster]
    my_treetool.finalstems = singles_point_cluster
    tp, fp, fn = cloud_match_clusters(gt_trees, cstems2, mean=False, tp_criteria=10)
    print_metrics('remove singles',tp, fp, fn)

# height_removal
if True:
    h_cstems = [i["tree"] for i in my_treetool.finalstems if len(i['tree'])>0]
    before_hight = my_treetool.finalstems
    h_singles_point_cluster = []
    for clust in my_treetool.finalstems:
        _min, _max =  clust['ground'], np.max(clust['tree'][:,2],axis=0)
        if _max - _min > 4:
            h_singles_point_cluster.append(clust)
    H_cstems2 = [i['tree'] for i in h_singles_point_cluster]

    tp, fp, fn = cloud_match_clusters(gt_trees, H_cstems2, mean=False, tp_criteria=10)
    print_metrics('height filter',tp, fp, fn)

    my_treetool.finalstems = h_singles_point_cluster

#tp_heights = [np.max(cstems2[i[1]][:,2])-np.min(H_cstems2[i[1]][:,2]) for i in tp]
#fp_heights = [np.max(cstems2[i][:,2])-np.min(H_cstems2[i][:,2]) for i in fp]

tp_heights = [np.max(my_treetool.finalstems[i[1]]['tree'][:,2])-my_treetool.finalstems[i[1]]['ground'] for i in tp]
fp_heights = [np.max(my_treetool.finalstems[i]['tree'][:,2])-my_treetool.finalstems[i]['ground'] for i in fp]
if False:
    import matplotlib.pyplot as plt
    plt.hist(tp_heights,30, alpha=0.5)
    plt.hist(fp_heights,30, alpha=0.5)
if False:

    tp_clouds, fp_clouds, fn_clouds = metrics_2_clouds(gt_trees, H_cstems2, tp, fp, fn)

    tree_vis_tool(
        {
            "filtred_points": new_pointcloud_cluster,
            'before':cstems,
            "after": cstems2,
            "tp": tp_clouds,
            "fp": fp_clouds,
            "fn": fn_clouds,
        },
        gt_cylinders=gt_trees,
        for_thesis=False,
        pointsize=6,
    )
if False:
    old_stems = my_treetool.finalstems
    my_treetool.finalstems = old_stems
    run_combine_stems(cfg, my_treetool)
    finalstems = [i for i in my_treetool.finalstems if len(i['tree'])>30]
    new_stems_ = [i["tree"] for i in finalstems]

    tp, fp, fn = cloud_match_clusters(gt_trees, new_stems_, mean=False, tp_criteria=10)
    print_metrics('combine stems',tp, fp, fn)

if False:
    tp_clouds, fp_clouds, fn_clouds = metrics_2_clouds(gt_trees, old_stems_, tp, fp, fn)
    tree_vis_tool(
        {
            "filtred_points": new_pointcloud_cluster,
            "cluster_points": stems_,
            'comb':old_stems_,
            "tp": tp_clouds,
            "fp": fp_clouds,
            "fn": fn_clouds,
        },
        gt_cylinders=gt_trees,
        for_thesis=False,
        pointsize=6,
    )
my_treetool.step_7_ellipse_fit(-1,4)
fin_stems = [i['tree'] for i in my_treetool.finalstems if i['final_diameter'] > 0.05]
tp, fp, fn = cloud_match_clusters(gt_trees, fin_stems, mean=False, )
print_metrics('ellipsefit',tp, fp, fn)

tp_diam = np.array([my_treetool.finalstems[i[1]]['final_diameter'] for i in tp])
fp_diam = np.array([my_treetool.finalstems[i]['final_diameter'] for i in fp])
tp_trees = np.array([my_treetool.finalstems[i[1]]['tree'] for i in tp],dtype=object)

tp_diam_error = np.array([abs(my_treetool.finalstems[i[1]]['final_diameter']-gt_trees[i[0]][2]) for i in tp])
if True:
    import matplotlib.pyplot as plt
    plt.hist(tp_diam[tp_diam_error>0.1],30, alpha=1)
    plt.hist(tp_diam[tp_diam_error<0.1],30, alpha=0.7)
    plt.hist(fp_diam,30, alpha=0.3)
tp_clouds, fp_clouds, fn_clouds = metrics_2_clouds(gt_trees, fin_stems, tp, fp, fn)
if False:
    tree_vis_tool(
            {
                #"non":my_treetool.non_filtered_points.xyz,
                "filtred_points": my_treetool.filtered_points.xyz,
                "cluster_points": pointcloud_cluster,
                "stems": fin_stems,
                "wrong_tp": np.vstack(tp_trees[tp_diam_error>0.2]),
                "right_tp": np.vstack(tp_trees[tp_diam_error<0.2]),
                "tp": tp_clouds,
                "fp": fp_clouds,
                "fn": fn_clouds,
            },
            gt_cylinders=gt_trees,
            for_thesis=False,
            pointsize=6,
            axis=0
        )
    print('')
if False:
    tree_vis_tool(
            {
                #"non":my_treetool.non_filtered_points.xyz,
                "cluster_points": new_pointcloud_cluster,
                "stems_$": stems_,
                "tp": tp_clouds,
                "fp": fp_clouds,
                "fn": fn_clouds,
            },
            gt_cylinders=gt_trees,
            for_thesis=False,
            pointsize=6,
            axis=0
        )
    
    old_tp, old_fp, old_fn = cloud_match_clusters(gt_trees, old_stems_, mean=False,)
    print_metrics(old_tp, old_fp, old_fn)
    old_tp_clouds, old_fp_clouds, old_fn_clouds = metrics_2_clouds(gt_trees, old_stems_, old_tp, old_fp, old_fn)
    if False:
        tree_vis_tool(
                {
                    "tp": tp_clouds,
                    "fp": fp_clouds,
                    "fn": fn_clouds,

                    "old_tp": old_tp_clouds,
                    "old_fp": old_fp_clouds,
                    "old_fn": old_fn_clouds,
                },
                gt_cylinders=gt_trees,
                for_thesis=False,
                pointsize=6,
            )

    new_stems = close_points_ratio_merge(old_stems_, min_dist=0.2)
    ld_tp, ld_fp, ld_fn = cloud_match_clusters(gt_trees, new_stems, mean=False)
    print_metrics(ld_tp, ld_fp, ld_fn)
    new_stems = close_points_ratio_merge(new_stems, min_dist=0.2)
    ld_tp, ld_fp, ld_fn = cloud_match_clusters(gt_trees, new_stems, mean=False)
    print_metrics(ld_tp, ld_fp, ld_fn)
    ld_tp_clouds, ld_fp_clouds, ld_fn_clouds = metrics_2_clouds(gt_trees, new_stems, ld_tp, ld_fp, ld_fn)
    if True:
        tree_vis_tool(
                {
                    'stems': stems_,
                    "tp": tp_clouds,
                    "fp": fp_clouds,
                    "fn": fn_clouds,
                    "new_stems":new_stems,
                    "old_tp": ld_tp_clouds,
                    "old_fp": ld_fp_clouds,
                    "old_fn": ld_fn_clouds,
                },
                gt_cylinders=gt_trees,
                for_thesis=False,
                pointsize=6,
            )


quit()

# %%
open3dpaint([np.vstack(i) for i in vis_dict2], pointsize=2, for_thesis=True)
# %%
