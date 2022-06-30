import sys
import os

sys.path.append('.')
sys.path.append("/home/omar/Documents/mine/IndividualTreeExtraction/voxel_region_grow/")
sys.path.append("/home/omar/Documents/mine/TreeTool")
from collections import defaultdict

print(os.getcwd())
import TreeTool.tree_tool as treeTool
import numpy as np
import pclpy
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from TreeToolML.utils.default_parser import default_argument_parser
from TreeToolML.benchmark.benchmark_utils import load_gt, store_metrics, save_eval_results, confusion_metrics
from TreeToolML.config.config import combine_cfgs
from TreeToolML.IndividualTreeExtraction.center_detection.center_detection_vis import (
    center_detection,
)
from TreeToolML.IndividualTreeExtraction.PointwiseDirectionPrediction_torch import (
    prediction,
)
from TreeToolML.model.build_model import build_model
from TreeToolML.utils.file_tracability import find_model_dir, get_checkpoint_file, get_model_dir
from TreeToolML.utils.py_util import (
    combine_IOU,
    data_preprocess,
    get_center_scale,
    shuffle_data,
)
from TreeToolML.utils.benchmark_util import make_metrics_dict
import pickle
from TreeToolML.Libraries.open3dvis import open3dpaint
import matplotlib.pyplot as plt


def main(args):
    cfg_path = args.cfg
    cfg = combine_cfgs(cfg_path, args.opts)
    args.device = 0
    args.gpu_number = 0
    args.amp = 1
    args.opts = []

    device = args.device
    device = "cuda" if device == "gpu" else device
    device = device if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_number)

    Nd = cfg.BENCHMARKING.XY_THRESHOLD
    ARe = np.deg2rad(cfg.BENCHMARKING.COMBINE_STEMS)
    voxel_size = cfg.BENCHMARKING.VOXEL_SIZE

    sample_side_size = cfg.BENCHMARKING.WINDOW_STRIDE
    overlap = cfg.BENCHMARKING.OVERLAP

    model_name = cfg.TRAIN.MODEL_NAME
    model_dir = os.path.join("results_training", model_name)
    model_dir = find_model_dir(model_dir)
    result_dir = get_model_dir(cfg.FILES.RESULT_FOLDER)
    result_dir = os.path.join("results_benchmark", result_dir)
    os.makedirs(result_dir, exist_ok=True)
    result_config_path = os.path.join(result_dir, "full_cfg.yaml")
    cfg_str = cfg.dump()
    with open(result_config_path, "w") as f:
        f.write(cfg_str)
    checkpoint_file = os.path.join(model_dir, "trained_model", "checkpoints")
    checkpoint_path = get_checkpoint_file(checkpoint_file)
    model = build_model(cfg).cuda()
    if device == "cuda":
        model.cuda()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    EvaluationMetrics = make_metrics_dict()
    visualize = False

    confMat_list = []
    result_list = []
    process_dict = []

    for number in tqdm(range(1, 7, 1), desc='forest plot'):
        cloud_file = f"benchmark/subsampled_data/TLS_Benchmarking_Plot_{number}_MS.pcd"
        PointCloud = pclpy.pcl.PointCloud.PointXYZ()
        pclpy.pcl.io.loadPCDFile(cloud_file, PointCloud)

        treetool = treeTool.TreeTool(PointCloud)
        treetool.step_1_remove_floor()
        treetool.step_2_normal_filtering(
            verticality_threshold=0.08, curvature_threshold=0.12, search_radius=0.08
        )

        x1, y1, z1 = np.min(treetool.filtered_points.xyz, 0)
        x2, y2, z2 = np.max(treetool.filtered_points.xyz, 0)

        cropfilter = pclpy.pcl.filters.CropBox.PointXYZ()
        results_dict = defaultdict(dict)
        vis_dict = []
        for n_x, x_start in enumerate(
                tqdm(np.arange(x1, x2, sample_side_size - sample_side_size * overlap), desc='plot row')):
            for n_y, y_start in enumerate(np.arange(y1, y2, sample_side_size - sample_side_size * overlap)):
                cropfilter.setMin(np.array([x_start, y_start, -100, 1.0]))
                cropfilter.setMax(
                    np.array([x_start + sample_side_size, y_start + sample_side_size, 100, 1.0])
                )
                cropfilter.setInputCloud(treetool.filtered_points)
                sub_pcd = pclpy.pcl.PointCloud.PointXYZ()
                cropfilter.filter(sub_pcd)
                cropfilter.setInputCloud(treetool.non_filtered_points)
                sub_pcd_nf = pclpy.pcl.PointCloud.PointXYZ()
                cropfilter.filter(sub_pcd_nf)
                if np.shape(sub_pcd.xyz)[0] > 0:
                    Odata_xyz = data_preprocess(sub_pcd_nf.xyz)
                    data_xyz = data_preprocess(sub_pcd.xyz)
                    data_xyz = shuffle_data(data_xyz)
                    _data_xyz = data_xyz[:4096]
                    center, scale = get_center_scale(sub_pcd.xyz)

                    if np.shape(_data_xyz)[0] >= 4096:
                        nor_testdata = torch.tensor(_data_xyz, device="cuda").squeeze()
                        xyz_direction = prediction(model, nor_testdata, args)
                        if cfg.DATA_PREPROCESSING.DISTANCE_FILTER == 0.0:
                            _xyz_direction = xyz_direction
                        else:
                            _xyz_direction = xyz_direction[xyz_direction[:, 6] < cfg.DATA_PREPROCESSING.DISTANCE_FILTER]

                        object_center_list, seppoints = center_detection(
                            _xyz_direction, voxel_size, ARe, Nd
                        )
                        if visualize:
                            object_center_list_, seppoints_ = center_detection(
                                xyz_direction, voxel_size, ARe, Nd
                            )
                        if len(seppoints) > 0:
                            # open3dpaint([xyz_direction[:, :3].tolist()] + [_xyz_direction[:, :3] + (0.1, 0, 0)] + [i+(1,0,0) for i in seppoints if len(i)>0],
                            if visualize:
                                open3dpaint(
                                    [xyz_direction[:, :3].tolist()] + [(_xyz_direction[:, :3] + (0, 2, 0)).tolist()] + [
                                        i + (4, 0, 0)
                                        for i in
                                        seppoints_ if
                                        len(i) > 0] + [i + (4, 2, 0)
                                                       for i in
                                                       seppoints if
                                                       len(i) > 0],
                                    pointsize=4)

                            seppoints = [i for i in seppoints if np.size(i, 0)]
                            results_dict[n_x][n_y] = {
                                "x": x_start,
                                "y": y_start,
                                "Opoints": Odata_xyz,
                                "Fpoints": data_xyz,
                                "Ipoints": _data_xyz,
                                "centers": object_center_list,
                                "segmentation": seppoints,
                                "center": center,
                                "scale": scale,
                            }
                            result_points = [(i * scale) + center for i in seppoints]
                            vis_dict.extend(result_points)

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
            dp = dbscan(np.array(models), eps=1, min_samples=2)[1]

            _models = np.array(models)[dp == -1].tolist()
            _vis = np.array(vis)[dp == -1].tolist()
            for clust in np.unique(dp):
                if clust == -1:
                    continue
                _models.append(
                    np.array(models)[dp == clust].tolist()[np.argmax([len(i) for i in np.array(models)[dp == clust]])])
                _vis.append(np.vstack(np.array(vis)[dp == clust]).tolist())
            treetool.finalstems = [{'tree': np.array(v), 'model': np.array(m)} for m, v in zip(_models, _vis)]
            # open3dpaint([np.vstack(_vis) + [0.1, 0, 0]] + vis, pointsize=2)
        treetool.step_7_ellipse_fit()
        process_dict.append({'before_IOU': vis_dict, 'after_IOU': vis_dict_})

        TreeDict = load_gt(path=f"benchmark/annotations/TLS_Benchmarking_Plot_{number}_LHD.txt")

        CostMat = np.ones([len(TreeDict), len(treetool.finalstems)])
        for X, datatree in enumerate(TreeDict):
            for Y, foundtree in enumerate(treetool.finalstems):
                CostMat[X, Y] = np.linalg.norm([datatree[0:2] - foundtree["model"][0:2]])

        dataindex, foundindex = linear_sum_assignment(CostMat, maximize=False)

        results_dict['non_filtered_points'] = treetool.non_filtered_points.xyz
        results_dict['filtered_points'] = treetool.filtered_points.xyz
        results_dict['complete_Stems'] = treetool.complete_Stems
        results_dict['finalstems'] = treetool.finalstems
        results_dict['visualization_cylinders'] = treetool.visualization_cylinders

        store_metrics(EvaluationMetrics, treetool, TreeDict, dataindex, foundindex)
        confMat_list.append(confusion_metrics(treetool, TreeDict, dataindex, foundindex))
        result_list.append(results_dict)
    save_eval_results(path=f'{result_dir}/results.npz', EvaluationMetrics=EvaluationMetrics)
    np.savez(f'{result_dir}/confusion_results.npz', confMat_list=confMat_list)
    with open(f'{result_dir}/results_dict.pk', 'wb') as f:
        pickle.dump(result_list, f)
    with open(f'{result_dir}/results_dict_iou.pk', 'wb') as f:
        pickle.dump(process_dict, f)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
