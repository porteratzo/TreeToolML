import os
import pickle
from collections import defaultdict

import numpy as np
import open3d as o3d
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
    get_close_points,
    combine_close_points,
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

    confMat_list = []
    result_list = []
    process_dict = []

    for number in tqdm(range(1, 7, 1), desc="forest plot"):
        cloud_file = f"benchmark/subsampled_data/TLS_Benchmarking_Plot_{number}_MS.pcd"
        PointCloud = o3d.io.read_point_cloud(cloud_file)

        treetool = treeTool.treetool(PointCloud)
        treetool.step_1_remove_floor()
        treetool.step_2_normal_filtering(
            verticality_threshold=cfg.BENCHMARKING.VERTICALITY,
            curvature_threshold=cfg.BENCHMARKING.CURVATURE,
            search_radius=0.08,
        )

        results_dict = defaultdict(dict)
        vis_dict = []

        sample_side_size = cfg.BENCHMARKING.WINDOW_STRIDE
        overlap = cfg.BENCHMARKING.OVERLAP
        generator = sample_generator(sample_side_size, overlap, treetool)

        run_detection(
            args,
            cfg,
            model,
            results_dict,
            vis_dict,
            generator,
            use_non_filtered=False,
            tolerence=0.25,
        )

        if cfg.BENCHMARKING.COMBINE_IOU:
            vis_dict_ = combine_IOU(vis_dict)
        else:
            vis_dict_ = vis_dict

        close_in = vis_dict_
        while True:
            close_out = combine_close_points(close_in, 0.2, 0.4)
            print("closepoints input:", len(close_in), " out:", len(close_out))
            if len(close_in) == len(close_out):
                break
            close_in = close_out

        if cfg.BENCHMARKING.GROUP_STEMS:
            g_stem_in = close_out
            while True:
                treetool.cluster_list = g_stem_in
                treetool.step_4_group_stems(0.3)
                print("stem groups input:", len(g_stem_in), " out:", len(treetool.complete_Stems))
                if len(g_stem_in) == len(treetool.complete_Stems):
                    break
                g_stem_in = treetool.complete_Stems

        new_pointcloud_cluster = []
        for clust in tqdm(treetool.complete_Stems):
            _min, _max = np.min(clust, axis=0) - 1, np.max(clust, axis=0) + 1
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=_min, max_bound=_max)
            non_filtered_points = o3d.geometry.PointCloud()
            non_filtered_points.points = o3d.utility.Vector3dVector(treetool.non_filtered_points)
            cropped_pcd = non_filtered_points.crop(bbox)
            _idx = get_close_points(clust, np.asarray(cropped_pcd.points), 0.02)
            new_pointcloud_cluster.append(np.asarray(cropped_pcd.points)[_idx])

        treetool.complete_Stems = new_pointcloud_cluster
        treetool.step_5_get_ground_level_trees(2, 4, True, True)
        treetool.step_6_get_cylinder_tree_models(distance_threshold=0.06)

        run_combine_stems(cfg, treetool)
        # open3dpaint([np.vstack(_vis) + [0.1, 0, 0]] + vis, pointsize=2)

        h_singles_point_cluster = []
        for clust in treetool.finalstems:
            _min, _max = clust["ground"], np.max(clust["tree"][:, 2], axis=0)
            if _max - _min > 6:
                h_singles_point_cluster.append(clust)

        treetool.finalstems = h_singles_point_cluster

        treetool.step_7_ellipse_fit(-1, 3)
        treetool.finalstems = [i for i in treetool.finalstems if i["final_diameter"] > 0.05]
        process_dict.append({"before_IOU": vis_dict, "after_IOU": vis_dict_})

        TreeDict = load_gt(path=f"benchmark/annotations/TLS_Benchmarking_Plot_{number}_LHD.txt")

        dataindex, foundindex = matching(treetool, TreeDict)

        results_dict["non_filtered_points"] = treetool.non_filtered_points
        results_dict["filtered_points"] = treetool.filtered_points
        results_dict["complete_Stems"] = treetool.complete_Stems
        results_dict["finalstems"] = treetool.finalstems
        results_dict["visualization_cylinders"] = treetool.visualization_cylinders

        store_metrics(EvaluationMetrics, treetool, TreeDict, dataindex, foundindex)
        confMat_list.append(confusion_metrics(treetool, TreeDict, dataindex, foundindex))
        result_list.append(results_dict)
    save_eval_results(path=f"{result_dir}/results.npz", EvaluationMetrics=EvaluationMetrics)
    np.savez(
        f"{result_dir}/confusion_results.npz",
        confMat_list=np.array(confMat_list, dtype=object),
    )
    with open(f"{result_dir}/results_dict.pk", "wb") as f:
        pickle.dump(result_list, f)
    with open(f"{result_dir}/results_dict_iou.pk", "wb") as f:
        pickle.dump(process_dict, f)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
