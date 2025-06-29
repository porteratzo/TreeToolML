import os

import os

import torch
import treetoolml.layers.Loss_torch as Loss_torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from treetoolml.config.config import combine_cfgs
from treetoolml.data.BatchSampleGenerator_torch import tree_dataset_cloud
from treetoolml.model.build_model import build_model
from treetoolml.utils.default_parser import default_argument_parser
from scipy.optimize import linear_sum_assignment
from treetoolml.IndividualTreeExtraction_utils.center_detection import (
    center_detection,
)
from treetoolml.benchmark.benchmark_utils import store_metrics_detection_only
import pandas as pd
import numpy as np
from collections import defaultdict
from treetoolml.utils.benchmark_util import geometrics
from treetoolml.utils.torch_utils import (
    device_configs,
    load_checkpoint,
    find_training_dir,
    make_xyz_mat,
)
import treetool.seg_tree as seg_tree


torch.backends.cudnn.benchmark = True

torch.manual_seed(123)


def main(args):
    cfg_path = args.cfg
    cfg = combine_cfgs(cfg_path, args.opts)
    use_amp = args.amp != 0

    model_dir = find_training_dir(cfg)
    print(model_dir)
    model = build_model(cfg)
    load_checkpoint(model_dir, model)

    device_configs(model, args)

    test_path = cfg.VALIDATION.PATH
    generator_val = tree_dataset_cloud(
        test_path,
        cfg.TRAIN.N_POINTS,
        normal_filter=True,
        distances=1,
        return_centers=True,
        center_collection_size=16,
        return_scale=True,
    )

    test_loader = DataLoader(
        generator_val, cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=1
    )

    val_loss, loss_dict = testing(model, test_loader, use_amp, cfg)
    if args.centers:
        pd.DataFrame([loss_dict]).to_csv(
            os.path.join(model_dir, "test_results_centers.csv")
        )
    else:
        pd.DataFrame([loss_dict]).to_csv(os.path.join(model_dir, "test_results.csv"))
    print(loss_dict)


def testing(model, generator, use_amp, cfg):
    if next(model.parameters()).is_cuda:
        device = "cuda"

    Nd = cfg.BENCHMARKING.XY_THRESHOLD
    ARe = np.deg2rad(cfg.BENCHMARKING.ANGLE_THRESHOLD)
    voxel_size = cfg.BENCHMARKING.VOXEL_SIZE

    num_batches_testing = len(generator)
    acc_loss = 0
    gen = iter(generator)
    loss_dict = {
        "slackloss": 0,
        "distance_slackloss": 0,
        "distance_loss": 0,
        "trees_found": 0,
        "tree_count": 0,
    }
    loss_dict = defaultdict(list)
    model.eval()
    for n in tqdm(range(num_batches_testing)):
        ###
        (
            batch_test_data,
            batch_direction_label_data,
            batch_object_label,
            batch_centers,
            batch_scales,
        ) = next(gen)
        batch_test_data = batch_test_data.half().to(device)
        batch_direction_label_data = batch_direction_label_data.half().to(device)
        ###
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                y = model(batch_test_data)
                calculate_losses(cfg, acc_loss, loss_dict, batch_direction_label_data, y)

            if n % 15 == 0:
                batch_centers_np = np.array(
                    [i.numpy() for i in batch_centers]
                ).swapaxes(0, 1)
                for n_center, dirs in enumerate(y):
                    xyz_direction = make_xyz_mat(batch_test_data, n_center, dirs)

                    scale = batch_scales[n_center]
                    true_centers = batch_centers_np[n_center]
                    true_centers = [
                        i for i in true_centers if np.all(i != np.array([-1, -1, -1]))
                    ]
                    if cfg.DATA_PREPROCESSING.DISTANCE_FILTER == 0.0:
                        _xyz_direction = xyz_direction
                    else:
                        f_distances = xyz_direction[:, 6]
                        if cfg.MODEL.CLASS_SIGMOID:
                            _xyz_direction = xyz_direction[ f_distances >  cfg.DATA_PREPROCESSING.DISTANCE_FILTER ]
                        else:
                            _xyz_direction = xyz_direction[ (f_distances - np.min(f_distances))
                                / (np.max(f_distances) - np.min(f_distances))
                                < cfg.DATA_PREPROCESSING.DISTANCE_FILTER ]

                    output_dict = geometrics(
                        batch_object_label, n_center, xyz_direction, true_centers, batch_direction_label_data
                    )

                    for key in output_dict.keys():
                        loss_dict[key].append(np.mean(output_dict[key]))


                    if cfg.BENCHMARKING.CENTER_DETECTION_ENABLE:
                        voxel_size = 0.04
                        object_center_list, seppoints = center_detection(
                                _xyz_direction, voxel_size, ARe, Nd
                            )
                    else:
                        tolerence = 0.05
                        seppoints = seg_tree.euclidean_cluster_extract(_xyz_direction[:,:3], tolerance=tolerence, min_cluster_size=10, max_cluster_size=100000)
                        object_center_list = None

                        object_center_list = [np.mean(i,0) for i in seppoints]

                    CostMat = np.ones([len(object_center_list), len(true_centers)])
                    for X, datatree in enumerate(object_center_list):
                        for Y, foundtree in enumerate(true_centers):
                            CostMat[X, Y] = np.linalg.norm(
                                [
                                    datatree[0:2] * scale.numpy()
                                    - foundtree[0:2] * scale.numpy()
                                ]
                            )

                    dataindex, foundindex = linear_sum_assignment(
                        CostMat, maximize=False
                    )
                    metrics = store_metrics_detection_only(
                        object_center_list,
                        true_centers,
                        foundindex,
                        dataindex,
                    )
                    loss_dict["n_ref"].append(metrics["n_ref"])
                    loss_dict["n_match"].append(metrics["n_match"])
                    loss_dict["n_extr"].append(metrics["n_extr"])
                    if metrics["Location_RMSE"] != 100:
                        loss_dict["Location_RMSE"].append(metrics["Location_RMSE"])

    for key in loss_dict.keys():
        
        if key in ["n_ref", "n_match", "n_extr"]:
            loss_dict[key] = np.sum(loss_dict[key])
        else:
            loss_dict[key] = np.mean(loss_dict[key])
    loss_dict["Completeness"] = loss_dict["n_match"] / loss_dict["n_ref"]
    loss_dict["Correctness"] = loss_dict["n_match"] / max(
        (loss_dict["n_extr"] + loss_dict["n_match"]), 1
    )
    return acc_loss / num_batches_testing, loss_dict

def calculate_losses(cfg, acc_loss, loss_dict, batch_direction_label_data, y):
    total_loss = (
                    Loss_torch.slack_based_direction_loss(
                        y,
                        batch_direction_label_data,
                        use_distance=cfg.TRAIN.DISTANCE_LOSS,
                    )
                    .cpu()
                    .numpy()
                )
    loss_dict["slackloss"].append(
                    Loss_torch.slack_based_direction_loss(
                        y, batch_direction_label_data, use_distance=0
                    )
                    .cpu()
                    .numpy()
                )
    loss_dict["distance_slackloss"].append(
                    Loss_torch.slack_based_direction_loss(
                        y, batch_direction_label_data, use_distance=1
                    )
                    .cpu()
                    .numpy()
                )
    loss_dict["distance_loss"].append(
                    Loss_torch.distance_loss(y, batch_direction_label_data)
                    .cpu()
                    .numpy()
                    * 0.2
                )
    loss_dict["distance_slackloss_scaled"].append(
                    Loss_torch.slack_based_direction_loss(
                        y, batch_direction_label_data, use_distance=1, scaled_dist=1
                    )
                    .cpu()
                    .numpy()
                )
    if cfg.TRAIN.DISTANCE:
        total_loss += (
                        Loss_torch.distance_loss(y, batch_direction_label_data)
                        .cpu()
                        .numpy()
                    )
    acc_loss += total_loss




if __name__ == "__main__":
    args = default_argument_parser()
    args.add_argument("--centers", action="store_true", help="save with center append")
    args = args.parse_args()
    main(args)
