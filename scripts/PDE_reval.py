import os

import os
import re
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
from porteratzo3D.geometry_utils import angle_between_two_vectors, dist_point_to_line
from treetoolml.utils.file_tracability import get_checkpoint_list
from treetoolml.utils.torch_utils import device_configs, load_checkpoint, find_training_dir


torch.backends.cudnn.benchmark = True

torch.manual_seed(123)


def main(args):
    cfg_path = args.cfg
    cfg = combine_cfgs(cfg_path, args.opts)
    use_amp = args.amp != 0

    model_dir = find_training_dir(cfg)
    print(model_dir)
    model = build_model(cfg)

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

    test_loader = DataLoader(generator_val, cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=1)
    checkpoint_file = os.path.join(model_dir, "trained_model", "checkpoints")
    cp_files, val_acc, file_mod_time = get_checkpoint_list(
        checkpoint_file,
    )
    cp_files = np.array(cp_files)[np.argsort(file_mod_time)]

    saved_list = []
    for checkpoint_path in tqdm(cp_files):
        checkpoint = torch.load(os.path.join(checkpoint_file, checkpoint_path))
        pattern = r"model-(\d+)-train_loss"
        iter_count = np.array(int(re.search(pattern, checkpoint_path).group(1)))
        model.load_state_dict(checkpoint["model_state_dict"])
        val_loss, loss_dict = testing(model, test_loader, use_amp, cfg)
        loss_dict["epoch"] = iter_count
        saved_list.append(loss_dict)
    pd.DataFrame(saved_list).to_csv(os.path.join(model_dir, "reval_results.csv"))
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
    for n in range(num_batches_testing):
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
                    Loss_torch.distance_loss(y, batch_direction_label_data).cpu().numpy() * 0.2
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
                        Loss_torch.distance_loss(y, batch_direction_label_data).cpu().numpy()
                    )
                acc_loss += total_loss

    for key in [
        "slackloss",
        "distance_slackloss",
        "distance_loss",
        "distance_slackloss_scaled",
    ]:
        loss_dict[key] = np.mean(loss_dict[key])
    return acc_loss / num_batches_testing, loss_dict


def make_xyz_mat(batch_test_data, n_center, dirs):
    pde_ = np.transpose(dirs.cpu().numpy(), [1, 0])
    testdata = batch_test_data[n_center].cpu().numpy()
    xyz_direction = np.concatenate([testdata, pde_], -1).astype(np.float32)

    return xyz_direction


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
