import os

import os

import torch
from treetoolml.utils.tictoc import bench_dict, g_timer1
import treetoolml.layers.Loss_torch as Loss_torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from treetoolml.config.config import combine_cfgs
from treetoolml.data.BatchSampleGenerator_torch import tree_dataset_cloud
from treetoolml.model.build_model import build_model
from treetoolml.utils.default_parser import default_argument_parser
from treetoolml.utils.file_tracability import get_checkpoint_file, find_model_dir
from scipy.optimize import linear_sum_assignment
from treetoolml.IndividualTreeExtraction.center_detection.center_detection_vis import (
    center_detection,
)
from treetoolml.benchmark.benchmark_utils import store_metrics_detection_only
import pandas as pd
import numpy as np
from collections import defaultdict

torch.backends.cudnn.benchmark = True

torch.manual_seed(123)


class start_bench:
    def __init__(self, dataloader) -> None:
        self.dataloader = dataloader

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        self.iter_obj = iter(self.dataloader)
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self.dataloader):
            bench_dict["epoch"].gstep()
            self.n += 1
            while True:
                try:
                    result = next(self.iter_obj)
                    break
                except:
                    raise
                    self.n += 1
                    print("error")
            return result
        else:
            raise StopIteration


def main(args):
    cfg_path = args.cfg
    cfg = combine_cfgs(cfg_path, args.opts)
    args.device = "gpu"
    args.gpu_number = 0
    args.amp = 1
    args.opts = []
    use_amp = args.amp != 0

    device = args.device
    device = "cuda" if device == "gpu" else device
    device = device if torch.cuda.is_available() else "cpu"

    model_name = cfg.TRAIN.MODEL_NAME
    model_dir = os.path.join("results_training", model_name)
    model_dir = find_model_dir(model_dir)
    print(model_dir)
    checkpoint_file = os.path.join(model_dir, "trained_model", "checkpoints")
    checkpoint_path = get_checkpoint_file(checkpoint_file, 'BEST')
    model = build_model(cfg).cuda()
    if device == "cuda":
        model.cuda()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    # summary(model, (4096, 3), device=device)

    # datapath = os.path.join(cfg.FILES.DATA_SET, cfg.FILES.DATA_WORK_FOLDER)
    # test_path = os.path.join(datapath, "testing_data")
    test_path = cfg.VALIDATION.PATH
    generator_val = tree_dataset_cloud(
        test_path,
        cfg.TRAIN.N_POINTS,
        normal_filter=True,
        distances=1,
        return_centers=True,
        center_collection_size=16,
        return_scale=True
    )

    test_loader = DataLoader(
        generator_val, cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=1
    )

    val_loss, loss_dict = testing(model, test_loader, use_amp, cfg)
    pd.DataFrame([loss_dict]).to_csv(os.path.join(model_dir, "test_results.csv"))
    print(loss_dict)


def device_configs(DeepPointwiseDirections, args):
    device = args.device
    device = "cuda" if device == "gpu" else device
    device = device if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_number)
    if device == "cuda":
        DeepPointwiseDirections.cuda()
    return device


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
            batch_scales
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
                    Loss_torch.distance_loss(y, batch_direction_label_data)
                    .cpu()
                    .numpy()
                )
                if cfg.TRAIN.DISTANCE:
                    total_loss += Loss_torch.distance_loss(
                        y, batch_direction_label_data
                    ).cpu().numpy()
                acc_loss += total_loss

            if n % 25 == 0:
                batch_centers_np = np.array(
                    [i.numpy() for i in batch_centers]
                ).swapaxes(0, 1)
                for n_center, dirs in enumerate(y):
                    scale = batch_scales[n_center]
                    pde_ = np.transpose(dirs.cpu().numpy(), [1, 0])
                    testdata = batch_test_data[n_center].cpu().numpy()
                    xyz_direction = np.concatenate([testdata, pde_], -1).astype(
                        np.float32
                    )
                    true_centers = batch_centers_np[n_center]
                    true_centers = [
                        i for i in true_centers if np.all(i != np.array([-1, -1, -1]))
                    ]
                    if cfg.DATA_PREPROCESSING.DISTANCE_FILTER == 0.0:
                        _xyz_direction = xyz_direction
                    else:
                        _xyz_direction = xyz_direction[
                            xyz_direction[:, 6] < cfg.DATA_PREPROCESSING.DISTANCE_FILTER
                        ]
                    
                    voxel_size = 0.04
                    object_center_list, seppoints = center_detection(
                        _xyz_direction, voxel_size, ARe, Nd
                    )

                    CostMat = np.ones([len(object_center_list), len(true_centers)])
                    for X, datatree in enumerate(object_center_list):
                        for Y, foundtree in enumerate(true_centers):
                            CostMat[X, Y] = np.linalg.norm(
                                [datatree[0:2]*scale.numpy() - foundtree[0:2]*scale.numpy()]
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

    for key in [
        "slackloss",
        "distance_slackloss",
        "distance_loss",
        "Location_RMSE",
    ]:
        loss_dict[key] = np.mean(loss_dict[key])
    for key in ["n_ref", "n_match", "n_extr"]:
        loss_dict[key] = np.sum(loss_dict[key])
    loss_dict["Completeness"] = loss_dict["n_match"] / loss_dict["n_ref"]
    loss_dict["Correctness"] = loss_dict["n_match"] / max((loss_dict["n_extr"] + loss_dict["n_match"]),1)
    return acc_loss / num_batches_testing, loss_dict

    if cfg.DATA_PREPROCESSING.DISTANCE_FILTER == 0.0:
        _xyz_direction = xyz_direction
    else:
        _xyz_direction = xyz_direction[
            xyz_direction[:, 6] < cfg.DATA_PREPROCESSING.DISTANCE_FILTER
        ]

    object_center_list, seppoints = center_detection(
        _xyz_direction, voxel_size, ARe, Nd
    )


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
