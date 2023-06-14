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
from porteratzolibs.visualization_o3d.open3dvis import open3dpaint
from treetoolml.utils.vis_utils import vis_trees_centers


torch.backends.cudnn.benchmark = True

torch.manual_seed(123)

def angle_b_vectors(a, b):
    dot_product = np.sum(np.multiply(a, b), axis=-1)
    norms_product = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    
    # Ensure the value is within the valid range for arccos
    value = np.clip(dot_product / norms_product, -1, 1)
    
    angle = np.arccos(value)
    return angle

def DistPoint2Line(point, linepoint1, linepoint2=np.array([0, 0, 0])):
    cross_product = np.cross((point - linepoint2), (point - linepoint1))
    distance = np.linalg.norm(cross_product, axis=-1) / np.linalg.norm(linepoint1 - linepoint2, axis=-1)
    return distance


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
    checkpoint_path = get_checkpoint_file(checkpoint_file, 'BEST', max=50)
    print(checkpoint_path)
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
                    .numpy()* 0.2
                ) 
                loss_dict["distance_slackloss_scaled"].append(
                    Loss_torch.slack_based_direction_loss(y, batch_direction_label_data, use_distance=1, scaled_dist=1)
                    .cpu()
                    .numpy()
                )
                if cfg.TRAIN.DISTANCE:
                    total_loss += Loss_torch.distance_loss(
                        y, batch_direction_label_data
                    ).cpu().numpy()
                acc_loss += total_loss

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
                        _xyz_direction = xyz_direction[
                            xyz_direction[:, 6] < cfg.DATA_PREPROCESSING.DISTANCE_FILTER
                        ]

                    average_pred_distance = []
                    average_pred_angle = []
                    average_correctly_assigned_01 = []
                    average_correctly_assigned_03 = []
                    average_correctly_assigned_06 = []

                    average_correctly_assigned_01_d = []
                    average_correctly_assigned_03_d = []
                    average_correctly_assigned_06_d = []
                    for n_c_,center in enumerate(true_centers):                                        

                        idx_s = batch_object_label[n_center]==n_c_
                        xyz_ = xyz_direction[:,0:3][idx_s]
                        directions_ = xyz_direction[:,3:6][idx_s]
                        distance_ = np.linalg.norm(center - xyz_, axis=-1)
                        
                        angles_ = angle_b_vectors(center-xyz_,directions_-xyz_)
                        dists_ = DistPoint2Line(center, xyz_, directions_)
                        average_pred_distance.append(np.mean(dists_))
                        average_pred_angle.append(np.mean(angles_))
                        average_correctly_assigned_01.append(np.sum(dists_<0.1)/len(dists_))
                        average_correctly_assigned_03.append(np.sum(dists_<0.3)/len(dists_))
                        average_correctly_assigned_06.append(np.sum(dists_<0.6)/len(dists_))

                        d_dists_ = dists_[distance_<0.5]
                        average_correctly_assigned_01_d.append(np.sum(d_dists_<0.1)/len(d_dists_))
                        average_correctly_assigned_03_d.append(np.sum(d_dists_<0.3)/len(d_dists_))
                        average_correctly_assigned_06_d.append(np.sum(d_dists_<0.6)/len(d_dists_))

                    loss_dict["average_pred_distance"].append(np.mean(average_pred_distance))
                    loss_dict["average_pred_angle"].append(np.mean(average_pred_angle))
                    loss_dict["average_correctly_assigned_01"].append(np.mean(average_correctly_assigned_01))
                    loss_dict["average_correctly_assigned_03"].append(np.mean(average_correctly_assigned_03))
                    loss_dict["average_correctly_assigned_06"].append(np.mean(average_correctly_assigned_06))

                    loss_dict["average_correctly_assigned_01_d"].append(np.mean(average_correctly_assigned_01_d))
                    loss_dict["average_correctly_assigned_03_d"].append(np.mean(average_correctly_assigned_03_d))
                    loss_dict["average_correctly_assigned_06_d"].append(np.mean(average_correctly_assigned_06_d))
                    
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
        "average_pred_distance",
        "average_pred_angle",
        'distance_slackloss_scaled',
        'average_correctly_assigned_01',
        'average_correctly_assigned_03',
        'average_correctly_assigned_06',
        'average_correctly_assigned_01_d',
        'average_correctly_assigned_03_d',
        'average_correctly_assigned_06_d',
    ]:
        loss_dict[key] = np.mean(loss_dict[key])
    for key in ["n_ref", "n_match", "n_extr"]:
        loss_dict[key] = np.sum(loss_dict[key])
    loss_dict["Completeness"] = loss_dict["n_match"] / loss_dict["n_ref"]
    loss_dict["Correctness"] = loss_dict["n_match"] / max((loss_dict["n_extr"] + loss_dict["n_match"]),1)
    return acc_loss / num_batches_testing, loss_dict

def make_xyz_mat(batch_test_data, n_center, dirs):
    pde_ = np.transpose(dirs.cpu().numpy(), [1, 0])
    testdata = batch_test_data[n_center].cpu().numpy()
    xyz_direction = np.concatenate([testdata, pde_], -1).astype(
                        np.float32
                    )
    
    return xyz_direction


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
