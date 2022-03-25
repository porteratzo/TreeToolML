"""
Created on Mon July 11 18:50:39 2020

@author: Haifeng Luo
"""
import os
import sys

sys.path.append(".")
import matplotlib.pyplot as plt
import numpy as np
import torch
import TreeToolML.layers.Loss_torch as Loss_torch
import TreeToolML.utils.py_util as py_util
from torch.utils.data import DataLoader
from TreeToolML.config.config import combine_cfgs
from TreeToolML.data.BatchSampleGenerator_torch import tree_dataset
from TreeToolML.IndividualTreeExtraction.center_detection.center_detection import \
    center_detection
from TreeToolML.IndividualTreeExtraction.PointwiseDirectionPrediction_torch import \
    prediction
from TreeToolML.Libraries.open3dvis import o3d_pointSetClass, open3dpaint
from TreeToolML.Libraries.Plane import makepointvector
# import AccessibleRegionGrowing as ARG
from TreeToolML.model.build_model import build_model
from TreeToolML.utils.default_parser import default_argument_parser


def makesphere(centroid=[0, 0, 0], radius=1, dense=90):
    n = np.arange(0, 360, int(360 / dense))
    n = np.deg2rad(n)
    x, y = np.meshgrid(n, n)
    x = x.flatten()
    y = y.flatten()
    sphere = np.vstack(
        [
            centroid[0] + np.sin(x) * np.cos(y) * radius,
            centroid[1] + np.sin(x) * np.sin(y) * radius,
            centroid[2] + np.cos(x) * radius,
        ]
    ).T
    return sphere


def show_AR_RG(voxels1, voxels2):
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ####accessible region
    ax.voxels(voxels2, facecolors="red", edgecolor="k", alpha=0.9)
    ####region growing results
    ax.voxels(voxels1, facecolors="green", edgecolor="k")
    plt.show()


def individual_tree_extraction(args):
    cfg_path = args.cfg
    cfg = combine_cfgs(cfg_path, args.opts)

    device = args.device
    device = "cuda" if device == "gpu" else device
    device = device if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_number)

    Nd = 80
    ARe = np.pi / 9.0
    voxel_size = 0.08


    """Individual Tree Extraction"""
    ####restore trained PDE-net
    model = build_model(cfg).cuda()
    if device == "cuda":
        model.cuda()

    generator_val = tree_dataset(cfg.TRAIN.PATH, cfg.TRAIN.N_POINTS, return_centers=True)
    test_loader = DataLoader(generator_val, 1, shuffle=True, num_workers=0)
    ####
    file_list = os.listdir(cfg.VALIDATION.PATH)
    for i in range(len(file_list[:10])):
        tree_index = 0
        filename, _ = os.path.splitext(file_list[i])
        print("Separating " + filename + "...")
        #### data[x, y, z] original coordinates
        testdata, directions, labels, object_centers = next(iter(test_loader))
        testdata, directions, labels, object_centers = (
            testdata.squeeze().numpy(),
            directions.squeeze().numpy(),
            labels.squeeze().numpy(),
            [i.squeeze().numpy() for i in object_centers],
        )
        ind_trees = [testdata[labels == i] for i in np.unique(labels)]
        #object_centers = [py_util.compute_object_center(i) for i in ind_trees]
        ####normalized coordinates
        nor_testdata = torch.tensor(testdata, device="cuda").squeeze()
        ####Pointwise direction prediction
        xyz_direction = prediction(model, nor_testdata, args)
        ####tree center detection
        if True:
            xyz = xyz_direction[:, :3]
            angles = np.rad2deg(np.arctan2(directions[:, 1], directions[:, 0]))
            ps = o3d_pointSetClass(xyz, angles)
            open3dpaint(
                [ps] + [makesphere(i, 0.05) for i in object_centers],
                pointsize=5,
                axis=0.1,
            )

            angles = np.rad2deg(np.arctan2(directions[:, 2], directions[:, 0]))
            ps = o3d_pointSetClass(xyz, angles)
            open3dpaint(
                [ps] + [makesphere(i, 0.05) for i in object_centers],
                pointsize=5,
                axis=0.1,
            )

            angles = np.rad2deg(np.arctan2(directions[:, 2], directions[:, 1]))
            ps = o3d_pointSetClass(xyz, angles)
            open3dpaint(
                [ps] + [makesphere(i, 0.05) for i in object_centers],
                pointsize=5,
                axis=0.1,
            )
            continue

            object_center_list, _ = center_detection(xyz_direction, voxel_size, ARe, Nd)
            loss_esd_ = Loss_torch.slack_based_direction_loss(
                torch.tensor(xyz_direction.T[np.newaxis, 3:6, :].astype(np.float32)),
                torch.tensor(directions[np.newaxis, :].astype(np.float32)),
            )
            print(loss_esd_)
            dirsxyz = xyz_direction[:, 3:]

            angles = np.rad2deg(np.arctan2(dirsxyz[:, 1], dirsxyz[:, 0]))
            ps = o3d_pointSetClass(xyz, angles)
            open3dpaint(
                [ps] + [makesphere(i, 0.1) for i in object_center_list],
                pointsize=5,
                axis=True,
            )

            angles = np.rad2deg(np.arctan2(dirsxyz[:, 2], dirsxyz[:, 0]))
            ps = o3d_pointSetClass(xyz, angles)
            open3dpaint(
                [ps] + [makesphere(i, 0.1) for i in object_center_list],
                pointsize=5,
                axis=True,
            )

            angles = np.rad2deg(np.arctan2(dirsxyz[:, 2], dirsxyz[:, 1]))
            ps = o3d_pointSetClass(xyz, angles)
            open3dpaint(
                [ps] + [makesphere(i, 0.1) for i in object_center_list],
                pointsize=5,
                axis=True,
            )
        else:
            xyz = xyz_direction[:, :3]
            dirsxyz = xyz_direction[:, 3:]
            ps = o3d_pointSetClass(xyz)

            open3dpaint(
                [ps] + [makesphere(i, 0.1) for i in object_centers],
                pointsize=5,
                axis=True,
            )

            object_center_list, sepponts = center_detection(xyz_direction, voxel_size, ARe, Nd)
            sepponts = [i for i in sepponts if np.size(i,0)]
            if len(sepponts) > 0:
                loss_esd_ = Loss_torch.slack_based_direction_loss(
                    torch.tensor(xyz_direction.T[np.newaxis, 3:6, :].astype(np.float32)),
                    torch.tensor(directions[np.newaxis, :].astype(np.float32)),
                )
                print(loss_esd_)
                open3dpaint(
                    sepponts
                    + [makesphere(i, 0.05) for i in object_center_list],
                    pointsize=8,
                    axis=True,
                )

        continue


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    individual_tree_extraction(args)
