"""
Created on Mon July 11 18:50:39 2020

@author: Haifeng Luo
"""
import sys

sys.path.append("..")
import os
import socket
from argparse import ArgumentParser

import pickle
import numpy as np
import pandas as pd
import torch
import treetoolml.utils.py_util as py_util
from scipy.spatial import distance_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from treetoolml.config.config import combine_cfgs
from treetoolml.data.BatchSampleGenerator_torch import tree_dataset
from treetoolml.IndividualTreeExtraction.center_detection.center_detection import (
    center_detection,
)
from torch.multiprocessing import Pool
from itertools import repeat
from treetoolml.IndividualTreeExtraction.PointwiseDirectionPrediction_torch import (
    prediction,
)
from treetoolml.model.build_model import build_model
from treetoolml.utils.default_parser import default_argument_parser
from treetoolml.utils.py_util import compute_object_center
from treetoolml.utils.file_tracability import get_model_dir, get_checkpoint_file, find_model_dir

print(socket.gethostname())
if socket.gethostname() == "omar-G5-KC":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def argparse():
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        default="IndividualTreeExtraction/backbone_network/pre_trained_PDE_net/",
        help="Log dir [default: log]",
    )
    parser.add_argument(
        "--valid_path",
        default="datasets/custom_data/PDE/validating_data/",
        # default='/container/directory/data/validating_data/',
        help="Make sure the source validating-data files path",
    )

    return parser.parse_args()


def centerdists(gt_centers, pred_centers):
    distmat = distance_matrix(gt_centers, pred_centers)
    return distmat


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


def individual_tree_extraction(args):
    cfg_path = args.cfg
    cfg = combine_cfgs(cfg_path, args.opts)

    device = args.device
    device = "cuda" if device == "gpu" else device
    device = device if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_number)

    NUM_POINT = cfg.TRAIN.N_POINTS
    Nd = 80
    ARe = np.pi / 9.0
    voxel_size = 0.08
    #######

    test_data_path = cfg.VALIDATION.PATH
    model_name = cfg.TRAIN.MODEL_NAME
    result_dir = os.path.join("results", model_name)
    result_dir = find_model_dir(result_dir)
    checkpoint_file = os.path.join(result_dir,'trained_model','checkpoints')
    checkpoint_path = get_checkpoint_file(checkpoint_file)
    ####restore trained PDE-net
    model = build_model(cfg).cuda()
    if device == "cuda":
        model.cuda()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    generator_val = tree_dataset(test_data_path, NUM_POINT)
    test_loader = DataLoader(generator_val, cfg.VALIDATION.BATCH_SIZE, shuffle=True, num_workers=0)
    model.eval()
    ####
    totals = {"tp": 0, "fp": 0, "fn": 0}
    distss = []
    pool = Pool()
    records = []
    for n,i in enumerate(tqdm(range(len(test_loader)))):
        #### data[x, y, z] original coordinates
        testdata, directions, labels = next(iter(test_loader))
        tree_n = [len(np.unique(i)) for i in labels]
        gt_centers = [
            [
                py_util.compute_object_center(testdata[n].numpy()[i.numpy() == j])
                for j in np.unique(i)
            ]
            for n, i in enumerate(labels)
        ]
        testdata = testdata.squeeze()
        tensor_testdata = torch.tensor(testdata, device="cuda").squeeze()
        out = prediction(model, tensor_testdata, args)
        if True:
            predicted_centers = [center_detection(i, voxel_size, ARe, Nd) for i in out]
        else:            
            predicted_centers = pool.starmap(center_detection, zip(
                    out,
                    repeat(voxel_size),
                    repeat(ARe),
                    repeat(Nd),
                ))        
        try:
            for gt_center, predicted_center in zip(gt_centers, predicted_centers):
                if False:
                    dist_mat = centerdists(gt_center, predicted_center)
                    close_matches = np.argsort(dist_mat)[:, 0][
                        np.sort(dist_mat)[:, 0] < 0.3
                    ]
                    dists = np.sort(dist_mat)[:, 0][np.sort(dist_mat)[:, 0] < 0.3]
                    true_matches = np.unique(close_matches)
                    tps = len(true_matches)
                    fps = max(len(predicted_center) - len(true_matches), 0)
                    fns = max(len(gt_center) - len(true_matches), 0)
                    totals["tp"] += tps
                    totals["fp"] += fps
                    totals["fn"] += fns
                    distss.extend(dists)
                else:
                    records.append([gt_center, predicted_center])
        except:
            continue
    test_path = os.path.join(result_dir,'test')
    os.makedirs(test_path, exist_ok=True)
    with  open(os.path.join(test_path, 'results.pkl'), 'wb') as f:
        pickle.dump(records, f)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    #######extract individual trees from tree clusters
    individual_tree_extraction(args)
