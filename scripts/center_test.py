"""
Created on Mon July 11 18:50:39 2020

@author: Haifeng Luo
"""
import sys
sys.path.append('..')
import numpy as np
import os
import IndividualTreeExtraction.utils.py_util as py_util
from IndividualTreeExtraction.utils.py_util import compute_object_center
import IndividualTreeExtraction.PointwiseDirectionPrediction_torch as PDE_net
from IndividualTreeExtraction.backbone_network.BatchSampleGenerator_torch import tree_dataset
from torch.utils.data import DataLoader
from IndividualTreeExtraction.center_detection.center_detection import center_detection
import torch
from tqdm import tqdm
from scipy.spatial import distance_matrix
import pandas as pd
import socket
from argparse import ArgumentParser
# python center_test.py --model /data2/omardata/TreeTransformer/IndividualTreeExtraction/backbone_network/pre_trained_PDE_net --valid_path /data2/omardata/TreeTransformer/datasets/custom_data/PDE/validating_data

print(socket.gethostname())
if socket.gethostname() == "omar-G5-KC":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def argparse():
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        default='IndividualTreeExtraction/backbone_network/pre_trained_PDE_net/',
        help="Log dir [default: log]",
    )
    parser.add_argument(
        "--valid_path",
        default='datasets/custom_data/PDE/validating_data/',
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


def individual_tree_extraction(PDE_net_model_path, test_data_path, result_path, voxel_size, Nd, ARe):
    '''Individual Tree Extraction'''
    ####restore trained PDE-net
    PDE_net_model_path
    model = PDE_net.restore_trained_model(PDE_net_model_path).cuda()
    generator_val = tree_dataset(
            test_data_path, NUM_POINT
        )
    test_loader = DataLoader(generator_val, 4, shuffle=True, num_workers=0)
    model.eval()
    ####
    totals = {'tp':0,'fp':0,'fn':0}
    distss = []
    for i in tqdm(range(len(test_loader))):
        #### data[x, y, z] original coordinates
        testdata, directions, labels = next(iter(test_loader))
        tree_n = [len(np.unique(i)) for i in labels]
        gt_centers = [[py_util.compute_object_center(testdata[n].numpy()[i.numpy()==j]) for j in np.unique(i)] for n,i in enumerate(labels)]
        with torch.cuda.amp.autocast():
            out = model(testdata.cuda())
        xyz = testdata.cpu().float().detach().numpy()
        dirs = out.cpu().detach().float().numpy().transpose(0,2,1)
        xyzdir = np.concatenate([xyz, dirs], axis=2)
        predicted_centers = [center_detection(i, voxel_size, ARe, Nd) for i in xyzdir]
        try:
            for gt_center, predicted_center in zip(gt_centers,predicted_centers):
                dist_mat = centerdists(gt_center, predicted_center)
                close_matches = np.argsort(dist_mat)[:,0][np.sort(dist_mat)[:,0] < 0.3]
                dists = np.sort(dist_mat)[:,0][np.sort(dist_mat)[:,0] < 0.3]
                true_matches = np.unique(close_matches)
                tps = len(true_matches)
                fps = max(len(predicted_center) - len(true_matches),0)
                fns = max(len(gt_center) - len(true_matches),0)
                totals['tp'] += tps
                totals['fp'] += fps
                totals['fn'] += fns            
                distss.extend(dists)
        except:
            continue
    totals['mse'] = [np.mean(distss)]
    df = pd.DataFrame().from_dict(totals)
    df.to_csv(result_path + '/cm.csv')


if __name__ == '__main__':
    args = argparse()
    NUM_POINT = 4096
    Nd = 80
    ARe = np.pi / 9.0
    voxel_size = 0.08
    #######
    PDE_net_model_path = args.model
    test_data_path = args.valid_path
    result_path = './result/'
    if not os.path.exists(result_path): os.mkdir(result_path)

    #######extract individual trees from tree clusters
    individual_tree_extraction(PDE_net_model_path, test_data_path, result_path, voxel_size, Nd, ARe)
