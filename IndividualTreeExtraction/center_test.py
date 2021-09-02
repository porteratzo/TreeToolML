"""
Created on Mon July 11 18:50:39 2020

@author: Haifeng Luo
"""
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'voxel_traversal'))
sys.path.append(os.path.join(BASE_DIR, 'accessible_region'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append('Libraries')
sys.path.append('utils')
sys.path.append('..')
import py_util
import PointwiseDirectionPrediction_torch as PDE_net
from BatchSampleGenerator_torch import tree_dataset
from torch.utils.data import DataLoader
from center_detection.center_detection import center_detection
import torch
from tqdm import tqdm
from scipy.spatial import distance_matrix
import pandas as pd


def match_centers(gt_centers, pred_centers):
    distmat = distance_matrix(gt_centers, pred_centers)
    return np.argsort(distmat,axis=1)[:,0]


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

############################################################
def compute_object_center(sample_xyz):
    min_xyz = np.min(sample_xyz, axis=0)
    max_xyz = np.max(sample_xyz, axis=0)
    deta_central_xyz = (max_xyz - min_xyz) / 2.0
    central_xyz = deta_central_xyz + min_xyz
    return central_xyz


def individual_tree_extraction(PDE_net_model_path, test_data_path, result_path, voxel_size, Nd, ARe):
    '''Individual Tree Extraction'''
    ####restore trained PDE-net
    PDE_net_model_path
    model = PDE_net.restore_trained_model(NUM_POINT, PDE_net_model_path).cuda()
    generator_val = tree_dataset(
            test_data_path, NUM_POINT
        )
    test_loader = DataLoader(generator_val, 4, shuffle=True, num_workers=0)
    model.eval()
    ####
    totals = {'tp':[0],'fp':[0],'fn':[0]}
    for i in tqdm(range(len(test_loader))):
        #### data[x, y, z] original coordinates
        testdata, directions, labels = next(iter(test_loader))
        tree_n = [len(np.unique(i)) for i in labels]
        sep_trees = [[py_util.compute_object_center(testdata[n].numpy()[i.numpy()==j]) for j in np.unique(i)] for n,i in enumerate(labels)]
        with torch.cuda.amp.autocast():
            out = model(testdata.cuda())
        xyz = testdata.cpu().float().detach().numpy()
        dirs = out.cpu().detach().float().numpy().transpose(0,2,1)
        xyzdir = np.concatenate([xyz, dirs], axis=2)
        object_center_list = [center_detection(i, voxel_size, ARe, Nd) for i in xyzdir]
        matches = [match_centers(i,j) for i,j in zip(sep_trees, object_center_list)]
        gt_centers = [len(i) for i in sep_trees]
        counts = [[len(i),len(np.unique(j))] for i,j in zip(sep_trees, matches)]
        
        tps = sum([min(i) for i in counts])
        fps = sum([i[0]-i[1] for i in counts if i[0]-i[1] < 0])
        fns = sum([i[0]-i[1] for i in counts if i[0]-i[1] > 0])
        totals['tp'][0] += tps
        totals['fp'][0] += fps
        totals['fn'][0] += fns
    df = pd.DataFrame().from_dict(totals)
    df.to_csv(result_path + '/cm.csv')


if __name__ == '__main__':

    NUM_POINT = 4096
    Nd = 80
    ARe = np.pi / 9.0
    voxel_size = 0.08
    #######
    PDE_net_model_path ='IndividualTreeExtraction/backbone_network/pre_trained_PDE_net/'
    test_data_path = 'datasets/custom_data/PDE/validating_data/'
    result_path = './result/'
    if not os.path.exists(result_path): os.mkdir(result_path)

    #######extract individual trees from tree clusters
    individual_tree_extraction(PDE_net_model_path, test_data_path, result_path, voxel_size, Nd, ARe)
