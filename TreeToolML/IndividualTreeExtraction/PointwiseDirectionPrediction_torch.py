"""
Created on Mon July 11 18:50:39 2020

@author: Haifeng Luo
"""

import os
import sys
import torch
import numpy as np
import IndividualTreeExtraction.backbone_network.PDE_net_torch as PDE_net_torch
import glob

def restore_trained_model(MODEL_DIR):
    paths = sorted(glob.glob(os.path.join(MODEL_DIR, '*.pt')))
    with torch.cuda.amp.autocast():
        model = PDE_net_torch.get_model_RRFSegNet()
    model.cuda()
    checkpoint = torch.load(paths[0])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def prediction(model, testdata):

    batch_train_data = torch.as_tensor(testdata)
    batch_train_data = torch.unsqueeze(batch_train_data, 0)
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            model.eval()
            xyz_direction = model(batch_train_data.cuda())
    xyz_direction = xyz_direction.cpu().numpy()
    testdata = np.squeeze(batch_train_data.cpu().numpy())
    pde_ = np.squeeze(xyz_direction).T
    ####################
    xyz_direction = np.concatenate([testdata, pde_], -1)
    return xyz_direction



