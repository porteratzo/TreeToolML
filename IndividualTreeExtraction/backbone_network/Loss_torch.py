"""
Loss functions for training the PDE-net

Created on Mon July 11 18:50:39 2020

@author: Haifeng Luo
"""
import torch
import torch.nn.functional as f


def slack_based_direction_loss(pre_direction, gt_direction, sigma=0.955):
    '''
    Error Slack-based Direction Loss
    '''
    pre_direction = pre_direction.permute(0,2,1)
    gt_direction = f.normalize(gt_direction, dim=2, eps=1e-20)
    pre_direction = f.normalize(pre_direction, dim=2, eps=1e-20)

    loss = sigma-torch.sum(torch.multiply(pre_direction, gt_direction), dim=2)
    tmp = torch.zeros_like(loss)
    condition = torch.greater(loss, 0.0)
    loss = torch.where(condition, loss, tmp)
    loss = torch.mean(loss)
    return loss


def direction_loss(pre_direction, gt_direction):
    '''
    Plain Direction Loss
    '''
    pre_direction = pre_direction.permute(0,2,1)
    gt_direction = f.normalize(gt_direction, dim=2, eps=1e-20)
    pre_direction = f.normalize(pre_direction, dim=2, eps=1e-20)
    loss = -torch.mean(torch.sum(torch.multiply(pre_direction, gt_direction), dim=2))
    return loss