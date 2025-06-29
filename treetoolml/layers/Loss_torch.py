"""
Loss functions for training the PDE-net

Created on Mon July 11 18:50:39 2020

@author: Haifeng Luo
"""
import torch
import torch.nn.functional as f


def distance_loss(pre_direction, gt_direction, sigma=0.955, classif=False):
    '''
    Loss in inferering distance from center
    '''
    pre_direction = pre_direction.permute(0, 2, 1)
    if classif:
        _gt_class = gt_direction[:, :, 3]
        _pre_class = pre_direction[:, :, 3]
        loss = torch.mean(torch.binary_cross_entropy_with_logits(_pre_class, _gt_class))
    else:
        if (pre_direction.shape[2]>3) and (gt_direction.shape[2]>3):
            _gt_direction = gt_direction[:, :, 3]
            _pre_direction = pre_direction[:, :, 3]
            loss = torch.mean(torch.square(_gt_direction - _pre_direction))
        else:
            loss = torch.ones(1)
    return loss


def slack_based_direction_loss(pre_direction, gt_direction, sigma=0.955, use_distance=False, scaled_dist = False, classif=False):
    '''
    Error Slack-based Direction Loss
    '''
    _pre_direction = pre_direction.permute(0, 2, 1)
    _gt_direction = f.normalize(gt_direction[:, :, :3], dim=2, eps=1e-20)
    _pre_direction = f.normalize(_pre_direction[:, :, :3], dim=2, eps=1e-20)

    if scaled_dist and use_distance:
        loss = (sigma - torch.sum(torch.multiply(_pre_direction, _gt_direction), dim=2)) * (1-gt_direction[:, :, 3]) / torch.mean(1-gt_direction[:, :, 3], dim=1).unsqueeze(1)
    elif use_distance:
        if classif:
            scale_v = gt_direction[:, :, 3]
        else:
            scale_v = (1-gt_direction[:, :, 3])
        loss = (sigma - torch.sum(torch.multiply(_pre_direction, _gt_direction), dim=2)) * scale_v
    else:
        loss = sigma - torch.sum(torch.multiply(_pre_direction, _gt_direction), dim=2)
    tmp = torch.zeros_like(loss)
    condition = torch.greater(loss, 0.0)
    loss = torch.where(condition, loss, tmp)
    loss = torch.mean(loss)
    return loss


def _slack_based_direction_loss(pre_direction, gt_direction, sigma=0.955, use_distance=False):
    '''
    Error Slack-based Direction Loss
    '''
    pre_direction = pre_direction.permute(0, 2, 1)
    gt_direction = f.normalize(gt_direction, dim=2, eps=1e-20)
    pre_direction = f.normalize(pre_direction, dim=2, eps=1e-20)

    loss = sigma - torch.sum(torch.multiply(pre_direction, gt_direction), dim=2)
    tmp = torch.zeros_like(loss)
    condition = torch.greater(loss, 0.0)
    loss = torch.where(condition, loss, tmp)
    loss = torch.mean(loss)
    return loss


def direction_loss(pre_direction, gt_direction):
    '''
    Plain Direction Loss
    '''
    pre_direction = pre_direction.permute(0, 2, 1)
    gt_direction = f.normalize(gt_direction, dim=2, eps=1e-20)
    pre_direction = f.normalize(pre_direction, dim=2, eps=1e-20)
    loss = -torch.mean(torch.sum(torch.multiply(pre_direction, gt_direction), dim=2))
    return loss
