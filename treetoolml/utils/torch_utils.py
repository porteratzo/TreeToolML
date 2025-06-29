import os
import torch
from treetoolml.utils.file_tracability import get_checkpoint_file, find_model_dir
import numpy as np


def device_configs(model, args):
    device = args.device
    device = "cuda" if device == "gpu" else device
    device = device if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_number)
        model.cuda()
    return device


def load_checkpoint(model_dir, model, max=None):
    checkpoint_file = os.path.join(model_dir, "trained_model", "checkpoints")
    checkpoint_path = get_checkpoint_file(checkpoint_file, "BEST", max=max)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(checkpoint_path)


def find_training_dir(cfg, name=False, prepending_dir="results_training"):
    if not name:
        model_name = cfg.TRAIN.MODEL_NAME
    else:
        model_name = cfg
    model_dir = os.path.join(prepending_dir, model_name)
    model_dir = find_model_dir(model_dir)
    return model_dir


def make_xyz_mat(batch_test_data, n_center, dirs):
    pde_ = np.transpose(dirs.cpu().numpy(), [1, 0])
    testdata = batch_test_data[n_center].cpu().numpy()
    xyz_direction = np.concatenate([testdata, pde_], -1).astype(np.float32)

    return xyz_direction
