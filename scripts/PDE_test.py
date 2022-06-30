from logging import exception
import os
from pickletools import optimize
import sys

sys.path.append(".")
sys.path.append("/home/omar/Documents/mine/TreeTool")
import os
import sys

import torch
from TreeToolML.utils.tictoc import bench_dict, g_timer1
import TreeToolML.layers.Loss_torch as Loss_torch
import TreeToolML.utils.py_util as py_util
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm import tqdm
from TreeToolML.config.config import combine_cfgs
from TreeToolML.data.BatchSampleGenerator_torch import tree_dataset, tree_dataset_cloud
from TreeToolML.model.build_model import build_model
from TreeToolML.utils.default_parser import default_argument_parser
from TreeToolML.utils.file_tracability import get_model_dir
import traceback
from TreeToolML.Libraries.open3dvis import open3dpaint_sphere
from TreeToolML.utils.file_tracability import get_model_dir, get_checkpoint_file, find_model_dir
import pandas as pd
torch.backends.cudnn.benchmark = True

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
                bench_dict['epoch'].gstep()
                self.n += 1
                while True:
                    try:
                        result = next(self.iter_obj)
                        break
                    except:
                        raise
                        self.n += 1
                        print('error')
                return result
            else:
                raise StopIteration


def main(args):
    cfg_path = args.cfg
    cfg = combine_cfgs(cfg_path, args.opts)
    args.device = 'gpu'
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
    checkpoint_path = get_checkpoint_file(checkpoint_file)
    model = build_model(cfg).cuda()
    if device == "cuda":
        model.cuda()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    #summary(model, (4096, 3), device=device)

    datapath = os.path.join(cfg.FILES.DATA_SET, cfg.FILES.DATA_WORK_FOLDER)

    test_path = os.path.join(datapath, "testing_data")
    generator_val = tree_dataset_cloud(test_path, cfg.TRAIN.N_POINTS, normal_filter=True, distances=cfg.TRAIN.DISTANCE)

    test_loader = DataLoader(
        generator_val, cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=1
    )

    val_loss, loss_dict = testing(model, test_loader, use_amp, cfg)
    pd.DataFrame([loss_dict]).to_csv(os.path.join(model_dir,'test_results.csv'))
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

    num_batches_testing = len(generator)
    acc_loss = 0
    gen = iter(generator)
    loss_dict = {'slackloss': 0, 'distance_slackloss': 0, 'distance_loss': 0}
    model.eval()
    for _ in tqdm(range(num_batches_testing)):
        ###
        batch_test_data, batch_direction_label_data, _ = next(gen)
        batch_test_data = batch_test_data.half().to(device)
        batch_direction_label_data = batch_direction_label_data.half().to(device)
        ###
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                y = model(batch_test_data)
                total_loss = Loss_torch.slack_based_direction_loss(
                    y, batch_direction_label_data, use_distance=cfg.TRAIN.DISTANCE_LOSS
                )
                loss_dict['slackloss'] += Loss_torch.slack_based_direction_loss(
                    y, batch_direction_label_data, use_distance=0
                )
                loss_dict['distance_slackloss'] += Loss_torch.slack_based_direction_loss(
                    y, batch_direction_label_data, use_distance=1
                )
                loss_dict['distance_loss'] += Loss_torch.distance_loss(
                    y, batch_direction_label_data
                )
                if cfg.TRAIN.DISTANCE:
                    total_loss += Loss_torch.distance_loss(
                        y, batch_direction_label_data
                    )
                acc_loss += total_loss
    for key in loss_dict.keys():
        loss_dict[key] = (loss_dict[key].cpu()/num_batches_testing).numpy()
    return acc_loss.cpu() / num_batches_testing, loss_dict


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
