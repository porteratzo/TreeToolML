import os
import socket

import sys
sys.path.append('.')
print(socket.gethostname())
if socket.gethostname() == "omar-G5-KC":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import os
import sys
import torch
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm
from TreeToolML.data.BatchSampleGenerator_torch import tree_dataset
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, "utils"))
import TreeToolML.utils.py_util as py_util
from TreeToolML.utils.default_parser import default_argument_parser
from TreeToolML.config.config import combine_cfgs
from TreeToolML.model.build_model import build_model
import TreeToolML.layers.Loss_torch as Loss_torch
from torch.utils.tensorboard import SummaryWriter


device = "cuda" if torch.cuda.is_available() else "cpu"


def main(args):
    cfg_path = args.cfg
    cfg = combine_cfgs(cfg_path, args.opts)

    if not os.path.exists(cfg.TRAIN.LOG_DIR):
        os.mkdir(cfg.TRAIN.LOG_DIR)
    writer = SummaryWriter(cfg.TRAIN.LOG_DIR)
    DeepPointwiseDirections = build_model(cfg)
    if device=='cuda':
        DeepPointwiseDirections.cuda()
    optomizer = Adam(
        DeepPointwiseDirections.parameters(),
        weight_decay=cfg.TRAIN.LOSS.L2,
        lr=cfg.TRAIN.HYPER_PARAMETERS.LR,
    )
    scaler = torch.cuda.amp.GradScaler()
    scheduler = lr_scheduler.ExponentialLR(
        optomizer, cfg.TRAIN.HYPER_PARAMETERS.DECAY_RATE
    )
    ###optimizer--Adam

    init_loss = 999.999
    init_val_loss = 999.999
    for epoch in range(cfg.TRAIN.EPOCHS):

        ####training data generator
        generator_training = tree_dataset(cfg.TRAIN.PATH, cfg.TRAIN.N_POINTS)
        train_loader = DataLoader(
            generator_training, cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=0
        )

        ####validating data generator
        generator_val = tree_dataset(cfg.VALIDATION.PATH, cfg.TRAIN.N_POINTS)
        test_loader = DataLoader(
            generator_val, cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=0
        )

        #####trainging steps
        temp_loss = train_one_epoch(
            DeepPointwiseDirections,
            epoch,
            train_loader,
            optomizer,
            scaler,
            scheduler,
        )
        for name, weight in DeepPointwiseDirections.named_parameters():
            writer.add_histogram(name, weight, epoch)
            writer.add_histogram(f"{name}.grad", weight.grad, epoch)
        writer.add_scalar("Loss/traning", temp_loss, epoch)
        torch.cuda.empty_cache()
        #####validating steps
        val_loss = validation(DeepPointwiseDirections, test_loader)
        writer.add_scalar("LR/lr", scheduler.get_last_lr()[0], epoch)
        writer.add_scalar("Loss/validation", val_loss, epoch)

        if (temp_loss < init_loss) or (epoch % 10 == 0) or (val_loss < init_val_loss):
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": DeepPointwiseDirections.state_dict(),
                    "optimizer_state_dict": optomizer.state_dict(),
                    "loss": temp_loss,
                    "val_loss": val_loss,
                },
                os.path.join(cfg.TRAIN.LOG_DIR, "epoch_" + str(epoch) + ".pt"),
            )
            init_loss = min(temp_loss, init_loss)
            init_val_loss = min(val_loss, init_val_loss)
        writer.flush()
    writer.close()


def train_one_epoch(model, epoch, generator, opt, scaler, scheduler):
    """ops: dict mapping from string to tf ops"""

    num_batches_training = len(generator)
    print("-----------------training--------------------")
    print("training steps: %d" % num_batches_training)

    total_loss = 0
    acc_loss = 0
    model.train()
    for i in tqdm(range(num_batches_training)):
        ###
        opt.zero_grad()
        batch_train_data, batch_direction_label_data, _ = next(iter(generator))
        if device=='cuda':
            with torch.cuda.amp.autocast():
                batch_train_data = torch.tensor(batch_train_data, device=device)
                batch_direction_label_data = torch.tensor(
                    batch_direction_label_data, device=device
                )
                y = model(batch_train_data)
                loss_esd_ = Loss_torch.slack_based_direction_loss(
                    y, batch_direction_label_data
                )
                total_loss = loss_esd_
                acc_loss += total_loss
        else:
            batch_train_data = torch.tensor(batch_train_data, device=device)
            batch_direction_label_data = torch.tensor(
                batch_direction_label_data, device=device
            )
            y = model(batch_train_data.float())
            loss_esd_ = Loss_torch.slack_based_direction_loss(
                y, batch_direction_label_data.float()
            )
            total_loss = loss_esd_
            acc_loss += total_loss
        scaler.scale(total_loss).backward()
        scaler.step(opt)
        scaler.update()

        if i % 20 == 0:
            print("loss: %f" % (total_loss))
    scheduler.step()

    print("trianing_log_epoch_%d" % epoch)
    return acc_loss.cpu() / (num_batches_training)


def validation(model, generator):

    num_batches_testing = len(generator)
    total_loss_esd = 0
    model.eval()
    for _ in tqdm(range(num_batches_testing)):
        ###
        batch_test_data, batch_direction_label_data, _ = next(iter(generator))
        ###
        with torch.no_grad():
            if device=='cuda':
                with torch.cuda.amp.autocast():
                    y = model(torch.as_tensor(batch_test_data, device=device))
                    batch_direction_label_data = torch.as_tensor(
                        batch_direction_label_data, device=device
                    )
                    loss_esd_ = Loss_torch.slack_based_direction_loss(
                        y, batch_direction_label_data
                    )
            else:
                y = model(torch.as_tensor(batch_test_data, device=device).float())
                batch_direction_label_data = torch.as_tensor(
                    batch_direction_label_data, device=device
                ).float()
                loss_esd_ = Loss_torch.slack_based_direction_loss(
                    y, batch_direction_label_data
                )
        total_loss_esd += loss_esd_


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
