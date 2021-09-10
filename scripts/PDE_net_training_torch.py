"""
Created on Mon July 11 18:50:39 2020

@author: Haifeng Luo
"""
import os
import socket

print(socket.gethostname())
if socket.gethostname() == "omar-G5-KC":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import os
import sys
import torch
from torch.optim import Adam, optimizer, lr_scheduler
from tqdm import tqdm
from BatchSampleGenerator_torch import tree_dataset
from torch.utils.data import DataLoader
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, "utils"))
import py_util
import PDE_net_torch
import Loss_torch
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument(
    "--log_dir",
    default="IndividualTreeExtraction/pre_trained_PDE_net",
    help="Log dir [default: log]",
)
parser.add_argument(
    "--num_point", type=int, default=4096, help="Point number [default: 4096]"
)
parser.add_argument(
    "--max_epoch", type=int, default=100, help="Epoch to run [default: 100]"
)
# parser.add_argument(
#    "--batch_size",
#    type=int,
#    default=20,
#    help="Batch Size during training for each GPU [default: 12]",
# )
parser.add_argument(
    "--learning_rate",
    type=float,
    default=0.001,
    help="Initial learning rate [default: 0.001]",
)
parser.add_argument(
    "--decay_step",
    type=int,
    default=50000,
    help="Decay step for lr decay [default: 50000]",
)
parser.add_argument(
    "--decay_rate",
    type=float,
    default=0.95,
    help="Decay rate for lr decay [default: 0.95]",
)
parser.add_argument(
    "--training_data_path",
    default="datasets/custom_data/PDE/training_data/",
    # default='/container/directory/data/training_data/',
    help="Make sure the source training-data files path",
)
parser.add_argument(
    "--validating_data_path",
    default="datasets/custom_data/PDE/validating_data/",
    # default='/container/directory/data/validating_data/',
    help="Make sure the source validating-data files path",
)

FLAGS = parser.parse_args()
TRAIN_DATA_PATH = FLAGS.training_data_path
VALIDATION_PATH = FLAGS.validating_data_path

if socket.gethostname() == "omar-G5-KC":
    BATCH_SIZE = 4
else:
    BATCH_SIZE = 32

NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, "log_train.txt"), "w")
LOG_FOUT.write(str(FLAGS) + "\n")
device = "cuda" if torch.cuda.is_available() else "cpu"


def log_string(out_str):
    LOG_FOUT.write(out_str + "\n")
    LOG_FOUT.flush()
    print(out_str)


def train():
    writer = SummaryWriter("runs/batchnorm_afineFalse_10k_agressive")
    pointclouds = torch.rand(size=(BATCH_SIZE, NUM_POINT, 3), device=device)
    #####DirectionEmbedding
    # with torch.cuda.amp.autocast():
    DeepPointwiseDirections = PDE_net_torch.get_model_RRFSegNet()
    DeepPointwiseDirections.cuda()
    #writer.add_graph(DeepPointwiseDirections, pointclouds)
    optomizer = Adam(
        DeepPointwiseDirections.parameters(), weight_decay=0.0005, lr=0.001
    )
    scaler = torch.cuda.amp.GradScaler()
    scheduler = lr_scheduler.ExponentialLR(optomizer, 0.95)
    ###optimizer--Adam

    init_loss = 999.999
    init_val_loss = 999.999
    for epoch in range(MAX_EPOCH):
        log_string("**** EPOCH %03d ****" % (epoch))
        sys.stdout.flush()

        ####training data generator
        generator_training = tree_dataset(
            TRAIN_DATA_PATH, NUM_POINT
        )
        train_loader = DataLoader(generator_training, BATCH_SIZE, shuffle=True, num_workers=1)

        ####validating data generator
        generator_val = tree_dataset(
            VALIDATION_PATH, NUM_POINT
        )
        test_loader = DataLoader(generator_val, BATCH_SIZE, shuffle=True, num_workers=1)

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
                os.path.join(LOG_DIR, "epoch_" + str(epoch) + ".pt"),
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
        scaler.scale(total_loss).backward()
        scaler.step(opt)
        scaler.update()
    

        if i % 20 == 0:
            print(
                "loss: %f" % (total_loss)
            )
    scheduler.step()

    print("trianing_log_epoch_%d" % epoch)
    log_string(
        "epoch: %d, loss: %f"
        % (
            epoch,
            acc_loss / (num_batches_training),
        )
    )
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
            with torch.cuda.amp.autocast():
                y = model(torch.as_tensor(batch_test_data, device=device))
                batch_direction_label_data = torch.as_tensor(
                    batch_direction_label_data, device=device
                )
                loss_esd_ = Loss_torch.slack_based_direction_loss(
                    y, batch_direction_label_data
                )
        total_loss_esd += loss_esd_

    log_string(
        "val loss: %f"
        % (
            total_loss_esd / num_batches_testing,
        )
    )
    return total_loss_esd / num_batches_testing


if __name__ == "__main__":
    train()
    LOG_FOUT.close()