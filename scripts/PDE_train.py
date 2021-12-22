import os
import sys

sys.path.append(".")
sys.path.append("/home/omar/Documents/Mine/Git/TreeTool")
import os
import sys

import torch
import TreeToolML.layers.Loss_torch as Loss_torch
import TreeToolML.utils.py_util as py_util
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm import tqdm
from TreeToolML.config.config import combine_cfgs
from TreeToolML.data.BatchSampleGenerator_torch import tree_dataset
from TreeToolML.model.build_model import build_model
from TreeToolML.utils.default_parser import default_argument_parser
from TreeToolML.utils.file_tracability import get_model_dir


def main(args):
    cfg_path = args.cfg
    cfg = combine_cfgs(cfg_path, args.opts)

    device = args.device
    device = "cuda" if device == "gpu" else device
    device = device if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_number)

    model_name = cfg.TRAIN.MODEL_NAME
    model_name = get_model_dir(model_name)
    result_dir = os.path.join("results", model_name, "trained_model")
    os.makedirs(result_dir, exist_ok=True)
    writer = SummaryWriter(result_dir)
    DeepPointwiseDirections = build_model(cfg)

    result_config_path = os.path.join("results", model_name, "full_cfg.yaml")
    cfg_str = cfg.dump()
    with open(result_config_path, "w") as f: 
        f.write(cfg_str)

    if device == "cuda":
        DeepPointwiseDirections.cuda()

    if args.amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    optomizer = Adam(
        DeepPointwiseDirections.parameters(),
        weight_decay=cfg.TRAIN.LOSS.L2,
        lr=cfg.TRAIN.HYPER_PARAMETERS.LR,
    )
    summary(DeepPointwiseDirections, (4096, 3), device=device)

    scheduler = lr_scheduler.ExponentialLR(
        optomizer, cfg.TRAIN.HYPER_PARAMETERS.DECAY_RATE
    )
    ###optimizer--Adam

    init_loss = 999.999
    init_val_loss = 999.999
    for epoch in range(cfg.TRAIN.EPOCHS)[:10]:

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
            args,
        )
        for name, weight in DeepPointwiseDirections.named_parameters():
            writer.add_histogram(name, weight, epoch)
            writer.add_histogram(f"{name}.grad", weight.grad, epoch)
        writer.add_scalar("Loss/traning", temp_loss, epoch)
        torch.cuda.empty_cache()
        #####validating steps
        val_loss = validation(DeepPointwiseDirections, test_loader, args)
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
                os.path.join(result_dir,'checkpoints', f"model-{epoch:03d}-train_loss:{temp_loss:.6f}-val_loss:{val_loss:.6f}.pt",)
                
            )
            init_loss = min(temp_loss, init_loss)
            init_val_loss = min(val_loss, init_val_loss)
        writer.flush()
    writer.close()


def train_one_epoch(model, epoch, generator, opt, scaler, scheduler, args):
    """ops: dict mapping from string to tf ops"""
    if next(model.parameters()).is_cuda:
        device = "cuda"
    else:
        device = "cpu"

    datatype = torch.float16 if (device == "cuda") and args.amp else torch.float32

    num_batches_training = len(generator)
    print("-----------------training--------------------")
    print("training steps: %d" % num_batches_training)

    total_loss = 0
    acc_loss = 0
    model.train()
    for i in tqdm(range(num_batches_training)[:10]):
        ###

        opt.zero_grad()
        batch_train_data, batch_direction_label_data, _ = next(iter(generator))

        batch_train_data = batch_train_data.to(datatype).to(device)
        batch_direction_label_data = batch_direction_label_data.to(datatype).to(device)
        if (args.amp) and device == "cuda":
            with torch.cuda.amp.autocast():
                y = model(batch_train_data)
                total_loss = Loss_torch.slack_based_direction_loss(
                    y, batch_direction_label_data
                )
                acc_loss += total_loss

            scaler.scale(total_loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            y = model(batch_train_data)
            total_loss = Loss_torch.slack_based_direction_loss(
                y, batch_direction_label_data
            )
            acc_loss += total_loss

            total_loss.backward()
            opt.step()

        if i % 20 == 0:
            print("loss: %f" % (total_loss))

    scheduler.step()

    print("trianing_log_epoch_%d" % epoch)
    return acc_loss.cpu() / (num_batches_training)


def validation(model, generator, args):
    if next(model.parameters()).is_cuda:
        device = "cuda"

    datatype = torch.float16 if (device == "cuda") and args.amp else torch.float32

    num_batches_testing = len(generator)
    total_loss_esd = 0
    model.eval()
    for _ in tqdm(range(num_batches_testing)[:10]):
        ###
        batch_test_data, batch_direction_label_data, _ = next(iter(generator))
        batch_test_data = batch_test_data.to(datatype).to(device)
        batch_direction_label_data = batch_direction_label_data.to(datatype).to(device)
        ###
        with torch.no_grad():
            if args.amp:
                with torch.cuda.amp.autocast():
                    y = model(batch_test_data)
                    loss_esd_ = Loss_torch.slack_based_direction_loss(
                        y, batch_direction_label_data
                    )
                    total_loss_esd += loss_esd_
            else:
                y = model(batch_test_data)
                loss_esd_ = Loss_torch.slack_based_direction_loss(
                    y, batch_direction_label_data
                )
                total_loss_esd += loss_esd_
    return total_loss_esd.cpu() / num_batches_testing


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
