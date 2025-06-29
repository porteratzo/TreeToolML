import os
import traceback

import torch
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm import tqdm

import treetoolml.layers.Loss_torch as Loss_torch
from treetoolml.config.config import combine_cfgs
from treetoolml.data.BatchSampleGenerator_torch import tree_dataset_cloud
from treetoolml.model.build_model import build_model
from treetoolml.utils.default_parser import default_argument_parser
from treetoolml.utils.file_tracability import (
    find_model_dir,
    get_checkpoint_file,
    get_model_dir,
    create_training_dir,
)
from treetoolml.utils.torch_utils import device_configs
from tictoc import bench_dict

torch.backends.cudnn.benchmark = True


def main(args):
    cfg_path = args.cfg
    cfg = combine_cfgs(cfg_path, args.opts)
    use_amp = args.amp != 0

    model_name = cfg.TRAIN.MODEL_NAME
    if not args.resume:
        result_dir = create_training_dir(cfg, model_name)

    DeepPointwiseDirections = build_model(cfg)

    device = device_configs(DeepPointwiseDirections, args)

    scaler, optomizer, scheduler = training_setup(cfg, use_amp, DeepPointwiseDirections)

    summary(DeepPointwiseDirections, (4096, 3), device=device)
    savepath = os.path.join(cfg.FILES.DATA_SET, cfg.FILES.DATA_WORK_FOLDER)

    train_path = os.path.join(savepath, "training_data")
    val_path = os.path.join(savepath, "validating_data")
    generator_training = tree_dataset_cloud(
        train_path,
        cfg.TRAIN.N_POINTS,
        normal_filter=cfg.DATA_PREPROCESSING.PC_FILTER,
        return_centers=False,
        distances=1,
        return_trunk=cfg.DATA_CREATION.STICK,
    )
    generator_val = tree_dataset_cloud(
        val_path,
        cfg.TRAIN.N_POINTS,
        normal_filter=True,
        distances=1,
        return_trunk=cfg.DATA_CREATION.STICK,
    )

    init_loss = 999.999
    init_val_loss = 999.999
    start_epoch = 0

    if args.resume:
        init_loss, init_val_loss, start_epoch, result_dir = load_checkpoint(
            cfg, DeepPointwiseDirections, scaler, optomizer, scheduler
        )
    writer = SummaryWriter(result_dir)

    try:
        for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
            ####training data generator
            generator_training[0]
            train_loader = DataLoader(
                generator_training, cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=1
            )

            ####validating data generator
            test_loader = DataLoader(
                generator_val, cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=1
            )

            #####trainging steps
            torch.cuda.empty_cache()
            temp_loss, loss_dict = train_one_epoch(
                DeepPointwiseDirections,
                epoch,
                train_loader,
                optomizer,
                scaler,
                scheduler,
                args,
                use_amp,
                cfg,
            )

            #####validating steps
            torch.cuda.empty_cache()
            val_loss, val_loss_dict = validation(
                DeepPointwiseDirections, test_loader, args, use_amp, cfg
            )
            torch.cuda.empty_cache()
            write_losses(scheduler, writer, epoch, temp_loss, loss_dict, val_loss, val_loss_dict)

            if (temp_loss < init_loss) or (epoch % 10 == 0) or (val_loss < init_val_loss):
                save_checkpoint(
                    DeepPointwiseDirections,
                    scaler,
                    optomizer,
                    scheduler,
                    result_dir,
                    epoch,
                    temp_loss,
                    loss_dict,
                    val_loss,
                    val_loss_dict,
                )
                init_loss = min(temp_loss, init_loss)
                init_val_loss = min(val_loss, init_val_loss)
            writer.flush()
        writer.close()

    except KeyboardInterrupt:
        print("saving perf")
        bench_dict.save()
        print("saved perf")
        writer.close()
        torch.cuda.empty_cache()
        quit()
    except Exception:
        raise
        print(traceback.format_exc())
        writer.close()
        quit()


def train_one_epoch(model, epoch, generator, opt, scaler, scheduler, args, use_amp, cfg):
    """ops: dict mapping from string to tf ops"""
    if next(model.parameters()).is_cuda:
        device = "cuda"
    else:
        device = "cpu"

    num_batches_training = len(generator)
    print("-----------------training--------------------")
    print("training steps: %d" % num_batches_training)

    acc_loss = 0
    loss_dict = {"slackloss": 0, "distance_slackloss": 0, "distance_loss": 0, "accuracy": 0}
    model.train()
    opt.zero_grad(set_to_none=True)
    for i, (batch_train_data, batch_direction_label_data, _) in enumerate(
        tqdm(
            generator,
        )
    ):
        # for i,(batch_train_data, batch_direction_label_data, _)  in enumerate(tqdm(generator)):
        bench_dict["epoch"].step("iter")
        batch_train_data = batch_train_data.half().to(device)
        batch_direction_label_data = batch_direction_label_data.half().to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            y = model(batch_train_data)
            total_loss = Loss_torch.slack_based_direction_loss(
                y,
                batch_direction_label_data,
                use_distance=cfg.TRAIN.DISTANCE_LOSS,
                scaled_dist=cfg.TRAIN.SCALED_DIST,
                classif=cfg.MODEL.CLASS_SIGMOID,
            )
            loss_dict["slackloss"] += (
                Loss_torch.slack_based_direction_loss(
                    y, batch_direction_label_data, use_distance=0, classif=cfg.MODEL.CLASS_SIGMOID
                )
                .cpu()
                .detach()
            )
            loss_dict["distance_slackloss"] += (
                Loss_torch.slack_based_direction_loss(
                    y, batch_direction_label_data, use_distance=1, classif=cfg.MODEL.CLASS_SIGMOID
                )
                .cpu()
                .detach()
            )
            loss_dict["distance_loss"] += (
                Loss_torch.distance_loss(
                    y, batch_direction_label_data, classif=cfg.MODEL.CLASS_SIGMOID
                ).detach()
            ).cpu().detach() * 0.2
            if cfg.TRAIN.DISTANCE:
                total_loss += (
                    Loss_torch.distance_loss(
                        y, batch_direction_label_data, classif=cfg.MODEL.CLASS_SIGMOID
                    )
                    * 0.2
                )
                if cfg.MODEL.CLASS_SIGMOID:
                    pred_labels = y[:, 3] > 0.5
                    gt_labels = batch_direction_label_data[:, :, 3]
                    loss_dict["accuracy"] += torch.sum(pred_labels == gt_labels) / torch.multiply(
                        *gt_labels.shape
                    )
            disp_total_loss = total_loss.cpu().detach()
            acc_loss += total_loss.cpu().detach()
        bench_dict["epoch"].step("infer")

        scaler.scale(total_loss).backward()
        bench_dict["epoch"].step("scale")
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        bench_dict["epoch"].step("backprop")

        bench_dict["epoch"].gstop()

        if i % (num_batches_training // 20) == 0:
            print("loss: %f" % (disp_total_loss))
    scheduler.step()
    opt.zero_grad(set_to_none=True)
    for key in loss_dict.keys():
        loss_dict[key] = loss_dict[key] / num_batches_training

    print("trianing_log_epoch_%d" % epoch)
    return acc_loss / (num_batches_training), loss_dict


def validation(model, generator, args, use_amp, cfg):
    if next(model.parameters()).is_cuda:
        device = "cuda"

    num_batches_testing = len(generator)
    total_loss_esd = 0
    gen = iter(generator)
    loss_dict = {"slackloss": 0, "distance_slackloss": 0, "distance_loss": 0, "accuracy": 0}
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
                loss_esd_ = Loss_torch.slack_based_direction_loss(
                    y,
                    batch_direction_label_data,
                    use_distance=cfg.TRAIN.DISTANCE_LOSS,
                    scaled_dist=cfg.TRAIN.SCALED_DIST,
                    classif=cfg.MODEL.CLASS_SIGMOID,
                )
                loss_dict["slackloss"] += Loss_torch.slack_based_direction_loss(
                    y, batch_direction_label_data, use_distance=0, classif=cfg.MODEL.CLASS_SIGMOID
                ).cpu()
                loss_dict["distance_slackloss"] += Loss_torch.slack_based_direction_loss(
                    y, batch_direction_label_data, use_distance=1, classif=cfg.MODEL.CLASS_SIGMOID
                ).cpu()
                loss_dict["distance_loss"] += (
                    Loss_torch.distance_loss(
                        y, batch_direction_label_data, classif=cfg.MODEL.CLASS_SIGMOID
                    )
                    * 0.2
                ).cpu()
                if cfg.TRAIN.DISTANCE:
                    loss_esd_ += (
                        Loss_torch.distance_loss(
                            y, batch_direction_label_data, classif=cfg.MODEL.CLASS_SIGMOID
                        )
                        * 0.2
                    )
                    if cfg.MODEL.CLASS_SIGMOID:
                        pred_labels = y[:, 3] > 0.5
                        gt_labels = batch_direction_label_data[:, :, 3]
                        loss_dict["accuracy"] += torch.sum(
                            pred_labels == gt_labels
                        ) / torch.multiply(*gt_labels.shape)
                total_loss_esd += loss_esd_
    for key in loss_dict.keys():
        loss_dict[key] = loss_dict[key] / num_batches_testing
    return total_loss_esd.cpu() / num_batches_testing, loss_dict


def write_losses(scheduler, writer, epoch, temp_loss, loss_dict, val_loss, val_loss_dict):
    writer.add_scalar("Loss/total_training", temp_loss, epoch)
    for key in loss_dict.keys():
        writer.add_scalar(f"Loss/train/{key}", loss_dict[key], epoch)
    for key in val_loss_dict.keys():
        writer.add_scalar(f"Loss/val/{key}", val_loss_dict[key], epoch)
    writer.add_scalar("LR/lr", scheduler.get_last_lr()[0], epoch)
    writer.add_scalar("Loss/total_validation", val_loss, epoch)


def save_checkpoint(
    DeepPointwiseDirections,
    scaler,
    optomizer,
    scheduler,
    result_dir,
    epoch,
    temp_loss,
    loss_dict,
    val_loss,
    val_loss_dict,
):
    os.makedirs(os.path.join(result_dir, "checkpoints"), exist_ok=True)
    save_dict = {
        "epoch": epoch,
        "model_state_dict": DeepPointwiseDirections.state_dict(),
        "optimizer_state_dict": optomizer.state_dict(),
        "scaler": scaler.state_dict(),
        "loss": temp_loss,
        "val_loss": val_loss,
        "scheduler": scheduler.state_dict(),
    }
    save_dict.update({"train_" + key: loss_dict[key] for key in loss_dict.keys()})
    save_dict.update({"val_" + key: val_loss_dict[key] for key in val_loss_dict.keys()})
    torch.save(
        save_dict,
        os.path.join(
            result_dir,
            "checkpoints",
            f"model-{epoch:03d}-train_loss:{temp_loss:.6f}-val_loss:{val_loss:.6f}.pt",
        ),
    )


def load_checkpoint(cfg, DeepPointwiseDirections, scaler, optomizer, scheduler):
    model_name = cfg.TRAIN.MODEL_NAME
    result_dir = os.path.join("results_training", model_name)
    result_dir = find_model_dir(result_dir)
    result_dir = os.path.join(result_dir, "trained_model")
    checkpoint_file = os.path.join(result_dir, "checkpoints")
    checkpoint_path = get_checkpoint_file(checkpoint_file)
    checkpoint = torch.load(checkpoint_path)
    (
        DeepPointwiseDirections.load_state_dict(checkpoint["model_state_dict"])
        if checkpoint.get("model_state_dict", False)
        else None
    )
    start_epoch = checkpoint["epoch"] if checkpoint.get("epoch", False) else None
    (
        optomizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("optimizer_state_dict", False)
        else None
    )
    scaler.load_state_dict(checkpoint["scaler"]) if checkpoint.get("scaler", False) else None
    (
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scheduler", False)
        else None
    )
    init_loss = checkpoint["loss"] if checkpoint.get("loss", False) else None
    init_val_loss = checkpoint["val_loss"] if checkpoint.get("val_loss", False) else None
    print(f"starting from epoch {start_epoch}")
    return init_loss, init_val_loss, start_epoch, result_dir


def training_setup(cfg, use_amp, DeepPointwiseDirections):
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    optomizer = Adam(
        DeepPointwiseDirections.parameters(),
        weight_decay=cfg.TRAIN.LOSS.L2,
        lr=cfg.TRAIN.HYPER_PARAMETERS.LR,
    )

    scheduler = lr_scheduler.ExponentialLR(optomizer, cfg.TRAIN.HYPER_PARAMETERS.DECAY_RATE)

    return scaler, optomizer, scheduler


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
