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
#from torch.profiler import profile, record_function, ProfilerActivity
from TreeToolML.utils.file_tracability import get_model_dir, get_checkpoint_file, find_model_dir
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
    use_amp = args.amp != 0


    model_name = cfg.TRAIN.MODEL_NAME
    if not args.resume:
        model_name = get_model_dir(model_name)
        result_dir = os.path.join("results_training", model_name, "trained_model")
        os.makedirs(result_dir, exist_ok=True)
        result_config_path = os.path.join("results_training", model_name, "full_cfg.yaml")
        cfg_str = cfg.dump()
        with open(result_config_path, "w") as f: 
            f.write(cfg_str)

    DeepPointwiseDirections = build_model(cfg)

    device = device_configs(DeepPointwiseDirections, args)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    optomizer = Adam(
        DeepPointwiseDirections.parameters(),
        weight_decay=cfg.TRAIN.LOSS.L2,
        lr=cfg.TRAIN.HYPER_PARAMETERS.LR,
    )
    summary(DeepPointwiseDirections, (4096, 3), device=device)

    scheduler = lr_scheduler.ExponentialLR(
        optomizer, cfg.TRAIN.HYPER_PARAMETERS.DECAY_RATE
    )
    savepath = os.path.join(cfg.FILES.DATA_SET, cfg.FILES.DATA_WORK_FOLDER)

    train_path = os.path.join(savepath, "training_data")
    val_path = os.path.join(savepath, "validating_data")
    generator_training = tree_dataset_cloud(train_path, cfg.TRAIN.N_POINTS, normal_filter=cfg.DATA_PREPROCESSING.PC_FILTER, return_centers=False, distances=cfg.TRAIN.DISTANCE)
    generator_val = tree_dataset_cloud(val_path, cfg.TRAIN.N_POINTS, normal_filter=True, distances=cfg.TRAIN.DISTANCE)
    ###optimizer--Adam

    init_loss = 999.999
    init_val_loss = 999.999
    start_epoch = 0

    if args.resume:
        model_name = cfg.TRAIN.MODEL_NAME
        result_dir = os.path.join("results_training", model_name, "trained_model")
        result_dir = find_model_dir(result_dir)
        checkpoint_file = os.path.join(result_dir,'trained_model','checkpoints')
        checkpoint_path = get_checkpoint_file(checkpoint_file)
        checkpoint = torch.load(checkpoint_path)
        DeepPointwiseDirections.load_state_dict(checkpoint['model_state_dict']) if checkpoint.get('model_state_dict',False) else None
        start_epoch = checkpoint['epoch'] if checkpoint.get('epoch',False) else None
        optomizer.load_state_dict(checkpoint['optimizer_state_dict']) if checkpoint.get('optimizer_state_dict',False) else None
        scaler.load_state_dict(checkpoint['scaler']) if checkpoint.get('scaler',False) else None
        scheduler.load_state_dict(checkpoint['scheduler']) if checkpoint.get('scheduler',False) else None
        init_loss = checkpoint['loss'] if checkpoint.get('loss',False) else None
        init_val_loss = checkpoint['val_loss'] if checkpoint.get('val_loss',False) else None
        print(f'starting from epoch {start_epoch}')
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
            temp_loss, loss_dict = train_one_epoch(
                DeepPointwiseDirections,
                epoch,
                train_loader,
                optomizer,
                scaler,
                scheduler,
                args,
                use_amp,
                cfg
            )
            for name, weight in DeepPointwiseDirections.named_parameters():
                writer.add_histogram(name, weight, epoch)
                writer.add_histogram(f"{name}.grad", weight.grad, epoch)
            writer.add_scalar("Loss/total_training", temp_loss, epoch)
            for key in loss_dict.keys():
                writer.add_scalar.add_scalar(f"Loss/train/{loss_dict[key]}", loss_dict[key], epoch)
            torch.cuda.empty_cache()
            #####validating steps
            val_loss, val_loss_dict = validation(DeepPointwiseDirections, test_loader, args, use_amp)
            for key in val_loss_dict.keys():
                writer.add_scalar.add_scalar(f"Loss/val/{val_loss_dict[key]}", val_loss_dict[key], epoch)
            writer.add_scalar("LR/lr", scheduler.get_last_lr()[0], epoch)
            writer.add_scalar("Loss/total_validation", val_loss, epoch)

            if (temp_loss < init_loss) or (epoch % 10 == 0) or (val_loss < init_val_loss):
                os.makedirs(os.path.join(result_dir,'checkpoints'), exist_ok=True)
                save_dict = {
                        "epoch": epoch,
                        "model_state_dict": DeepPointwiseDirections.state_dict(),
                        "optimizer_state_dict": optomizer.state_dict(),
                        'scaler':scaler.state_dict(),
                        "loss": temp_loss,
                        "val_loss": val_loss,
                        'scheduler': scheduler.state_dict(),
                    }
                save_dict.update({'train_'+key:loss_dict[key] for key in loss_dict.keys()})
                save_dict.update({'val_' + key: val_loss_dict[key] for key in val_loss_dict.keys()})
                torch.save(save_dict,
                    os.path.join(result_dir,'checkpoints', f"model-{epoch:03d}-train_loss:{temp_loss:.6f}-val_loss:{val_loss:.6f}.pt",)
                    
                )
                init_loss = min(temp_loss, init_loss)
                init_val_loss = min(val_loss, init_val_loss)
            writer.flush()
        writer.close()

    except KeyboardInterrupt:
        print('saving perf')
        bench_dict.save()
        print('saved perf')
        writer.close()
        quit()
    except Exception:
        raise
        print(traceback.format_exc())
        writer.close()
        quit()


def device_configs(DeepPointwiseDirections, args):
    device = args.device
    device = "cuda" if device == "gpu" else device
    device = device if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_number)
    if device == "cuda":
        DeepPointwiseDirections.cuda()
    return device


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
    loss_dict = {'slackloss':0,'distance_slackloss':0,'distance_loss':0}
    model.train()
    for i, (batch_train_data, batch_direction_label_data, _)  in enumerate(tqdm(start_bench(generator), )):
    #for i,(batch_train_data, batch_direction_label_data, _)  in enumerate(tqdm(generator)):
        opt.zero_grad(set_to_none=True)
        bench_dict['epoch'].step('iter')

        batch_train_data = batch_train_data.half().to(device)
        batch_direction_label_data = batch_direction_label_data.half().to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            y = model(batch_train_data)
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
        bench_dict['epoch'].step('infer')

        scaler.scale(total_loss).backward()
        scaler.step(opt)
        bench_dict['epoch'].step('backprop')
        scaler.update()

        bench_dict['epoch'].gstop()

        if i % 20 == 0:
            print("loss: %f" % (total_loss))
    scheduler.step()

    for key in loss_dict.keys():
        loss_dict[key] = loss_dict[key].cpu()/num_batches_training

    print("trianing_log_epoch_%d" % epoch)
    return acc_loss.cpu() / (num_batches_training), loss_dict


def validation(model, generator, args, use_amp):
    if next(model.parameters()).is_cuda:
        device = "cuda"

    num_batches_testing = len(generator)
    total_loss_esd = 0
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
                loss_esd_ = Loss_torch.slack_based_direction_loss(
                    y, batch_direction_label_data
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
                total_loss_esd += loss_esd_
    for key in loss_dict.keys():
        loss_dict[key] = loss_dict[key].cpu()/num_batches_testing
    return total_loss_esd.cpu() / num_batches_testing, loss_dict


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
