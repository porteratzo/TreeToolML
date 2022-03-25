from logging import exception
import os
import sys

sys.path.append(".")
sys.path.append("/home/omar/Documents/Mine/Git/TreeTool")
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
from TreeToolML.data.BatchSampleGenerator_torch import tree_dataset
from TreeToolML.model.build_model import build_model
from TreeToolML.utils.default_parser import default_argument_parser
from TreeToolML.utils.file_tracability import get_model_dir
import traceback

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
                        self.n += 1
                        print('error')
                return result
            else:
                raise StopIteration


def main(args):
    cfg_path = args.cfg
    cfg = combine_cfgs(cfg_path, args.opts)
    use_amp = args.amp != 0

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
    generator_training = tree_dataset(train_path, cfg.TRAIN.N_POINTS, normal_filter=True)
    generator_val = tree_dataset(val_path, cfg.TRAIN.N_POINTS, normal_filter=True)
    ###optimizer--Adam

    init_loss = 999.999
    init_val_loss = 999.999

    try:
        for epoch in range(cfg.TRAIN.EPOCHS):

            ####training data generator
            
            train_loader = DataLoader(
                generator_training, cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=6
            )

            ####validating data generator
            test_loader = DataLoader(
                generator_val, cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=6
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
                use_amp
            )
            for name, weight in DeepPointwiseDirections.named_parameters():
                writer.add_histogram(name, weight, epoch)
                writer.add_histogram(f"{name}.grad", weight.grad, epoch)
            writer.add_scalar("Loss/traning", temp_loss, epoch)
            torch.cuda.empty_cache()
            #####validating steps
            val_loss = validation(DeepPointwiseDirections, test_loader, args, use_amp)
            writer.add_scalar("LR/lr", scheduler.get_last_lr()[0], epoch)
            writer.add_scalar("Loss/validation", val_loss, epoch)

            if (temp_loss < init_loss) or (epoch % 10 == 0) or (val_loss < init_val_loss):
                os.makedirs(os.path.join(result_dir,'checkpoints'), exist_ok=True)
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

    except KeyboardInterrupt:
        print('saving perf')
        bench_dict.save()
        print('saved perf')
        writer.close()
        quit()
    except Exception:
        print(traceback.format_exc())
        writer.close()
        quit()


def train_one_epoch(model, epoch, generator, opt, scaler, scheduler, args, use_amp):
    """ops: dict mapping from string to tf ops"""
    if next(model.parameters()).is_cuda:
        device = "cuda"
    else:
        device = "cpu"

    num_batches_training = len(generator)
    print("-----------------training--------------------")
    print("training steps: %d" % num_batches_training)

    total_loss = 0
    acc_loss = 0
    model.train()
    for i, (batch_train_data, batch_direction_label_data, _)  in enumerate(tqdm(start_bench(generator))):
    #for i,(batch_train_data, batch_direction_label_data, _)  in enumerate(tqdm(generator)):
        opt.zero_grad()
        bench_dict['epoch'].step('iter')

        batch_train_data = batch_train_data.half().to(device)
        batch_direction_label_data = batch_direction_label_data.half().to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            y = model(batch_train_data)
            total_loss = Loss_torch.slack_based_direction_loss(
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

    print("trianing_log_epoch_%d" % epoch)
    return acc_loss.cpu() / (num_batches_training)


def validation(model, generator, args, use_amp):
    if next(model.parameters()).is_cuda:
        device = "cuda"

    num_batches_testing = len(generator)
    total_loss_esd = 0
    gen = iter(generator)
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
                total_loss_esd += loss_esd_
    return total_loss_esd.cpu() / num_batches_testing


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
