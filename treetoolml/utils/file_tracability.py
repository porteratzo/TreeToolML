import os
import numpy as np
from datetime import datetime
from glob import glob
import re


def get_checkpoint_file(partial_path, file_type="LATEST", max=None):
    cp_files, val_acc, file_mod_time = get_checkpoint_list(partial_path, max)
    if file_type == "LATEST":
        weights_file = cp_files[np.argmax(file_mod_time)]
        weights_file = os.path.join(partial_path, weights_file)
    elif file_type == "BEST":
        weights_file = cp_files[np.argmin(val_acc)]
        weights_file = os.path.join(partial_path, weights_file)
    else:
        weights_file = os.path.join(partial_path, file_type)
    return weights_file


def get_checkpoint_list(partial_path, max = None):
    cp_files = os.listdir(partial_path)
    if max is not None:
        pattern = r"model-(\d+)-train_loss"
        iter_count = np.array(
            [int(re.search(pattern, cp_file).group(1)) for cp_file in cp_files]
        )
        cp_files = np.array(cp_files)[iter_count < max]

    val_acc = [
        float(os.path.splitext(cp_file.split("-")[3])[0].split(":")[1])
        for cp_file in cp_files
    ]
    file_mod_time = [
        os.path.getmtime(os.path.join(partial_path, cp_file)) for cp_file in cp_files
    ]

    return cp_files, val_acc, file_mod_time


def get_model_dir(base_dir):
    today = datetime.now()
    time_string = today.strftime("%d:%m:%Y:%H:%M")

    return base_dir + "_" + time_string


def get_dir_list(base_dir):
    cp_files = glob(base_dir + "_[0-9][0-9]:*")
    assert len(cp_files) > 0, f"No directories containing {base_dir} found"
    date_times = [os.path.basename(file_name) for file_name in cp_files]
    date_order = []
    NA_dates = []
    for date_str in date_times:
        date_ = date_str.split("_")[-1]
        if is_our_datetime(date_):
            date_order.append(date_)
        else:
            NA_dates.append(date_)

    date_objects = []
    for date_str in date_order:
        day, month, year, hour, minute = date_str.split(":")
        day, month, year, hour, minute = [
            int(i) for i in [day, month, year, hour, minute]
        ]
        date_objects.append(datetime(year, month, day, hour, minute))

    _, final_dates = zip(*sorted(zip(date_objects, date_order), reverse=True))
    final_dates = list(final_dates)
    final_dates.extend(NA_dates)

    return final_dates


def is_our_datetime(subject):
    return (
        (subject.find(":") != -1)
        and (len(subject.split(":")) == 5)
        and all([i.isdecimal() for i in subject.split(":")])
    )


def find_model_dir(base_dir, file_type="LATEST"):
    dir_list = get_dir_list(base_dir)
    if file_type == "LATEST":
        final_path = dir_list[0]
        final_path = "_".join([base_dir, final_path])
    elif is_our_datetime(file_type):
        final_path = dir_list[dir_list.index(file_type)]
        final_path = "_".join([base_dir, final_path])
    else:
        final_path = os.path.join(os.path.split(base_dir)[0], file_type)
    return final_path


def create_training_dir(cfg, model_name):
    model_name = get_model_dir(model_name)
    result_dir = os.path.join("results_training", model_name, "trained_model")
    os.makedirs(result_dir, exist_ok=True)
    result_config_path = os.path.join("results_training", model_name, "full_cfg.yaml")
    cfg_str = cfg.dump()
    with open(result_config_path, "w") as f:
        f.write(cfg_str)
    return result_dir
