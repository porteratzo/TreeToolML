from data_gen_utils.all_dataloader import all_data_loader
from data_gen_utils.dataloaders import save_cloud
from tqdm import tqdm
import copy
import os
from argparse import ArgumentParser


def parse_args(argv=None):
    parser = ArgumentParser()
    parser.add_argument(
        "--save-path",
        type=str,
        default="datasets/custom_data",
        help="Root path of datasets",
    )
    return parser.parse_args(argv)


def main(args):
    loader = all_data_loader(onlyTrees=False, preprocess=True, default=True)

    for keys, i in tqdm(loader.loader_list.items()):

        print(keys)
        i.load_data()

        if not os.path.isdir(args.save_path):
            os.mkdir(args.save_path)

        if not os.path.isdir(args.save_path + "/preprocessed"):
            os.mkdir(args.save_path + "/preprocessed")

        buffer_label = copy.copy(i.labels)
        i.labels[buffer_label == i.tree_label] = 1
        i.labels[buffer_label != i.tree_label] = 0
        save_cloud(
            args.save_path + "/preprocessed/" + keys + ".ply",
            i.point_cloud,
            i.labels,
            i.instances,
        )

if __name__ == "__main__":
    args = parse_args()
    main(args)