#%%
from data_gen_utils.all_dataloader import all_data_loader
import numpy as np
from tqdm import tqdm
import os
from argparse import ArgumentParser


def parse_args(argv=None):
    parser = ArgumentParser()
    parser.add_argument(
        "--training_amount",
        type=int,
        default=10000,
        help="Root path of datasets",
    )
    parser.add_argument(
        "--testing_amount",
        type=int,
        default=2000,
        # windows
        help="Root path config files",
    )
    parser.add_argument(
        "--maxtrees",
        type=int,
        default=4,
        # windows
        help="Root path config files",
    )
    parser.add_argument(
        "--maxdist",
        type=int,
        default=6,
        # windows
        help="Root path config files",
    )
    parser.add_argument(
        "--translationxy",
        type=int,
        default=4,
        # windows
        help="Root path config files",
    )
    parser.add_argument(
        "--translationz",
        type=int,
        default=0.2,
        # windows
        help="Root path config files",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=0.2,
        # windows
        help="Root path config files",
    )
    parser.add_argument(
        "--xyrotation",
        type=int,
        default=0,
        # windows
        help="Root path config files",
    )
    parser.add_argument(
        "--dist_between",
        type=int,
        default=3,
        # windows
        help="Root path config files",
    )
    return parser.parse_args(argv)


def main(args):
    loader = all_data_loader(
        onlyTrees=False, preprocess=False, default=False, train_split=True
    )

    loader.load_all("datasets/custom_data/preprocessed")
    savepath = os.path.join("datasets", "custom_data", "PDE")
    if not os.path.isdir(savepath):
        os.mkdir(savepath)

    savepathtrain = os.path.join("datasets", "custom_data", "PDE", "training_data")
    if not os.path.isdir(savepathtrain):
        os.mkdir(savepathtrain)

    savepathtest = os.path.join("datasets", "custom_data", "PDE", "validating_data")
    if not os.path.isdir(savepathtest):
        os.mkdir(savepathtest)
    #%%
    for i in tqdm(range(10000)):
        while True:
            cluster, labels = loader.get_tree_cluster(
                train=True,
                max_trees=args.maxtrees,
                max_dist=args.maxdist,
                translationxy=args.translationxy,
                translationz=args.translationz,
                scale=args.scale,
                xyrotation=args.xyrotation,
                dist_between=args.dist_between,
            )
            array = np.hstack([np.vstack(cluster), np.vstack(labels)]).astype(
                np.float16
            )
            if len(array) > 1024 * 4:
                break

        np.save(os.path.join(savepathtrain, str(i) + ".npy"), array)

    for i in tqdm(range(2000)):
        while True:
            cluster, labels = loader.get_tree_cluster(train=False)
            array = np.hstack([np.vstack(cluster), np.vstack(labels)]).astype(
                np.float16
            )
            if len(array) > 1024 * 4:
                break
        np.save(os.path.join(savepathtest, str(i) + ".npy"), array)


if __name__ == "__main__":
    args = parse_args()
    main(args)
