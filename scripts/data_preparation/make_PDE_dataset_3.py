# %%
from treetoolml.config.config import combine_cfgs
from treetoolml.utils.default_parser import default_argument_parser
from treetoolml.data.data_gen_utils.all_dataloader import all_data_loader
from treetoolml.data.data_gen_utils.all_dataloader_fullcloud import all_data_loader_cloud
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from shutil import rmtree


class make_dataset_loader(Dataset):
    def __init__(self, loader: all_data_loader_cloud, cfg, size):
        self.loader = loader
        self.cfg = cfg
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        while True:

            return_val = self.loader.get_tree_cluster(
                split="train",
                max_trees=self.cfg.DATA_CREATION.AUGMENTATION.MAX_TREES,
                translation_xy=self.cfg.DATA_CREATION.AUGMENTATION.TRANSLATION_XY,
                translation_z=self.cfg.DATA_CREATION.AUGMENTATION.TRANSLATION_Z,
                min_height=self.cfg.DATA_CREATION.AUGMENTATION.MIN_HEIGHT,
                max_height=self.cfg.DATA_CREATION.AUGMENTATION.MAX_HEIGHT,
                xy_rotation=self.cfg.DATA_CREATION.AUGMENTATION.XY_ROTATION,
                dist_between=self.cfg.DATA_CREATION.AUGMENTATION.MIN_DIST_BETWEEN,
                do_normalize=self.cfg.DATA_CREATION.AUGMENTATION.DO_NORMALIZE,
                center_method=self.cfg.DATA_CREATION.CENTER_METHOD,
                use_trunks=self.cfg.DATA_CREATION.STICK,
                noise=self.cfg.DATA_CREATION.NOISE,
            )
            if self.cfg.DATA_CREATION.STICK:
                cluster, labels, centers, trunks = return_val
                cluster, labels, centers, trunks = (
                    cluster.astype(np.float16),
                    labels.astype(np.float16),
                    [i.astype(np.float16) for i in centers],
                    [i.astype(np.float16) for i in trunks],
                )
            else:
                cluster, labels, centers = return_val
                (
                    cluster,
                    labels,
                    centers,
                ) = (
                    cluster.astype(np.float16),
                    labels.astype(np.float16),
                    [i.astype(np.float16) for i in centers],
                )

            array = np.hstack([cluster, labels])
            if self.cfg.DATA_CREATION.STICK:
                trunk_array = np.vstack(trunks)
            centers = np.hstack([np.vstack(centers), np.vstack(np.unique(np.vstack(labels)))])
            if len(array) > self.cfg.DATA_CREATION.MIN_SIZE:
                break
            else:
                print("too small")
        if self.cfg.DATA_CREATION.STICK:
            return array, [centers], trunk_array
        else:
            return array, [centers]


def main(args):
    cfg_path = args.cfg
    cfg = combine_cfgs(cfg_path, args.opts)

    loader = all_data_loader_cloud(
        onlyTrees=False, preprocess=False, default=False, train_split=True, normal_filter=False
    )
    if cfg.DATA_CREATION.USE_CENTER_FILTERED:
        loader.load_all("datasets/custom_data/full_cloud")
    else:
        loader.load_all("datasets/custom_data/orig_full_cloud")
    savepath = os.path.join(cfg.FILES.DATA_SET, cfg.FILES.DATA_WORK_FOLDER)
    if not os.path.isdir(savepath):
        os.mkdir(savepath)
    else:
        rmtree(savepath)
        os.mkdir(savepath)

    savepathtrain = os.path.join(savepath, "training_data")
    if not os.path.isdir(savepathtrain):
        os.mkdir(savepathtrain)

    savepathtest = os.path.join(savepath, "testing_data")
    if not os.path.isdir(savepathtest):
        os.mkdir(savepathtest)

    savepathval = os.path.join(savepath, "validating_data")
    if not os.path.isdir(savepathval):
        os.mkdir(savepathval)
    # %%
    amounts = [
        cfg.DATA_CREATION.TRAIN_AMOUNT,
        cfg.DATA_CREATION.TEST_AMOUNT,
        cfg.DATA_CREATION.VAL_AMOUNT,
    ]

    paths = [savepathtrain, savepathtest, savepathval]
    for amount, path in zip(amounts, paths):
        generator = DataLoader(
            make_dataset_loader(loader, cfg, amount), num_workers=12, batch_size=1
        )
        try:
            for i, return_var1 in enumerate(tqdm(generator)):
                if cfg.DATA_CREATION.STICK:
                    array, centers, trunks = return_var1
                    inter = zip(array, centers, trunks)
                else:
                    array, centers = return_var1
                    inter = zip(array, centers)
                for return_var2 in inter:
                    if cfg.DATA_CREATION.STICK:
                        _array, _centers, _trunks = return_var2
                    else:
                        _array, _centers = return_var2
                    _centers = np.array(_centers).reshape(-1, 4)
                    if cfg.DATA_CREATION.STICK:
                        np.savez(
                            os.path.join(path, str(i) + ".npz"),
                            cloud=np.array(_array).astype(np.float16),
                            centers=_centers,
                            trunks=_trunks,
                        )
                    else:
                        np.savez(
                            os.path.join(path, str(i) + ".npz"),
                            cloud=np.array(_array).astype(np.float16),
                            centers=_centers,
                        )

        except KeyboardInterrupt:

            quit()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
