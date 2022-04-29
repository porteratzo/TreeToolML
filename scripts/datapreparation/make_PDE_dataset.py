#%%
import pclpy
import open3d as o3d
from json import load
import sys

sys.path.append(".")
from TreeToolML.config.config import combine_cfgs
from TreeToolML.utils.default_parser import default_argument_parser
from TreeToolML.data.data_gen_utils.all_dataloader import all_data_loader
from TreeToolML.data.data_gen_utils.all_dataloader_fullcloud import all_data_loader_cloud
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from TreeToolML.utils.tictoc import bench_dict
from TreeToolML.Libraries.open3dvis import open3dpaint

bench_dict.disable()


class make_dataset_loader(Dataset):
    def __init__(self, loader, cfg, size):
        self.loader = loader
        self.cfg = cfg
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        while True:
            bench_dict["train_data"].gstep()
            cluster, labels, centers = self.loader.get_tree_cluster(
                split="train",
                max_trees=self.cfg.DATA_CREATION.AUGMENTATION.MAX_TREES,
                translation_xy=self.cfg.DATA_CREATION.AUGMENTATION.TRANSLATION_XY,
                translation_z=self.cfg.DATA_CREATION.AUGMENTATION.TRANSLATION_Z,
                scale=self.cfg.DATA_CREATION.AUGMENTATION.SCALE,
                xy_rotation=self.cfg.DATA_CREATION.AUGMENTATION.XY_ROTATION,
                dist_between=self.cfg.DATA_CREATION.AUGMENTATION.MIN_DIST_BETWEEN,
                do_normalize=self.cfg.DATA_CREATION.AUGMENTATION.DO_NORMALIZE,
            )
            cluster, labels, centers = (
                [i.astype(np.float16) for i in cluster],
                [i.astype(np.float16) for i in labels],
                [i.astype(np.float16) for i in centers],
            )
            bench_dict["train_data"].step("get cluster")
            array = np.hstack([np.vstack(cluster), np.vstack(labels)])
            centers = np.hstack([np.vstack(centers), np.vstack(np.unique(np.vstack(labels)))])
            if len(array) > self.cfg.DATA_CREATION.MIN_SIZE:
                break
            else:
                print("too small")
        return array, [centers]


def main(args):
    cfg_path = args.cfg
    cfg = combine_cfgs(cfg_path, args.opts)

    loader = all_data_loader_cloud(
        onlyTrees=False, preprocess=False, default=False, train_split=True
    )

    loader.load_all("datasets/custom_data/full_cloud")
    savepath = os.path.join(cfg.FILES.DATA_SET, cfg.FILES.DATA_WORK_FOLDER)
    if not os.path.isdir(savepath):
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
    #%%
    amounts = [
        cfg.DATA_CREATION.TRAIN_AMOUNT,
        cfg.DATA_CREATION.TEST_AMOUNT,
        cfg.DATA_CREATION.VAL_AMOUNT,
    ]

    paths = [savepathtrain, savepathtest, savepathval]
    for amount, path in zip(amounts, paths):
        generator = DataLoader(
            make_dataset_loader(loader, cfg, amount), num_workers=4, batch_size=1
        )
        try:
            for i, (array, centers) in enumerate(tqdm(generator)):
                for _array, _centers in zip(array, centers):
                    _centers = np.array(_centers).reshape(-1,4)
                    np.savez(
                        os.path.join(path, str(i) + ".npz"),
                        cloud=np.array(_array).astype(np.float16),
                        centers=_centers,
                    )
                    bench_dict["train_data"].step("save")
                    bench_dict["train_data"].gstop()
        except KeyboardInterrupt:
            bench_dict.save()
            quit()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
