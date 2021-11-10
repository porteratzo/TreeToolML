#%%
import sys
sys.path.append('.')
from TreeToolML.config.config import combine_cfgs
from TreeToolML.utils.default_parser import default_argument_parser
from TreeToolML.data.data_gen_utils.all_dataloader import all_data_loader
import numpy as np
from tqdm import tqdm
import os


def main(args):
    cfg_path = args.cfg
    cfg = combine_cfgs(cfg_path, args.opts)

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
    for i in tqdm(range(cfg.DATA_CREATION.TRAIN_AMOUNT)):
        while True:
            cluster, labels = loader.get_tree_cluster(
                train=True,
                max_trees=cfg.DATA_CREATION.AUGMENTATION.MAX_TREES,
                max_dist=cfg.DATA_CREATION.AUGMENTATION.MAX_DIST,
                translation_xy=cfg.DATA_CREATION.AUGMENTATION.TRANSLATION_XY,
                translation_z=cfg.DATA_CREATION.AUGMENTATION.TRANSLATION_Z,
                scale=cfg.DATA_CREATION.AUGMENTATION.SCALE,
                xy_rotation=cfg.DATA_CREATION.AUGMENTATION.XY_ROTATION,
                dist_between=cfg.DATA_CREATION.AUGMENTATION.MIN_DIST_BETWEEN,
            )
            array = np.hstack([np.vstack(cluster), np.vstack(labels)]).astype(
                np.float16
            )
            if len(array) > cfg.DATA_CREATION.MIN_SIZE:
                break

        np.save(os.path.join(savepathtrain, str(i) + ".npy"), array)

    for i in tqdm(range(cfg.DATA_CREATION.TEST_AMOUNT)):
        while True:
            cluster, labels = loader.get_tree_cluster(
                train=False,
                max_trees=cfg.DATA_CREATION.AUGMENTATION.MAX_TREES,
                max_dist=cfg.DATA_CREATION.AUGMENTATION.MAX_DIST,
                translation_xy=cfg.DATA_CREATION.AUGMENTATION.TRANSLATION_XY,
                translation_z=cfg.DATA_CREATION.AUGMENTATION.TRANSLATION_Z,
                scale=cfg.DATA_CREATION.AUGMENTATION.SCALE,
                xy_rotation=cfg.DATA_CREATION.AUGMENTATION.XY_ROTATION,
                dist_between=cfg.DATA_CREATION.AUGMENTATION.MIN_DIST_BETWEEN,
            )
            array = np.hstack([np.vstack(cluster), np.vstack(labels)]).astype(
                np.float16
            )
            if len(array) > cfg.DATA_CREATION.MIN_SIZE:
                break
        np.save(os.path.join(savepathtest, str(i) + ".npy"), array)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
