from TreeToolML.datapreparation.data_gen_utils.all_dataloader import all_data_loader
from TreeToolML.datapreparation.data_gen_utils.dataloaders import save_cloud
from TreeToolML.config.config import combine_cfgs
from TreeToolML.utils.default_parser import default_argument_parser
from tqdm import tqdm
import copy
import os


def main(args):
    cfg_path = args.cfg
    cfg = combine_cfgs(cfg_path, args.opts)
    loader = all_data_loader(onlyTrees=False, preprocess=True, default=True)
    data_path = cfg.DATA_PREPROCESSING.DATA_PATH + "/preprocessed"

    for keys, i in tqdm(loader.loader_list.items()):

        print(keys)
        i.load_data()

        if not os.path.isdir():
            os.makedirs(data_path, exist_ok=True)

        buffer_label = copy.copy(i.labels)
        i.labels[buffer_label == i.tree_label] = 1
        i.labels[buffer_label != i.tree_label] = 0
        save_cloud(
            data_path + "/" + keys + ".ply",
            i.point_cloud,
            i.labels,
            i.instances,
        )

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)