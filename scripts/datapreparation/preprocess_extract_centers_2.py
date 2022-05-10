from collections import defaultdict
import enum
import sys
import os

from matplotlib.pyplot import axis

sys.path.append(".")
sys.path.append('/home/omar/Documents/mine/TreeTool')
import TreeToolML.utils.py_util as py_util


import TreeTool.utils as utils
from TreeToolML.config.config import combine_cfgs
from TreeToolML.utils.default_parser import default_argument_parser
from TreeToolML.data.data_gen_utils.all_dataloader import all_data_loader
import numpy as np
from tqdm import tqdm
import pickle

from TreeToolML.utils.tictoc import bench_dict

from TreeToolML.utils.tictoc import bench_dict
from torch.utils.data import Dataset
from TreeToolML.Libraries.open3dvis import open3dpaint, sidexsidepaint

######################
# Extra processing for Paris_lille dataset to remove bad trees
######################


def main(args):
    cfg_path = args.cfg
    cfg = combine_cfgs(cfg_path, args.opts)

    loader = all_data_loader(
        onlyTrees=False, preprocess=False, default=False, train_split=True
    )

    all_datasets = defaultdict(list)

    loader.load_all("datasets/custom_data/preprocessed")
    for dataset_loader in tqdm(loader.tree_list):
        saved_trees = []
        saved_centers = []
        saved_models = []
        saved_filtered = []
        visualization_cylinders = []
        badd_trees = []
        all_trees = dataset_loader.get_trees()
        out_trees = []
        for temp_index_object_xyz in all_trees:
            out_tree = py_util.outliers(temp_index_object_xyz, 100, 10)
            norm_tree = py_util.normalize_2(out_tree)
            down_points = py_util.downsample(norm_tree, 0.005)
            out_trees.append(down_points)

        print("orig tree number", len(all_trees))
        for centered_tree in out_trees:
            filtered_points = py_util.normal_filter(centered_tree, 0.04, 0.4, 0.1)
            if len(filtered_points) == 0:
                continue
            tree_groups = py_util.group_trees(filtered_points, 0.1, 100)
            max_points = 0
            for single_tree in tree_groups:
                index, model = py_util.seg_normals(
                    single_tree,
                    0.04,
                    0.000001,
                    distance=0.01,
                    rlim=[0, 0.05],
                )
                if max_points < len(single_tree[index]):
                    max_points = len(single_tree[index])
                    if (
                        abs(np.dot(model[3:6], [0, 0, 1]) / np.linalg.norm(model[3:6]))
                        > 0.5
                    ):
                        model = np.array(model)
                        Z = 0.2
                        Y = model[1] + model[4] * (Z - model[2]) / model[5]
                        X = model[0] + model[3] * (Z - model[2]) / model[5]
                        model[0:3] = np.array([X, Y, Z])
                        # make sure the vector is pointing upward

                        model[3:6] = utils.similarize(model[3:6], [0, 0, 1])
                        best_model = model
                        best_cylinder = utils.makecylinder(
                            model=model, height=1, density=30
                        )
            if max_points / len(filtered_points) > 0.4:
                print(max_points / len(filtered_points))
                saved_trees.append(centered_tree)
                saved_centers.append(best_model[:3])
                saved_models.append(best_model)
                saved_filtered.append(filtered_points)
                visualization_cylinders.append(best_cylinder)
            else:
                badd_trees.append(centered_tree)
        print("orig tree number", len(saved_trees))
        #sidexsidepaint(saved_trees, visualization_cylinders, [utils.makesphere(i,0.1) for i in saved_centers],pointsize=2, axis=1)
        #sidexsidepaint(saved_filtered, visualization_cylinders, [utils.makesphere(i,0.1) for i in saved_centers],pointsize=2, axis=1)
        #sidexsidepaint(badd_trees,pointsize=2, axis=1)

        all_datasets["cloud"].extend(saved_trees)
        all_datasets["centers"].extend(saved_centers)
        all_datasets["filtered"].extend(saved_filtered)
        all_datasets["cylinders"].extend(visualization_cylinders)
        all_datasets["models"].extend(saved_models)

    point_cloud = np.vstack(all_datasets["cloud"])
    instances = np.hstack(
        [np.ones(len(tree)) * n for n, tree in enumerate(all_datasets["cloud"])]
    )
    labels = np.ones(len(point_cloud))

    from TreeToolML.data.data_gen_utils.dataloaders import (
        save_cloud,
        save_cloud_filtered,
    )

    data_path = "datasets/custom_data/full_cloud/"
    os.makedirs(data_path, exist_ok=True)

    save_cloud(
        data_path + "/" + "full_clouds" + ".ply",
        point_cloud.astype(np.float32),
        labels.astype(np.int32),
        instances.astype(np.float32),
    )

    point_cloud = np.vstack(all_datasets["filtered"])
    instances = np.hstack(
        [np.ones(len(tree)) * n for n, tree in enumerate(all_datasets["filtered"])]
    )
    labels = np.ones(len(point_cloud))

    save_cloud(
        data_path + "/" + "full_filter" + ".ply",
        point_cloud.astype(np.float32),
        labels.astype(np.int32),
        instances.astype(np.float32),
    )

    info_dict = {
        n: {"center": center, "cyl": cyl, "model": model}
        for n, (center, cyl, model) in enumerate(
            zip(
                all_datasets["centers"],
                all_datasets["cylinders"],
                all_datasets["models"],
            )
        )
    }

    with open(data_path + "/" + "info" + ".pk", "wb") as f:
        pickle.dump(info_dict, f)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)

# %%
