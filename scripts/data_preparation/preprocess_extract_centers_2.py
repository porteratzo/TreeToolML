from collections import defaultdict
import os

import treetoolml.utils.py_util as py_util


import treetool.utils as utils
from treetoolml.utils.default_parser import default_argument_parser
from treetoolml.data.data_gen_utils.all_dataloader import all_data_loader
import numpy as np
from tqdm import tqdm
import pickle
from treetoolml.data.data_gen_utils.dataloaders import (
    save_cloud,
)
import treetool.seg_tree as seg_tree
from scipy.spatial import distance

######################
# Extra processing for Paris_lille dataset to remove bad trees
######################


def main():

    loader = all_data_loader(onlyTrees=False, preprocess=False, default=False, train_split=True)

    centered_datasets = defaultdict(list)
    all_datasets = defaultdict(list)

    loader.load_all("datasets/custom_data/preprocessed")
    for dataset_loader in tqdm(loader.dataset_list):
        good_trees, all_trees = get_tree_data(dataset_loader)
        (
            saved_trees,
            saved_centers,
            saved_models,
            saved_filtered,
            saved_trunk,
            visualization_cylinders,
        ) = good_trees
        (
            all_saved_trees,
            all_saved_centers,
            all_saved_models,
            all_saved_filtered,
            all_saved_trunk,
            all_visualization_cylinders,
        ) = all_trees
        """
        
        sidexsidepaint(
            saved_filtered,
            visualization_cylinders,
            [utils.makesphere(i, 0.1) for i in saved_centers],
            pointsize=2,
            axis=1,
        )
        
        sidexsidepaint(
            saved_trees,
            visualization_cylinders,
            [utils.makesphere(i, 0.1) for i in saved_centers],
            pointsize=2,
            axis=1,
            for_thesis=True
        )
        sidexsidepaint(badd_trees, pointsize=2, axis=1, for_thesis=True)
        """
        centered_datasets["cloud"].extend(saved_trees)
        centered_datasets["trunks"].extend(saved_trunk)
        centered_datasets["centers"].extend(saved_centers)
        centered_datasets["filtered"].extend(saved_filtered)
        centered_datasets["cylinders"].extend(visualization_cylinders)
        centered_datasets["models"].extend(saved_models)

        all_datasets["cloud"].extend(all_saved_trees)
        all_datasets["trunks"].extend(all_saved_trunk)
        all_datasets["centers"].extend(all_saved_centers)
        all_datasets["filtered"].extend(all_saved_filtered)
        all_datasets["cylinders"].extend(all_visualization_cylinders)
        all_datasets["models"].extend(all_saved_models)

    save_datasets(centered_datasets)
    save_datasets(all_datasets, centered=False)


def get_tree_data(dataset_loader):

    saved_trees = []
    saved_trunk = []
    saved_centers = []
    saved_models = []
    saved_filtered = []
    visualization_cylinders = []

    all_saved_trees = []
    all_saved_trunk = []
    all_saved_centers = []
    all_saved_models = []
    all_saved_filtered = []
    all_visualization_cylinders = []
    all_trees = dataset_loader.get_trees()
    print("orig tree number", len(all_trees))
    for temp_index_object_xyz in tqdm(all_trees):
        out_tree = py_util.outliers(temp_index_object_xyz, 100, 10)
        norm_tree = py_util.normalize_2(out_tree)
        # norm_tree = py_util.normalize_2(out_tree)
        down_points = py_util.downsample(norm_tree, 0.004)
        centered_tree = down_points
        filtered_points = py_util.normal_filter(centered_tree, 0.01, 0.3, 0.14)
        if len(filtered_points) == 0:
            continue
        tree_groups = py_util.group_trees(filtered_points, 0.2, 100)
        max_points = 0
        for single_tree in tree_groups:
            if True:
                index, model = py_util.seg_normals(
                    single_tree, distance=0.01, rlim=[0, 0.015], miter=2000
                )
                rg_clusters = seg_tree.region_growing(
                    single_tree,
                    search_radius=20,
                    nn=20,
                    angle_threshold=np.radians(30),
                    curvature_threshold=1,
                )
                rg_clusters = rg_clusters[np.argmax([len(i) for i in rg_clusters])]
                rg_clusters = centered_tree[
                    np.min(distance.cdist(rg_clusters, centered_tree), axis=0) < 0.01
                ]

            if max_points < len(single_tree[index]):
                max_points = len(single_tree[index])
                if abs(np.dot(model[3:6], [0, 0, 1]) / np.linalg.norm(model[3:6])) > 0.5:
                    model = np.array(model)
                    Z = 0.2
                    Y = model[1] + model[4] * (Z - model[2]) / model[5]
                    X = model[0] + model[3] * (Z - model[2]) / model[5]
                    model[0:3] = np.array([X, Y, Z])
                    # make sure the vector is pointing upward

                    model[3:6] = utils.similarize(model[3:6], [0, 0, 1])
                    best_model = model
                    best_cylinder = utils.makecylinder(model=model, height=1, density=30)
                    #best_model[:3] = np.mean(rg_clusters, axis=0)

        if max_points / len(filtered_points) > 0.4:
            print(max_points / len(filtered_points))
            saved_trunk.append(rg_clusters)
            saved_trees.append(centered_tree)
            saved_centers.append(best_model[:3])
            saved_models.append(best_model)
            # saved_filtered.append(filtered_points)
            saved_filtered.append(down_points)
            visualization_cylinders.append(best_cylinder)
        all_saved_trunk.append(rg_clusters)
        all_saved_trees.append(centered_tree)
        all_saved_centers.append(best_model[:3])
        all_saved_models.append(best_model)
        # all_saved_filtered.append(filtered_points)
        all_saved_filtered.append(down_points)
        all_visualization_cylinders.append(best_cylinder)
    print("new tree number", len(saved_trees))
    return [
        saved_trees,
        saved_centers,
        saved_models,
        saved_filtered,
        saved_trunk,
        visualization_cylinders,
    ], [
        all_saved_trees,
        all_saved_centers,
        all_saved_models,
        all_saved_filtered,
        all_saved_trunk,
        all_visualization_cylinders,
    ]


def save_datasets(all_datasets, centered=True):
    point_cloud = np.vstack(all_datasets["cloud"])
    instances = np.hstack([np.ones(len(tree)) * n for n, tree in enumerate(all_datasets["cloud"])])
    labels = np.ones(len(point_cloud))

    if centered:
        data_path = "datasets/custom_data/full_cloud/"
    else:
        data_path = "datasets/custom_data/orig_full_cloud/"
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
        n: {"center": center, "cyl": cyl, "model": model, "trunks": trunk}
        for n, (center, cyl, model, trunk) in enumerate(
            zip(
                all_datasets["centers"],
                all_datasets["cylinders"],
                all_datasets["models"],
                all_datasets["trunks"],
            )
        )
    }

    with open(data_path + "/" + "info" + ".pk", "wb") as f:
        pickle.dump(info_dict, f)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main()

# %%
