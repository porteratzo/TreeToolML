# %%
from treetoolml.data.data_gen_utils.all_dataloader import all_data_loader
from treetoolml.data.data_gen_utils.all_dataloader_fullcloud import all_data_loader_cloud
import numpy as np
from tqdm import tqdm
from treetoolml.data.BatchSampleGenerator_torch import tree_dataset_cloud
from porteratzo3D.visualization.open3d_vis import open3dpaint
from porteratzo3D.create_geometries import make_sphere, make_cylinder

# %%
import os

os.chdir("/home/omar/Documents/mine/MY_LIBS/TreeToolML")
# %%
# original datasets
loader = all_data_loader(onlyTrees=False, preprocess=False, default=False, train_split=True)

loader.load_all("datasets/custom_data/preprocessed")
all_trees = {}
for key in tqdm(["open", "paris", "tropical"]):
    all_trees[key] = loader.loader_list[key].get_trees()

# %%

centered_trees = []
for x, key in enumerate(all_trees.keys()):
    for n, tree in enumerate(all_trees[key]):
        centerd_tree = tree - np.multiply(np.min(tree, 0), [0, 0, 1])
        centerd_tree = centerd_tree - np.multiply(np.mean(centerd_tree, axis=0), [1, 1, 0])
        centerd_tree = centerd_tree / np.max(np.linalg.norm(centerd_tree, axis=1))
        centerd_tree = centerd_tree + [n * 0.1, x * 1, 0]
        centered_trees.append(centerd_tree)

#open3dpaint(centered_trees, axis=1)

# %%

loader = all_data_loader_cloud(onlyTrees=False, preprocess=False, default=False, train_split=True)
loader.load_all("datasets/custom_data/full_cloud")
centered_trees = []
grid_size = int(np.ceil(np.sqrt(len(loader.full_cloud.trees))))

for n, tree in enumerate(loader.full_cloud.trees):
    row = n // grid_size
    col = n % grid_size    
    center = loader.full_cloud.centers[n]
    center_points = make_sphere(center, 0.1)
    cylinder = loader.full_cloud.cylinders[n]
    trunk = loader.full_cloud.trunks[n]

    tree = np.concatenate([tree, center_points, cylinder])
    centerd_tree = tree + [col * 1.0, row * 1.0, 0]
    trunys = trunk + [col * 1.0 + 0.01, row * 1.0 + 0.01, 0]
    centered_trees.append(centerd_tree)
    centered_trees.append(trunys)
open3dpaint(centered_trees, axis=1)
[len(i) for i in loader.full_cloud.trees]
quit()
# %%
val_path = "datasets/custom_data/trunks/validating_data"
loader = tree_dataset_cloud(val_path, 4096, normal_filter=True, distances=1, return_centers=True)
# %%
from treetoolml.utils.vis_utils import vis_trees_centers

n = 21
vis_trees_centers(loader[n][0], loader[n][3])
# %%
