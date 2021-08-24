#%%
from data_gen_utils.all_dataloader import all_data_loader
from data_gen_utils.dataloaders import save_cloud
from tqdm import tqdm
import copy
import os

loader = all_data_loader(onlyTrees=False, preprocess=True, default=True)
#%%
for keys, i in tqdm(loader.loader_list.items()):
    print(keys)
    i.load_data()
    
    if not os.path.isdir("datasets/custom_data"):
        os.mkdir("datasets/custom_data")

    if not os.path.isdir("datasets/custom_data/preprocessed"):
        os.mkdir("datasets/custom_data/preprocessed")

    if keys != 'semantic':
        buffer_label = copy.copy(i.labels)
        i.labels[buffer_label==i.tree_label] = 1
        i.labels[buffer_label!=i.tree_label] = 0
        save_cloud('datasets/custom_data/preprocessed/' + keys + '.ply',i.point_cloud,
        i.labels,
        i.instances)
    else:
        for j in range(len(i.point_cloud)):
            buffer_label = copy.copy(i.labels[j])
            i.labels[j][buffer_label==i.tree_label] = 1
            i.labels[j][buffer_label!=i.tree_label] = 0
            save_cloud('datasets/custom_data/preprocessed/' + keys + str(j) + '.ply',i.point_cloud[j],
            i.labels[j],
            i.instances[j])


# %%
