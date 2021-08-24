#%%
from data_gen_utils.all_dataloader import all_data_loader
from data_gen_utils.dataloaders import save_cloud
from tqdm import tqdm
import copy
import os

loader = all_data_loader(onlyTrees=False, preprocess=True, default=True)
#%%
for keys, i in tqdm(loader.loader_list.items()):
    
    if keys.find('semantic') != -1:
        print(keys)
        i.load_data()
    else:
        print(keys.split('_'))
        path = keys.split('_')[1]
        print(path)
        i.load_data("datasets/Semantic3D/" + path + "*.txt")
    
    if not os.path.isdir("datasets/custom_data"):
        os.mkdir("datasets/custom_data")

    if not os.path.isdir("datasets/custom_data/preprocessed"):
        os.mkdir("datasets/custom_data/preprocessed")

    buffer_label = copy.copy(i.labels)
    i.labels[buffer_label==i.tree_label] = 1
    i.labels[buffer_label!=i.tree_label] = 0
    save_cloud('datasets/custom_data/preprocessed/' + keys + '.ply',i.point_cloud,
    i.labels,
    i.instances)

# %%
