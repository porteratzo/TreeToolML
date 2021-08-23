#%%
from data_gen_utils.all_dataloader import all_data_loader
import numpy as np
from tqdm import tqdm
import os

loader = all_data_loader(onlyTrees=False, preprocess=True, default=False, train_split=True)

# %%
loader.load_all('datasets/custom_data/preprocessed')
savepath = os.path.join('datasets', 'custom_data', 'PDE')
if not os.path.isdir(savepath):
    os.mkdir(savepath)

savepathtrain = os.path.join('datasets', 'custom_data', 'PDE', 'training_data')
if not os.path.isdir(savepathtrain):
    os.mkdir(savepathtrain)

savepathtest = os.path.join('datasets', 'custom_data', 'PDE', 'validating_data')
if not os.path.isdir(savepathtest):
    os.mkdir(savepathtest)
#%%
for i in tqdm(range(10000)):
    while True:
        cluster, labels = loader.get_tree_cluster(train=True)
        array = np.hstack([np.vstack(cluster), np.vstack(labels)]).astype(np.float16)
        if len(array) > 1024*4:
            break

    np.save(os.path.join(savepathtrain, str(i)+'.npy'), array)

for i in tqdm(range(1000)):
    while True:
        cluster, labels = loader.get_tree_cluster(train=False)
        array = np.hstack([np.vstack(cluster), np.vstack(labels)]).astype(np.float16)
        if len(array) > 1024*4:
            break
    np.save(os.path.join(savepathtest, str(i)+'.npy'), array)

# %%
os.getcwd()