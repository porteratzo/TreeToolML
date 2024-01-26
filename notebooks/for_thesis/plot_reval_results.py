#%%
import os
import pandas as pd
from treetoolml.utils.torch_utils import find_training_dir
import matplotlib.pyplot as plt


#%%
models_multi = set()
models_multi.update(['trunks','center_filtered','original'])
data = {}
metrics = ['slackloss','distance_slackloss','distance_loss','distance_slackloss_scaled']
for name in models_multi:
    model_dir = find_training_dir(name, name=True)
    data[name] = pd.read_csv(os.path.join(model_dir,'reval_results.csv'))

# %%

for metric_name in metrics:
    plt.figure()
    for name in data.keys():
        x = data[name]['epoch']
        y = data[name][metric_name]
        plt.plot(x,y,label=name)
    plt.title(metric_name)
    plt.legend()
    plt.savefig(f'figures/{metric_name}_reval_datasets.jpg',bbox_inches='tight', dpi=400)
    #plt.show()

# %%
models_multi = set()
models_multi.update(['trunks','trunks_distance_loss','trunks_distance_out_loss','trunks_distance_out'])
data = {}
metrics = ['slackloss','distance_slackloss','distance_loss','distance_slackloss_scaled']
for name in models_multi:
    model_dir = find_training_dir(name, name=True)
    data[name] = pd.read_csv(os.path.join(model_dir,'reval_results.csv')).drop_duplicates('epoch')

# %%
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np
for metric_name in metrics:
    plt.figure()
    for name in data.keys():
        x = data[name]['epoch']
        y = gaussian_filter1d(data[name][metric_name], sigma=2)
        plt.plot(x,y,label=name)
    plt.title(metric_name)
    plt.legend()
    plt.savefig(f'figures/{metric_name}_reval.jpg',bbox_inches='tight', dpi=400)
    #plt.show()
# %%
