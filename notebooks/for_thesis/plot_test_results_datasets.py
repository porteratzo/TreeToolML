#%%
import os
import pandas as pd
from treetoolml.utils.torch_utils import find_training_dir
import matplotlib.pyplot as plt
import numpy as np

#%%
models_multi = set()
models_multi.update(['trunks'])

data = {}
for name in models_multi:
    model_dir = find_training_dir(name, name=True)
    data[name] = pd.read_csv(os.path.join(model_dir,'test_results.csv'))
# %%
#data = {}
models_multi = set()
models_multi.update(['center_filtered','original'])
for name in models_multi:
    model_dir = find_training_dir(name, name=True)
    data[name] = pd.read_csv(os.path.join(model_dir,'test_results_centers.csv'))

metrics = ['average_pred_angle','average_pred_angle_xy','Completeness','Correctness']
angle_met = ['average_pred_angle','average_pred_angle_xy']
angle_y = ['degrees', 'percentage']
for metric_name in metrics:
    if metric_name == 'Unnamed: 0':
        continue
    plt.figure()
    lit = []
    for n,name in enumerate(data.keys()):
        x = n
        try:
            y = data[name][metric_name].iloc()[0]
            y = np.rad2deg(y) if metric_name in angle_met else y*100
        except KeyError:
            continue
        lit.append(y)
        plt.bar(x,y,label=name.replace('_',' '))
        plt.text(x, y, str(round(y,3)), ha='center', va='bottom')
    plt.xlabel('Dataset')
    plt.ylabel('degrees' if metric_name in angle_met else 'percentage')
    plt.xticks([])
    title_name = metric_name.replace('_',' ').replace('pred','predicted').replace('xy','XY').replace('angle','angle error')
    plt.title(title_name)
    plt.grid(True, linestyle='--', linewidth=0.5, zorder=0) 
    plt.legend(bbox_to_anchor=(1, 1), loc="upper right", ncol=1)
    plt.ylim(np.min(lit)*.9,np.max(lit)*1.1)
    plt.savefig(f'figures/dataset_figs/c{metric_name}_test_results_datasets.jpg',bbox_inches='tight', dpi=400)
# %%
