#%%
import os
import pandas as pd
from treetoolml.utils.torch_utils import find_training_dir
import matplotlib.pyplot as plt
import numpy as np

#%%
models_multi = set()
models_multi.update(['trunks'])
models_multi.update(['trunks_distance_out_loss','trunks_distance_out','trunks_distance_loss'])

data = {}
metrics = [
    'slackloss',
    'distance_slackloss',
    'distance_loss',
    'distance_slackloss_scaled',
    'average_pred_distance',
    'average_pred_angle',
    'n_ref',
    'n_match',
    'n_extr',
    'Location_RMSE',
    'Completeness',
    'Correctness',
    ]
for name in models_multi:
    model_dir = find_training_dir(name, name=True)
    data[name] = pd.read_csv(os.path.join(model_dir,'test_results.csv'))


# %%
metrics = ['average_pred_angle','Completeness','Correctness']
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
    plt.title(metric_name.replace('_',' '))
    plt.grid(True, linestyle='--', linewidth=0.5, zorder=0) 
    plt.legend(bbox_to_anchor=(1, 1), loc="upper right", ncol=1)
    plt.ylim(np.min(lit)*.9,np.max(lit)*1.1)
    plt.savefig(f'figures/figs/{metric_name}_test_results_datasets.jpg',bbox_inches='tight', dpi=400)
    #plt.show()
# %%
