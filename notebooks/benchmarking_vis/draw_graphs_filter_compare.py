# %%
import os
import numpy as np
from treetoolml.benchmark.benchmark_utils import load_eval_results, \
    make_benchmark_metrics
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from treetoolml.utils.file_tracability import find_model_dir

# %%
#get evaluation metrics

models_multi = set()
#models_multi.update(['baseline_no_IOU', 'baseline_no_stem', 'baseline_no_stem_iou', 'baseline_group_stems'])
#models_multi.update(['baseline', 'baseline_smaller_window_4', 'baseline_smaller_window_6', 'baseline_smaller_window_10'])
#models_multi.update(['trunks', 'distance', 'distance_loss',])
#models_multi.update(['distance_loss', 'distance_loss_distance_filter_001', 'distance_loss_distance_filter_005'])
#models_multi.update(['distance_out_loss_scale_05', 'distance_out_loss_scale_00', 'distance_out_loss_scale_03','distance_out_loss_scale_08'])
#models_multi.update(['distance_out_loss_clustering', 'distance_out_loss'])
#models_multi.update(['group', 'IOU', 'stems_IOU', 'stems','none'])
#models_multi.update(['little','NORMAL'])
#models_multi.update(['distance_loss_distance_filter_005','distance_loss_distance_filter_01','distance_loss_distance_filter_02'])
#models_multi.update(['distance_loss_distance_filter_005_seg','distance_loss_distance_filter_01_seg','distance_loss_distance_filter_02_seg'])
models_multi.update(['distance_loss'])
#models_multi.update(['distance_loss','RRFSegNet'])
#models_multi.update(['distance_loss','distance_out_loss','distance_out_loss_scale','distance_out'])
metrics = {}
BenchmarkMetrics, Methods = make_benchmark_metrics()
for model in list(models_multi):
    file_name = model
    result_dir = os.path.join("results_benchmark", file_name)
    result_dir = os.path.join(find_model_dir(result_dir), 'results.npz')
    if os.path.isfile(result_dir):
        EvaluationMetrics = load_eval_results(result_dir)
        metrics[model] = EvaluationMetrics
    else:
        print(model, 'not found')
    

# %%
#get Completeness graph 

alldata = ["Completeness","Correctness","Location_RMSE", "Diameter_RMSE"]
dificulties = ["easy", "medium", "hard"]
scale = 4
from collections import defaultdict
import pandas as pd
results = defaultdict(dict)
for plot_number, metric_name in enumerate(alldata):
    plt.figure(figsize=(14*scale, 10*scale))
    for dificulty_n in range(3):
        plt.subplot(3, 1, dificulty_n + 1)
        plt.title(metric_name + " " + dificulties[dificulty_n])
        mine = {}
        for m in models_multi:
            if  m == 'distance_loss':
                _m = 'TreeToolML'
            else:
                _m = m
            if metric_name == 'Diameter_RMSE':
                metric = metrics[m]['Diameter_RMSE_C']
            else:
                metric = metrics[m][metric_name]
            mine[_m] = np.mean(metric[slice(dificulty_n, dificulty_n + 2)]) * 100
            results[metric_name + '_' +  dificulties[dificulty_n]][m] = mine[_m] / 100
        colors = [np.array(cm.gist_rainbow(i)) * 0.3 if n < len(Methods) - 1 else cm.gist_rainbow(i) for
                  n, i in enumerate(np.linspace(0, 1, len(Methods) + len(models_multi)))]
        sortstuff = sorted(
            zip(
                Methods + list(mine.keys()),
                BenchmarkMetrics[metric_name][dificulty_n] + list(mine.values()),
                colors,
            ),
            key=lambda x: x[1],
        )
        if plot_number > 1:
            sortstuff = sortstuff[::-1]
        sortmethods = [i[0] for i in sortstuff]
        sortnum = [i[1] for i in sortstuff]
        sortcol = [i[2] for i in sortstuff]
        # plt.bar(np.arange(len(BenchmarkMetrics[i][n2])+1),BenchmarkMetrics[i][n2]+[mine])
        plt.bar(sortmethods, sortnum, color=sortcol, width=0.4)
        plt.tight_layout(pad=3.0)
        plt.xticks(rotation=-65, fontsize=18)
        plt.grid(axis="y")
        if plot_number < 2:
            plt.yticks(np.arange(0,100,10))
        else:
            plt.yticks(np.linspace(0,np.max(sortnum),8))
        print(metric_name + " " + dificulties[dificulty_n])
        print([[i,j] for i,j in zip(sortmethods,sortnum) if i in ['TreeToolML','TreeTool']])
    plt.savefig(f'{metric_name}.jpg',bbox_inches='tight', dpi=400)
    #plt.show()

# %%
