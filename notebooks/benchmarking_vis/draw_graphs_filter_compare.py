# %%
import sys

# sys.path.append("../..")
import os

sys.path.append(".")
sys.path.append("/home/omar/Documents/mine/TreeTool")
import numpy as np
from TreeToolML.utils.default_parser import default_argument_parser
from TreeToolML.benchmark.benchmark_utils import load_gt, store_metrics, save_eval_results, load_eval_results, \
    make_benchmark_metrics
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from TreeToolML.config.config import combine_cfgs
from TreeToolML.utils.file_tracability import find_model_dir

# %%
models_multi = []
models_multi.extend(['baseline_no_IOU', 'baseline_no_stem', 'baseline_no_stem_iou'])
models_multi.extend(['baseline', 'baseline_smaller_window_4', 'baseline_smaller_window_6', 'baseline_smaller_window_10'])
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
alldata = ["Completeness", "Correctness"]
dificulties = ["easy", "medium", "hard"]
plt.figure(figsize=(16, 30))
for n, i in enumerate(alldata):
    for n2 in range(3):
        plt.subplot(6, 1, n * 3 + n2 + 1)
        plt.title(i + " " + dificulties[n2])
        mine = {}
        for m in models_multi:
            mine[m] = np.mean(metrics[m][i][slice(n2, n2 + 2)]) * 100
        colors = [np.array(cm.gist_rainbow(i)) * 0.3 if n < len(Methods)-1 else cm.gist_rainbow(i) for
                  n, i in enumerate(np.linspace(0, 1, len(Methods) + len(models_multi)))]
        sortstuff = sorted(
            zip(
                Methods + list(mine.keys()),
                BenchmarkMetrics[i][n2] + list(mine.values()),
                colors,
            ),
            key=lambda x: x[1],
        )
        sortmethods = [i[0] for i in sortstuff]
        sortnum = [i[1] for i in sortstuff]
        sortcol = [i[2] for i in sortstuff]
        # plt.bar(np.arange(len(BenchmarkMetrics[i][n2])+1),BenchmarkMetrics[i][n2]+[mine])
        plt.bar(sortmethods, sortnum, color=sortcol, width=0.2)
        plt.tight_layout(pad=3.0)
        plt.xticks(rotation=30, fontsize=8)
        plt.grid(axis="y")
# plt.savefig(f'{result_dir}/fig1.jpg',bbox_inches='tight')
plt.show()


# %%

# %%

alldata = ["Location_RMSE", "Diameter_RMSE"]
dificulties = ["easy", "medium", "hard"]
plt.figure(figsize=(16, 38))
for n, i in enumerate(alldata):
    for n2 in range(3):
        plt.subplot(6, 1, n * 3 + n2 + 1)
        plt.title(i + " " + dificulties[n2])
        mine = {}
        for m in models_multi:
            mine[m] = np.mean(metrics[m][i][slice(n2, n2 + 2)]) * 100
        colors = [np.array(cm.gist_rainbow(i)) * 0.3 if n < len(Methods) - 1 else cm.gist_rainbow(i) for
                  n, i in enumerate(np.linspace(0, 1, len(Methods) + len(models_multi)))]
        sortstuff = sorted(
            zip(
                Methods + list(mine.keys()),
                BenchmarkMetrics[i][n2] + list(mine.values()),
                colors,
            ),
            key=lambda x: x[1],
        )
        sortmethods = [i[0] for i in sortstuff]
        sortnum = [i[1] for i in sortstuff]
        sortcol = [i[2] for i in sortstuff]
        plt.bar(sortmethods, sortnum, color=sortcol, width=0.2)
        plt.tight_layout(pad=10.0)
        plt.grid(axis="y")
        plt.xticks(rotation=45, fontsize=8)
#plt.savefig(f'{result_dir}/fig2.jpg', bbox_inches='tight')
plt.show()
