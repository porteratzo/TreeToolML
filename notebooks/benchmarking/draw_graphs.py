#%%
import sys

#sys.path.append("../..")
import os

#os.chdir("../..")
#%%
sys.path.append("/home/omar/Documents/mine/TreeTool")
import numpy as np
from TreeToolML.benchmark.benchmark_utils import load_gt, store_metrics, save_eval_results, load_eval_results, make_benchmark_metrics
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# %%
EvaluationMetrics = load_eval_results()
BenchmarkMetrics, Methods = make_benchmark_metrics()
# %%
alldata = ["Completeness", "Correctness"]
dificulties = ["easy", "medium", "hard"]
plt.figure(figsize=(16, 46))
for n, i in enumerate(alldata):
    for n2 in range(3):
        plt.subplot(12, 1, n * 3 + n2 + 1)
        plt.title(i + " " + dificulties[n2])
        mine = np.mean(EvaluationMetrics[i][slice(n2, n2 + 2)]) * 100
        colors = [np.array(cm.gist_rainbow(i))*0.5 if i!=1 else cm.gist_rainbow(i) for i in np.linspace(0,1,len(Methods)+1)]
        sortstuff = sorted(
            zip(
                Methods + ["our_imp"],
                BenchmarkMetrics[i][n2] + [mine],
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
        plt.xticks(rotation=30, fontsize=18)
        plt.grid(axis="y")
plt.savefig('fig1.jpg',bbox_inches='tight')
# %%
alldata = ["Location_RMSE", "Diameter_RMSE"]
dificulties = ["easy", "medium", "hard"]
plt.figure(figsize=(16, 46))
for n, i in enumerate(alldata):
    for n2 in range(3):
        plt.subplot(12, 1, n * 3 + n2 + 1)
        plt.title(i + " " + dificulties[n2])
        mine = np.mean(EvaluationMetrics[i][slice(n2, n2 + 2)]) * 100
        sortstuff = sorted(
            zip(
                Methods + ["our_imp"],
                BenchmarkMetrics[i][n2] + [mine],
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
        plt.grid(axis="y")
        plt.xticks(rotation=30, fontsize=18)
plt.savefig('fig2.jpg',bbox_inches='tight')
# %%
