#%%

#sys.path.append("../..")
import os
import numpy as np
from treetoolml.benchmark.benchmark_utils import load_eval_results, make_benchmark_metrics
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def main():
    result_dir = os.path.join("results_benchmark", 'treetool')
    EvaluationMetrics = load_eval_results(f'{result_dir}/results.npz')
    BenchmarkMetrics, Methods = make_benchmark_metrics()
    alldata = ["Completeness", "Correctness"]
    dificulties = ["easy", "medium", "hard"]
    plt.figure(figsize=(16, 46))
    for n, i in enumerate(alldata):
        for n2 in range(3):
            plt.subplot(12, 1, n * 3 + n2 + 1)
            plt.title(i + " " + dificulties[n2])
            mine = np.mean(EvaluationMetrics[i][slice(n2, n2 + 2)]) * 100
            print(i, n2, mine)
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
    plt.savefig(f'{result_dir}/fig1.jpg',bbox_inches='tight')

    alldata = ["Location_RMSE", "Diameter_RMSE"]
    dificulties = ["easy", "medium", "hard"]
    plt.figure(figsize=(16, 46))
    for n, i in enumerate(alldata):
        for n2 in range(3):
            plt.subplot(12, 1, n * 3 + n2 + 1)
            plt.title(i + " " + dificulties[n2])
            mine = np.mean(EvaluationMetrics[i][slice(n2, n2 + 2)]) * 100
            print(i, n2, mine)
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
            plt.bar(sortmethods, sortnum, color=sortcol, width=0.2)
            plt.tight_layout(pad=3.0)
            plt.grid(axis="y")
            plt.xticks(rotation=30, fontsize=18)
    plt.savefig(f'{result_dir}/fig2.jpg',bbox_inches='tight')

if __name__ == "__main__":
    main()