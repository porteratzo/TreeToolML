import time
from collections import defaultdict
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


class timer:
    def __init__(self) -> None:
        self.clock_time = time.perf_counter()

    def tic(self):
        self.clock_time = time.perf_counter()

    def toc(self):
        return time.perf_counter() - self.clock_time

    def ttoc(self):
        val = self.toc()
        self.tic()
        return val

    def ptoc(self, message=None):
        print(message, self.toc())

    def pttoc(self, message=None):
        print(message, self.ttoc())


class benchmarker:
    def __init__(self, file="performance/base") -> None:
        self.step_timer = timer()
        self.global_timer = timer()
        self.global_dict = []
        self.step_dict = defaultdict(int)
        self.file = file
        self.folder = "/".join(file.split("/")[:-1])
        self.started = False

    def start(self):
        self.step_timer.tic()
        self.global_timer.tic()
        self.started = True

    def gstep(self):
        if self.started:
            if not "global" in self.step_dict.keys():
                self.step_dict["global"] = self.global_timer.ttoc()
        else:
            self.start()
        self.step_dict = defaultdict(int)
        self.start()

    def gstop(self):
        self.step_dict["global"] = self.global_timer.ttoc()
        self.global_dict.append(self.step_dict)

    def step(self, topic=""):
        self.step_dict[topic] += self.step_timer.ttoc()

    def save_data(self):
        df = pd.DataFrame(self.global_dict)
        os.makedirs(self.folder, exist_ok=True)
        df.to_csv(self.file + ".csv")

    def data_summary(self):
        self.series = defaultdict(dict)
        for n, i in enumerate(self.global_dict):
            for key in i.keys():
                self.series[key][n] = i[key]
        means = {
            k: [np.mean(list(self.series[k].values()))] for k, v in self.series.items()
        }
        df = pd.DataFrame().from_dict(means)
        os.makedirs(self.folder, exist_ok=True)
        df.to_csv(self.file + "_summary.csv")

    def plot_data(self):
        self.data_summary()
        series = self.series
        plt.figure(figsize=(9, 3))
        plt.title(os.path.basename(self.file))
        for keys in series.keys():
            X = series[keys].keys()
            Y = series[keys].values()
            plt.plot(X, Y)
        plt.legend(list(series.keys()))
        plt.savefig(self.file + ".png", dpi=200)
        
g_bench1 = benchmarker("performance/bench1")
g_bench2 = benchmarker("performance/bench2")
g_bench3 = benchmarker("performance/bench3")

g_timer1 = timer()
g_timer2 = timer()
g_timer3 = timer()