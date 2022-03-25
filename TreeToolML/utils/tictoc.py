import os
import time
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

try:
    import pandas as pd

    no_pandas = False
except:
    no_pandas = True


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


class timed_counter:
    def __init__(self, enabled=True) -> None:
        if enabled:
            self.timer = timer()
            self.counter = 0
            self.stop_time = 0
            self.stop_count = 0
        self.enabled = enabled

    def start(self):
        if self.enabled:
            if self.counter == 0:
                self.timer.tic()

    def stop(self):
        if self.enabled:    
            self.stop_time = self.timer.toc()
            self.stop_count = self.counter

    def count(self):
        if self.enabled:
            self.counter += 1

    def get_frequency(self):
        if self.enabled:
            if self.stop_time == 0:
                return self.counter/self.timer.toc()
            else:
                return self.stop_count/self.stop_time

    def reset(self):
        if self.enabled:
            self.timer.tic()
            self.counter = 0
    
    def disable(self):
        self.enabled = False


class benchmarker:
    def __init__(self, file="performance/base") -> None:
        self.enable = True
        self.step_timer = timer()
        self.global_timer = timer()
        self.global_dict = []
        self.step_dict = defaultdict(int)
        self.file = file
        self.folder = "/".join(file.split("/")[:-1])
        self.started = False

    def enable(self):
        self.enable = True

    def disable(self):
        self.enable = False

    def start(self):
        if self.enable:
            self.step_timer.tic()
            self.global_timer.tic()
            self.started = True

    def gstep(self):
        if self.enable:
            self.gstop()
            self.step_dict = defaultdict(int)
            self.start()

    def gstop(self):
        if self.enable:
            if self.started:
                if not "global" in self.step_dict.keys():
                    self.step_dict["global"] = self.global_timer.ttoc()
                self.global_dict.append(self.step_dict)
                self.started = False

    def step(self, topic=""):
        if self.enable:
            self.step_dict[topic] += self.step_timer.ttoc()

    def data_summary(self):
        if self.enable:
            self.gstop()

            self.series = defaultdict(dict)
            for n, i in enumerate(self.global_dict):
                for key in i.keys():
                    self.series[key][n] = i[key]
            means = {
                k: [np.mean(list(self.series[k].values()))]
                for k, v in self.series.items()
            }
            if not no_pandas:
                df = pd.DataFrame().from_dict(means)
                os.makedirs(self.folder, exist_ok=True)
                df.to_csv(self.file + "_summary.csv")

    def plot_data(self):
        if self.enable:
            self.data_summary()
            series = self.series
            plt.figure(figsize=(18, 6))
            plt.title(os.path.basename(self.file))
            colormap = plt.cm.nipy_spectral
            color_cycle = [colormap(i) for i in np.linspace(0, 1, len(series))]
            for n, keys in enumerate(series.keys()):
                X = np.array(list(series[keys].keys()))
                Y = np.array(list(series[keys].values()))

                Q1 = np.percentile(Y, 25, interpolation="midpoint")

                Q3 = np.percentile(Y, 75, interpolation="midpoint")
                IQR = Q3 - Q1
                bool_idx = (Y < (IQR + 1.5 * Q3)) & (Y > (IQR - 1.5 * Q1))
                X = X[bool_idx]
                Y = Y[bool_idx]

                plt.plot(X, Y, color=color_cycle[n])
            plt.tight_layout()
            plt.legend(list(series.keys()))
            plt.savefig(self.file + ".png", dpi=200)


class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


class g_benchmarker:
    def __init__(self) -> None:
        self.benchmarkers = {}
        self.enable = True
        today = datetime.now()
        self.time_string = today.strftime("%d:%m:%Y:%H:%M")

    def enable(self):
        self.enable = True
        for bench in self.benchmarkers.values():
            bench.enable()

    def disable(self):
        self.enable = False
        for bench in self.benchmarkers.values():
            bench.disable()

    def __getitem__(self, item):
        get_bench = self.benchmarkers.get(item, None)
        if get_bench is None:
            self.benchmarkers[item] = benchmarker(f"performance_{self.time_string}/{item}")
        return self.benchmarkers[item]

    def save(self):
        if self.enable:
            for bench in self.benchmarkers.values():
                bench.plot_data()


bench_dict = g_benchmarker()

g_timer1 = timer()
g_timer2 = timer()
g_timer3 = timer()
