import numpy as np
import random
import math
import os
import sys
import pclpy
import open3d as o3d
from TreeToolML.utils.tictoc import bench_dict

sys.path.append("/home/omar/Documents/Mine/Git/TreeTool")
import TreeTool.seg_tree as seg_tree

def make_metrics_dict():
    EvaluationMetrics = {}
    EvaluationMetrics["Completeness"] = []
    EvaluationMetrics["Correctness"] = []
    EvaluationMetrics["Mean_AoD"] = []
    EvaluationMetrics["Diameter_RMSE"] = []
    EvaluationMetrics["Diameter_RMSE_E"] = []
    EvaluationMetrics["Diameter_RMSE_C"] = []
    EvaluationMetrics["Diameter_bias"] = []
    EvaluationMetrics["Location_RMSE"] = []
    EvaluationMetrics["Location_bias"] = []
    EvaluationMetrics["Relative_Diameter_RMSE"] = []
    EvaluationMetrics["Relative_Diameter_bias"] = []
    EvaluationMetrics["Relative_Location_RMSE"] = []
    EvaluationMetrics["Relative_Location_bias"] = []
    EvaluationMetrics["n_ref"] = []
    EvaluationMetrics["n_match"] = []
    EvaluationMetrics["n_extr"] = []
    EvaluationMetrics["location_y"] = []
    EvaluationMetrics["diameter_y"] = []
    return EvaluationMetrics