from collections import defaultdict

#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pclpy
import torch
import treetool.seg_tree as seg_tree
import treetool.utils as utils
from tqdm import tqdm
from porteratzolibs.visualization_o3d.create_geometries import make_arrow, make_cylinder
from treetoolml.IndividualTreeExtraction_utils.center_detection.center_detection_vis import \
    center_detection
from treetoolml.IndividualTreeExtraction_utils.PointwiseDirectionPrediction_torch import \
    prediction
from treetoolml.utils.py_util import (data_preprocess,
                                      get_center_scale, shuffle_data)
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance

def make_benchmark_metrics():
    Methods = [
        "CAF",
        "TUDelft",
        "FGI",
        "IntraIGN",
        "RADI",
        "NJU",
        "Shinshu",
        "SLU",
        "TUZVO",
        "TUWien",
        "RILOG",
        "TreeMetrics",
        "UofL",
        "WHU",
        "TreeTool",
    ]
    CompletenessEasy = [88, 66, 94, 84, 74, 88, 89, 94, 87, 82, 96, 36, 69, 89,81, ]
    CompletenessMedium = [75, 49, 88, 65, 59, 81, 78, 88, 74, 68, 87, 27, 58, 80,64, ]
    CompletenessHard = [44, 16, 66, 27, 25, 45, 46, 64, 39, 39, 63, 18, 37, 52,54, ]

    CorrectnessEasy = [96, 69, 90, 97, 94, 44, 77, 86, 94, 95, 77, 99, 86, 86, 84]
    CorrectnessMedium = [99, 61, 89, 98, 97, 44, 76, 91, 95, 96, 89, 99, 91, 88, 84]
    CorrectnessHard = [99, 41, 93, 97, 97, 70, 74, 91, 95, 97, 88, 99, 91, 90, 90]

    StemRMSEEasy = [
        2.2,
        15.4,
        2.8,
        1.2,
        3.2,
        13.2,
        5.2,
        2.0,
        2.0,
        1.6,
        3.2,
        3.0,
        8.8,
        5.8,
        1.5,
    ]
    StemRMSEMedium = [
        3.2,
        18.2,
        3.0,
        4.0,
        4.8,
        20.0,
        8.0,
        4.2,
        3.6,
        2.6,
        6.4,
        2.6,
        11.6,
        8.2,
        2.5,
    ]
    StemRMSEHard = [
        4.0,
        27.3,
        6.4,
        7.0,
        8.0,
        20.0,
        12.0,
        6.6,
        8.2,
        4.8,
        11.2,
        2.4,
        14.4,
        12.0,
        2.6,
    ]

    DBHRMSEEasy = [
        2.0,
        12.8,
        1.4,
        1.6,
        2.0,
        21.0,
        4.8,
        1.8,
        2.2,
        1.4,
        8.6,
        1.2,
        6.2,
        7.2,
        2.6,
    ]
    DBHRMSEMedium = [
        2.2,
        12.2,
        1.8,
        3.4,
        4.0,
        24.0,
        8.0,
        3.2,
        3.4,
        1.6,
        11.0,
        2.0,
        8.4,
        9.6,
        2.2,
    ]
    DBHRMSEHard = [
        1.8,
        17.4,
        2.0,
        30.0,
        7.4,
        25.0,
        9.4,
        3.0,
        3.6,
        1.2,
        17.4,
        2.4,
        9.4,
        12.4,
        2.2,
    ]

    BenchmarkMetrics = {}
    BenchmarkMetrics["Completeness"] = [
        CompletenessEasy,
        CompletenessMedium,
        CompletenessHard,
    ]
    BenchmarkMetrics["Correctness"] = [
        CorrectnessEasy,
        CorrectnessMedium,
        CorrectnessHard,
    ]
    BenchmarkMetrics["Location_RMSE"] = [StemRMSEEasy, StemRMSEMedium, StemRMSEHard]
    BenchmarkMetrics["Diameter_RMSE"] = [DBHRMSEEasy, DBHRMSEMedium, DBHRMSEHard]
    return BenchmarkMetrics, Methods


def load_gt(path="benchmark/annotations/TLS_Benchmarking_Plot_1_LHD.txt"):
    treedata = pd.read_csv(path, sep="\t", names=["x", "y", "height", "DBH"])
    Xcor, Ycor, diam = treedata.iloc[0, [0, 1, 3]]
    Zcor = 0
    TreeDict = [np.array([Xcor, Ycor, diam])]
    for i, rows in treedata.iloc[1:].iterrows():
        Xcor, Ycor, diam = rows.iloc[[0, 1, 3]]
        if not np.any(np.isnan([Xcor, Ycor, diam])):
            TreeDict.append(np.array([Xcor, Ycor, diam]))
    return TreeDict


def save_eval_results(path="results", EvaluationMetrics={}):
    np.savez(
        path,
        n_ref=EvaluationMetrics["n_ref"],
        n_match=EvaluationMetrics["n_match"],
        n_extr=EvaluationMetrics["n_extr"],
        location_y=EvaluationMetrics["location_y"],
        diameter_y=EvaluationMetrics["diameter_y"],
        Completeness=EvaluationMetrics["Completeness"],
        Correctness=EvaluationMetrics["Correctness"],
        Mean_AoD=EvaluationMetrics["Mean_AoD"],
        Diameter_RMSE=EvaluationMetrics["Diameter_RMSE"],
        Diameter_bias=EvaluationMetrics["Diameter_bias"],
        Location_RMSE=EvaluationMetrics["Location_RMSE"],
        Location_bias=EvaluationMetrics["Location_bias"],
        Relative_Diameter_RMSE=EvaluationMetrics["Relative_Diameter_RMSE"],
        Relative_Diameter_bias=EvaluationMetrics["Relative_Diameter_bias"],
        Relative_Location_RMSE=EvaluationMetrics["Relative_Location_RMSE"],
        Relative_Location_bias=EvaluationMetrics["Relative_Location_bias"],
        Diameter_RMSE_E=EvaluationMetrics["Diameter_RMSE_E"],
        Diameter_RMSE_C=EvaluationMetrics["Diameter_RMSE_C"],
    )


def load_eval_results(path="results.npz"):
    fileFGI = np.load(path)

    alldata = [
        "n_ref",
        "n_match",
        "n_extr",
        "Completeness",
        "Correctness",
        "Diameter_RMSE",
        "Diameter_bias",
        "Location_RMSE",
        "Location_bias",
        "Relative_Diameter_RMSE",
        "Relative_Diameter_bias",
        "Relative_Location_RMSE",
        "Relative_Location_bias",
        "Diameter_RMSE_C",
        "Diameter_RMSE_E",
    ]

    EvaluationMetrics = {}
    for i in alldata:
        EvaluationMetrics[i] = fileFGI[i]

    return EvaluationMetrics


def make_graph1(EvaluationMetrics, Methods, BenchmarkMetrics):
    alldata = ["Completeness", "Correctness"]
    dificulties = ["easy", "medium", "hard"]
    plt.figure(figsize=(16, 46))
    for n, i in enumerate(alldata):
        for n2 in range(3):
            plt.subplot(12, 1, n * 3 + n2 + 1)
            plt.title(i + " " + dificulties[n2])
            mine = np.mean(EvaluationMetrics[i][slice(n2, n2 + 2)]) * 100
            colors = [
                "black",
                "red",
                "green",
                "blue",
                "cyan",
                "magenta",
                "yellow",
                "black",
                "red",
                "green",
                "blue",
                "cyan",
                "magenta",
                "yellow",
            ]
            sortstuff = sorted(
                zip(
                    Methods[0:3] + ["our_imp"] + ["our_imp_2"] + Methods[3:],
                    BenchmarkMetrics[i][n2][0:3] + [mine] + BenchmarkMetrics[i][n2][3:],
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
    plt.savefig("metricsnew.pdf", pad_inches=0.1, bbox_inches="tight")


def make_graph2(EvaluationMetrics, Methods, BenchmarkMetrics):
    alldata = ["Location_RMSE", "Diameter_RMSE"]
    dificulties = ["easy", "medium", "hard"]
    plt.figure(figsize=(16, 46))
    for n, i in enumerate(alldata):
        for n2 in range(3):
            plt.subplot(12, 1, n * 3 + n2 + 1)
            plt.title(i + " " + dificulties[n2])
            mine = np.mean(EvaluationMetrics[i][slice(n2, n2 + 2)]) * 100
            colors = [
                "black",
                "red",
                "green",
                "blue",
                "cyan",
                "magenta",
                "yellow",
                "black",
                "red",
                "green",
                "blue",
                "cyan",
                "magenta",
                "yellow",
            ]
            sortstuff = sorted(
                zip(
                    Methods[0:3] + ["our_imp"] + ["our_imp_2"] + Methods[3:],
                    BenchmarkMetrics[i][n2][0:3] + [mine] + BenchmarkMetrics[i][n2][3:],
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
    plt.savefig("metricsnew2.pdf", pad_inches=0.1, bbox_inches="tight")


def store_metrics(EvaluationMetrics, treetool, TreeDict, dataindex, foundindex):
    # Get metrics
    locationerror = []
    correctlocationerror = []
    diametererror = []
    diametererrorElipse = []
    diametererrorComb = []
    for i, j in zip(dataindex, foundindex):
        locationerror.append(
            np.linalg.norm((treetool.finalstems[j]["model"][0:2] - TreeDict[i][0:2]))
        )
        if locationerror[-1] < 0.3:
            diametererror.append(
                #abs(treetool.finalstems[j]["model"][6] * 2 - TreeDict[i][2])
                abs(treetool.finalstems[j]["final_diameter"] - TreeDict[i][2])
            )
            correctlocationerror.append(
                np.linalg.norm(
                    (treetool.finalstems[j]["model"][0:2] - TreeDict[i][0:2])
                )
            )
            diametererrorElipse.append(
                abs(treetool.finalstems[j]["ellipse_diameter"] - TreeDict[i][2])
            )
            mindi = max(
                treetool.finalstems[j]["cylinder_diameter"],
                treetool.finalstems[j]["ellipse_diameter"],
            )
            diametererrorComb.append(abs(mindi - TreeDict[i][2]))

    EvaluationMetrics["n_ref"].append(len(TreeDict))
    EvaluationMetrics["n_match"].append(len(diametererror))
    EvaluationMetrics["n_extr"].append(
        len(locationerror) - EvaluationMetrics["n_match"][-1]
    )
    EvaluationMetrics["location_y"].append(
        np.linalg.norm(
            np.sum(np.array([TreeDict[i][0:2] for i in dataindex]), axis=0)
            / len(dataindex)
        )
    )
    EvaluationMetrics["diameter_y"].append(
        np.sum(
            np.array([treetool.finalstems[i]["model"][6] * 2 for i in foundindex]),
            axis=0,
        )
        / len(foundindex)
    )

    EvaluationMetrics["Completeness"].append(
        EvaluationMetrics["n_match"][-1] / EvaluationMetrics["n_ref"][-1]
    )
    EvaluationMetrics["Correctness"].append(
        EvaluationMetrics["n_match"][-1] / max((EvaluationMetrics["n_extr"][-1] + EvaluationMetrics["n_match"][-1]),1)
    )
    EvaluationMetrics["Mean_AoD"].append(
        2
        * EvaluationMetrics["n_match"][-1]
        / (EvaluationMetrics["n_ref"][-1] + EvaluationMetrics["n_extr"][-1])
    )
    EvaluationMetrics["Diameter_RMSE"].append(
        np.sqrt(np.sum(np.array(diametererror) ** 2) / len(diametererror))
    )
    EvaluationMetrics["Diameter_bias"].append(
        np.sum(np.array(diametererror)) / len(diametererror)
    )
    EvaluationMetrics["Location_RMSE"].append(
        np.sqrt(np.sum(np.array(correctlocationerror) ** 2) / len(correctlocationerror))
    )
    EvaluationMetrics["Location_bias"].append(
        np.sum(np.array(correctlocationerror)) / len(correctlocationerror)
    )

    EvaluationMetrics["Relative_Diameter_RMSE"].append(
        EvaluationMetrics["Diameter_RMSE"][-1] / EvaluationMetrics["diameter_y"][-1]
    )
    EvaluationMetrics["Relative_Diameter_bias"].append(
        EvaluationMetrics["Diameter_bias"][-1] / EvaluationMetrics["diameter_y"][-1]
    )
    EvaluationMetrics["Relative_Location_RMSE"].append(
        EvaluationMetrics["Location_RMSE"][-1] / EvaluationMetrics["location_y"][-1]
    )
    EvaluationMetrics["Relative_Location_bias"].append(
        EvaluationMetrics["Location_bias"][-1] / EvaluationMetrics["location_y"][-1]
    )

    EvaluationMetrics["Diameter_RMSE_E"].append(
        np.sqrt(np.sum(np.array(diametererrorElipse) ** 2) / len(diametererrorElipse))
    )
    EvaluationMetrics["Diameter_RMSE_C"].append(
        np.sqrt(np.sum(np.array(diametererrorComb) ** 2) / len(diametererrorComb))
    )
    return EvaluationMetrics


def confusion_metrics(treetool, TreeDict, dataindex, foundindex):
    # Get metrics
    FNS = []
    FPS = []
    TPS = []
    for i, j in zip(dataindex, foundindex):
        dist = np.linalg.norm((treetool.finalstems[j]["model"][0:2] - TreeDict[i][0:2]))
        Xcor, Ycor, diam = TreeDict[i]
        gt_model = [Xcor, Ycor, 0, 0, 0, 1, diam / 2]
        gt_tree = utils.makecylinder(model=gt_model, height=7, density=60)
        if dist < 0.4:
            stat_dict = {'true_tree': gt_tree, 'found_tree': treetool.finalstems[j]['tree'], 'true_model': gt_model,
                         'found_model': treetool.finalstems[j]['model']}
            TPS.append(stat_dict)
        else:
            stat_dict = {'true_tree': gt_tree, 'found_tree': None, 'true_model': gt_model,
                         'found_model': None}
            FNS.append(stat_dict)
            stat_dict = {'true_tree': None, 'found_tree': treetool.finalstems[j]['tree'], 'true_model': None,
                         'found_model': treetool.finalstems[j]['model']}
            FPS.append(stat_dict)

    if len(TreeDict) > len(dataindex):
        for i in np.arange(len(TreeDict))[~np.isin(np.arange(len(TreeDict)), dataindex)]:
            Xcor, Ycor, diam = TreeDict[i]
            gt_model = [Xcor, Ycor, 0, 0, 0, 1, diam / 2]
            gt_tree = utils.makecylinder(model=gt_model, height=7, density=60)
            stat_dict = {'true_tree': gt_tree, 'found_tree': None, 'true_model': gt_model,
                         'found_model': None}
            FNS.append(stat_dict)

    if len(treetool.finalstems) > len(foundindex):
        for i in np.arange(len(treetool.finalstems))[~np.isin(np.arange(len(treetool.finalstems)), foundindex)]:
            stat_dict = {'true_tree': None, 'found_tree': treetool.finalstems[i]['tree'], 'true_model': None,
                         'found_model': treetool.finalstems[i]['model']}
            FPS.append(stat_dict)

    return [TPS, FPS, FNS]


def store_metrics_detection_only(found_trees, gt_trees, gt_index, found_index):
    # Get metrics
    EvaluationMetrics = defaultdict(list)
    locationerror = []
    correctlocationerror = []
    for i, j in zip(gt_index, found_index):
        locationerror.append(
            np.linalg.norm((found_trees[j][0:2] - gt_trees[i][0:2]))
        )
        if locationerror[-1] < 0.5:
            correctlocationerror.append(
                np.linalg.norm(
                    (found_trees[j][0:2] - gt_trees[i][0:2])
                )
            )

    
    if len(gt_index) == 0:
        EvaluationMetrics["location_y"] = 100
    else:
        EvaluationMetrics["location_y"] = np.linalg.norm(
            np.sum(np.array([gt_trees[i][0:2] for i in gt_index]), axis=0)
            / len(gt_index)
        )

    EvaluationMetrics["n_ref"] = len(gt_trees)
    EvaluationMetrics["n_match"] = len(correctlocationerror)
    EvaluationMetrics["n_extr"] = len(locationerror) - EvaluationMetrics["n_match"]

    EvaluationMetrics["Completeness"] = EvaluationMetrics["n_match"] / EvaluationMetrics["n_ref"]
    
    EvaluationMetrics["Correctness"] = EvaluationMetrics["n_match"] / max((EvaluationMetrics["n_extr"] + EvaluationMetrics["n_match"]),1)
    
    if len(correctlocationerror) != 0:
        EvaluationMetrics["Location_RMSE"] = np.sqrt(np.sum(np.array(correctlocationerror) ** 2) / len(correctlocationerror))
        EvaluationMetrics["Location_bias"] = np.sum(np.array(correctlocationerror)) / len(correctlocationerror)
        if len(gt_index) != 0:
            EvaluationMetrics["Relative_Location_RMSE"] = EvaluationMetrics["Location_RMSE"] / EvaluationMetrics["location_y"]
            EvaluationMetrics["Relative_Location_bias"] = EvaluationMetrics["Location_bias"] / EvaluationMetrics["location_y"]
        else:
            EvaluationMetrics["Relative_Location_RMSE"] = 100
            EvaluationMetrics["Relative_Location_bias"] = 100
    else:
        EvaluationMetrics["Location_RMSE"] = 100
        EvaluationMetrics["Location_bias"] = 100
        EvaluationMetrics["Relative_Location_RMSE"] = 100
        EvaluationMetrics["Relative_Location_bias"] = 100    
    
    return EvaluationMetrics

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

def run_detection(args, cfg, model, results_dict, vis_dict, generator, use_non_filtered=False, tolerence=0.02):
    Nd = cfg.BENCHMARKING.XY_THRESHOLD
    ARe = np.deg2rad(cfg.BENCHMARKING.ANGLE_THRESHOLD)
    voxel_size = cfg.BENCHMARKING.VOXEL_SIZE
    for n_x, x_start, n_y, y_start, sub_pcd, sub_pcd_nf  in generator:
        if np.shape(sub_pcd.xyz)[0] > 0:
            Odata_xyz = data_preprocess(sub_pcd_nf.xyz)
            Fdata_xyz = data_preprocess(sub_pcd.xyz)
            if use_non_filtered:
                data_xyz = Odata_xyz
            else:
                data_xyz = Fdata_xyz
            data_xyz = shuffle_data(data_xyz)
            _data_xyz = data_xyz[:4096]
            center, scale = get_center_scale(sub_pcd.xyz)

            if np.shape(_data_xyz)[0] < 4096:
                additional_elements = 4096 - len(_data_xyz)
                indexes_to_repeat = np.random.choice(len(_data_xyz), additional_elements)
                repeated_values = _data_xyz[indexes_to_repeat]
                _data_xyz = np.concatenate((_data_xyz, repeated_values))            
            nor_testdata = torch.tensor(_data_xyz, device="cuda").squeeze()
            xyz_direction = prediction(model, nor_testdata, args)
                
            if cfg.DATA_PREPROCESSING.DISTANCE_FILTER == 0.0:
                _xyz_direction = xyz_direction
            else:
                f_distances = xyz_direction[:, 6]
                if cfg.MODEL.CLASS_SIGMOID:
                    _xyz_direction = xyz_direction[ f_distances >  cfg.DATA_PREPROCESSING.DISTANCE_FILTER ]
                else:
                    _xyz_direction = xyz_direction[ (f_distances - np.min(f_distances))
                        / (np.max(f_distances) - np.min(f_distances))
                        < cfg.DATA_PREPROCESSING.DISTANCE_FILTER ]

            if cfg.BENCHMARKING.CENTER_DETECTION_ENABLE:
                object_center_list, seppoints = center_detection(
                        _xyz_direction, voxel_size, ARe, Nd
                    )
            else:
                points = _xyz_direction[:,:3] * scale + center
                seppoints = seg_tree.euclidean_cluster_extract(points, tolerance=tolerence, min_cluster_size=10, max_cluster_size=100000)
                object_center_list = None
            if len(seppoints) > 0:
                seppoints = [i for i in seppoints if np.size(i, 0)]
                results_dict[n_x][n_y] = {
                        "x": x_start,
                        "y": y_start,
                        "Opoints": Odata_xyz,
                        "Fpoints": data_xyz,
                        "Ipoints": _data_xyz,
                        "centers": object_center_list,
                        "segmentation": seppoints,
                        'prediction':_xyz_direction,
                        "center": center,
                        "scale": scale,
                    }
                result_points = seppoints
                vis_dict.extend(result_points)

def sample_generator(sample_side_size, overlap, treetool):
    x1, y1, z1 = np.min(treetool.filtered_points.xyz, 0)
    x2, y2, z2 = np.max(treetool.filtered_points.xyz, 0)
    cropfilter = pclpy.pcl.filters.CropBox.PointXYZ()
        
    for n_x, x_start in enumerate(
                tqdm(np.arange(x1, x2, sample_side_size - sample_side_size * overlap), desc='plot row')):
        for n_y, y_start in enumerate(np.arange(y1, y2, sample_side_size - sample_side_size * overlap)):
            cropfilter.setMin(np.array([x_start, y_start, -1000, 1.0]))
            cropfilter.setMax(
                    np.array([x_start + sample_side_size, y_start + sample_side_size, 1000, 1.0])
                )
            cropfilter.setInputCloud(treetool.filtered_points)
            sub_pcd = pclpy.pcl.PointCloud.PointXYZ()
            cropfilter.filter(sub_pcd)
            cropfilter.setInputCloud(treetool.non_filtered_points)
            sub_pcd_nf = pclpy.pcl.PointCloud.PointXYZ()
            cropfilter.filter(sub_pcd_nf)
            yield n_x,x_start,n_y,y_start,sub_pcd,sub_pcd_nf

def run_combine_stems(cfg, treetool, eps = 1):
    if cfg.BENCHMARKING.COMBINE_STEMS:
        from sklearn.cluster import dbscan
        models = [i['model'] for i in treetool.finalstems]
        vis = [i['tree'] for i in treetool.finalstems]
        if len(models) > 0:
            dp = dbscan(np.array(models), eps=eps, min_samples=2)[1]

            _models = np.array(models)[dp == -1].tolist()
            _vis = np.array(vis, dtype=object)[dp == -1].tolist()
            for clust in np.unique(dp):
                if clust == -1:
                    continue
                _models.append(
                        np.array(models)[dp == clust].tolist()[np.argmax([len(i) for i in np.array(vis, dtype=object)[dp == clust]])])
                _vis.append(np.vstack(np.array(vis, dtype=object)[dp == clust]).tolist())
            treetool.finalstems = [{'tree': np.array(v), 'model': np.array(m)} for m, v in zip(_models, _vis)]
        else:
            treetool.finalstems
        treetool.visualization_cylinders = [utils.makecylinder(model=stem['model'], height=7, density=60) for stem in treetool.finalstems]


def matching(treetool, TreeDict):
    CostMat = np.ones([len(TreeDict), len(treetool.finalstems)])
    for X, datatree in enumerate(TreeDict):
        for Y, foundtree in enumerate(treetool.finalstems):
            CostMat[X, Y] = np.linalg.norm([datatree[0:2] - foundtree["model"][0:2]])

    dataindex, foundindex = linear_sum_assignment(CostMat, maximize=False)
    return dataindex,foundindex


def cloud_match(gt_trees, pointcloud, min_dist=0.2, min_points=100):
    cloud = pclpy.pcl.PointCloud.PointXYZ(pointcloud)
    good = {}
    bad = {}
    for n, datatree in enumerate(gt_trees):
        min_crop = datatree[0:2] - [0.5, 0.5]
        max_crop = datatree[0:2] + [0.5, 0.5]
        cropped_points = seg_tree.box_crop(cloud, [*min_crop, -2, 1], [*max_crop, 2, 1])
        distance = np.linalg.norm(datatree[0:2] - cropped_points[:, :2], axis=-1)
        n_dist = sum(distance < min_dist)
        if n_dist > min_points:
            good[n] = cropped_points
        else:
            if len(cropped_points) > 0:
                bad[n] = np.vstack(
                    [
                        cropped_points,
                        make_cylinder(
                            [*datatree[0:2], 0, 0, 0, 1, datatree[2]/4],
                            length=2,
                            dense=20,
                        ),
                    ]
                )
            else:
                bad[n] = make_cylinder(
                    [*datatree[0:2], 0, 0, 0, 1, datatree[2]/4], length=2, dense=20
                )
    return good, bad


def cloud_match_clusters(gt_trees, pointcloud_cluster, min_dist=0.2, mean=False, tp_criteria=20):
    tp = {}
    fp = {}
    fn = {}
    gt_index, foundindex, CostMat = cluster_match(gt_trees, pointcloud_cluster, min_dist, mean, return_mat=True)
    if mean:
        raise NotImplemented
    else:
        true_mat = CostMat[gt_index, foundindex] > tp_criteria
        tp = np.hstack([gt_index[true_mat][:,np.newaxis],foundindex[true_mat][:,np.newaxis]], dtype=np.int32)
        fn = np.hstack([gt_index[~true_mat][:,np.newaxis],foundindex[~true_mat][:,np.newaxis]], dtype=np.int32)
        not_found = set(np.arange(len(gt_trees))) - set(gt_index)
        if len(not_found) > 0:
            more_fn = np.hstack([np.array(list(not_found))[:,np.newaxis],-np.ones_like(list(not_found))[:,np.newaxis]], dtype=np.int32)
            fn = np.vstack([fn,more_fn], dtype=np.int32)
        fp = list(set(np.arange(len(pointcloud_cluster))) - set(foundindex)) + foundindex[~true_mat].tolist()
    return tp,fp,fn
    

def cluster_match(gt_trees, pointcloud_cluster, min_dist=0.2, mean=False, return_mat=False):
    if mean:
        centers = [np.mean(i, axis=0) for i in pointcloud_cluster]
        CostMat = make_cost_mat(gt_trees, centers)
        gt_index, foundindex = linear_sum_assignment(CostMat, maximize=False)
    else:
        CostMat = make_cost_mat(
        gt_trees,
        pointcloud_cluster,
        lambda x, y: sum(np.linalg.norm(x[0:2] - y[:, :2], axis=-1) < min_dist),
    )
        gt_index, foundindex = linear_sum_assignment(CostMat, maximize=True)
    if return_mat:
        return gt_index, foundindex, CostMat
    else:   
        return gt_index, foundindex

def make_cost_mat(
    gt_trees, centers, fun=lambda x, y: np.linalg.norm([x[0:2] - y[0:2]])
):
    CostMat = np.ones([len(gt_trees), len(centers)])
    for X, datatree in enumerate(gt_trees):
        for Y, foundtree in enumerate(centers):
            CostMat[X, Y] = fun(datatree, foundtree)
    return CostMat

def get_com_cor(tp, fp, fn):
    comp = len(tp) / (len(tp) + len(fn))
    corr = len(tp) / (len(tp) + len(fp))
    return comp, corr

def print_metrics(message,tp, fp, fn):
    comp, corr = get_com_cor(tp, fp, fn)
    print(message)
    print('    completeness:', round(comp,4), '%')
    print('    correctness:', round(corr,4), '%')

def metrics_2_clouds(gt_trees, pointcloud_cluster, tp, fp, fn):
    tp_clouds = []
    fp_clouds =[]
    fn_clouds = []
    for i in tp:
        cloud = np.vstack([ pointcloud_cluster[i[1]], 
        make_cylinder(
                    [*gt_trees[i[0]][0:2], 0, 0, 0, 1, gt_trees[i[0]][2]/4], length=2, dense=25
                ) 
                    ]
        )
        tp_clouds.append(cloud)
    
    for i in fp:
        cloud = np.vstack([pointcloud_cluster[i],make_cylinder(
                    [*np.mean(pointcloud_cluster[i], axis=0)[:2], 0, 0, 0, 1, 0.01], length=8, dense=15
                ) ])
        fp_clouds.append(cloud)

    for i in fn:
        dists = np.linalg.norm(pointcloud_cluster[i[1]][:,:2] - gt_trees[i[0]][:2], axis=-1)
        cloud = np.vstack([pointcloud_cluster[i[1]][dists<0.2],make_cylinder(
                    [*gt_trees[i[0]][:2], 0, 0, 0, 1, gt_trees[i[0]][2]/3], length=6, dense=40
                ) ])
        fn_clouds.append(cloud)

    tp_clouds = np.vstack(tp_clouds)
    fp_clouds = np.vstack(fp_clouds)
    fn_clouds = np.vstack(fn_clouds)
    return tp_clouds,fp_clouds,fn_clouds

def get_close_point_ratio(x,y,min_dist):
    point_distances = distance.cdist(x, y)
    good_points = sum(np.min(point_distances, axis=np.argmax(point_distances.shape))<min_dist)
    ratio = good_points / np.min(point_distances.shape)
    return ratio

def get_close_points(keys,query,min_dist):
    point_distances = distance.cdist(keys, query)
    good_index = np.min(point_distances, axis=0)<min_dist
    return good_index


def close_points_ratio_merge(stems, min_dist =0.1):
    mat = make_cost_mat(stems, stems, lambda x,y:get_close_point_ratio(x[:,:2], y[:,:2], min_dist))
    for x in range(len(mat)):
        mat[x,x] = 0

    new_stems = []
    used = []
    thresh = 0.2
    gt_index, foundindex = linear_sum_assignment(mat, maximize=True)
    for i in zip(gt_index, foundindex):
        if (i[0] in used):
            continue
        if mat[i[0],i[1]] > thresh:
            if (i[1] not in used):
                new_stems.append(np.vstack([stems[i[0]],stems[i[1]]]))
                used.extend(i)
            else:
                new_stems.append(stems[i[0]])
                used.append(i[0])
        else:
            new_stems.append(stems[i[0]])
            used.append(i[0])
    return new_stems


def combine_close_points(vis_dict_, distance=0.05, ratio=0.6):
    new_clust = vis_dict_.copy()
    centers = [np.mean(i,0)[:2] for i in new_clust]
    for treenumber1 in reversed(range(0, len(new_clust))):
        for treenumber2 in reversed(range(0, treenumber1)):
            center1 = centers[treenumber1]
            center2 = centers[treenumber2]
            if np.linalg.norm(center1-center2) < 1:
                points1 = vis_dict_[treenumber1]
                points2 = vis_dict_[treenumber2]
                if get_close_point_ratio(points1,points2, distance) > ratio:
                    new_clust[treenumber2] = np.vstack(
                        [new_clust[treenumber2], new_clust.pop(treenumber1)]
                    )
                    break
    return new_clust