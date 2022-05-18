import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import TreeTool.utils as utils

def make_benchmark_metrics():
    Methods = [
        "TreeTool",
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
    ]
    CompletenessEasy = [81, 88, 66, 94, 84, 74, 88, 89, 94, 87, 82, 96, 36, 69, 89]
    CompletenessMedium = [64, 75, 49, 88, 65, 59, 81, 78, 88, 74, 68, 87, 27, 58, 80]
    CompletenessHard = [54, 44, 16, 66, 27, 25, 45, 46, 64, 39, 39, 63, 18, 37, 52]

    CorrectnessEasy = [84, 96, 69, 90, 97, 94, 44, 77, 86, 94, 95, 77, 99, 86, 86]
    CorrectnessMedium = [84, 99, 61, 89, 98, 97, 44, 76, 91, 95, 96, 89, 99, 91, 88]
    CorrectnessHard = [90, 99, 41, 93, 97, 97, 70, 74, 91, 95, 97, 88, 99, 91, 90]

    StemRMSEEasy = [
        1.5,
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
    ]
    StemRMSEMedium = [
        2.5,
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
    ]
    StemRMSEHard = [
        2.6,
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
    ]

    DBHRMSEEasy = [
        2.6,
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
    ]
    DBHRMSEMedium = [
        2.2,
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
    ]
    DBHRMSEHard = [
        2.2,
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
        if locationerror[-1] < 0.4:
            diametererror.append(
                abs(treetool.finalstems[j]["model"][6] * 2 - TreeDict[i][2])
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
        EvaluationMetrics["n_match"][-1]
        / (EvaluationMetrics["n_extr"][-1] + EvaluationMetrics["n_match"][-1])
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
        if dist < 2:
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



    return [TPS,FPS,FNS]
