import numpy as np
from collections import defaultdict
from porteratzo3D.geometry_utils import (
    angle_between_two_vectors,
    dist_point_to_line
)

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

def geometrics(batch_object_label, n_center, xyz_direction, true_centers, batch_direction_label_data):
    output_dict = defaultdict(list)
    for n_c_, center in enumerate(true_centers):
        idx_s = batch_object_label[n_center] == n_c_
        xyz_ = xyz_direction[:, 0:3][idx_s]
        directions_ = xyz_direction[:, 3:6][idx_s]
        distance_ = np.linalg.norm(center - xyz_, axis=-1)
        val1 = 0.01
        val2 = 0.02
        val3 = 0.05

        angles__xy = angle_between_two_vectors(
            (center - xyz_) * [1, 1, 0.00001],
            directions_ * [1, 1, 0.00001],
        )
        dists__xy = dist_point_to_line(
            center * [1, 1, 0.0001],
            xyz_ * [1, 1, 0.00001],
            directions_ * [1, 1, 0.00001],
        )
        output_dict["average_pred_distance_xy"].append(np.mean(dists__xy))
        output_dict["average_pred_angle_xy"].append(np.mean(angles__xy))
        output_dict["average_correctly_assigned_1_xy"].append(
            np.sum(dists__xy < val1) / len(dists__xy)
        )
        output_dict["average_correctly_assigned_2_xy"].append(
            np.sum(dists__xy < val2) / len(dists__xy)
        )
        output_dict["average_correctly_assigned_3_xy"].append(
            np.sum(dists__xy < val3) / len(dists__xy)
        )

        d_dists__xy = dists__xy[distance_ < 0.3]
        output_dict["average_correctly_assigned_1_d_xy"].append(
            np.sum(d_dists__xy < val1) / len(d_dists__xy)
        )
        output_dict["average_correctly_assigned_2_d_xy"].append(
            np.sum(d_dists__xy < val2) / len(d_dists__xy)
        )
        output_dict["average_correctly_assigned_3_d_xy"].append(
            np.sum(d_dists__xy < val3) / len(d_dists__xy)
        )

        angles_ = angle_between_two_vectors(
            (center - xyz_) ,
            directions_ ,
        )
        dists_ = dist_point_to_line(
            center,
            xyz_ ,
            directions_ ,
        )
        output_dict["average_pred_distance"].append(np.mean(dists_))
        output_dict["average_pred_angle"].append(np.mean(angles_))
        output_dict["average_correctly_assigned_1"].append(
            np.sum(dists_ < val1) / len(dists_)
        )
        output_dict["average_correctly_assigned_2"].append(
            np.sum(dists_ < val2) / len(dists_)
        )
        output_dict["average_correctly_assigned_3"].append(
            np.sum(dists_ < val3) / len(dists_)
        )

        d_dists_ = dists_[distance_ < 0.3]
        output_dict["average_correctly_assigned_1_d"].append(
            np.sum(d_dists_ < val1) / len(d_dists_)
        )
        output_dict["average_correctly_assigned_2_d"].append(
            np.sum(d_dists_ < val2) / len(d_dists_)
        )
        output_dict["average_correctly_assigned_3_d"].append(
            np.sum(d_dists_ < val3) / len(d_dists_)
        )

    return output_dict