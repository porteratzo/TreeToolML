from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict
from dataclasses import dataclass

from inter_det.data.feature import Event

TargetKey = Tuple[str, str]


def evaluation_metrics(cm):
    TP = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return TP, FP, FN, precision, recall


def evaluation_metrics_evaluator(evaluator):
    TP = evaluator.tp
    FP = evaluator.fp
    FN = evaluator.fn
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return precision, recall


def event_timestamp_overlap_calculation(truth_event: Event, eva_event: Event):
    # calculate intersection between groundtruth and events if gt contains all eva then return 1 if it contains nothing then 0 else
    truth_start_time = truth_event.t_start
    truth_end_time = truth_event.t_end
    eva_start_time = eva_event.t_start
    eva_end_time = eva_event.t_end

    overlap_rate = 0
    if (truth_start_time >= eva_end_time) | (eva_start_time >= truth_end_time):
        overlap_rate = 0
    else:
        if ((truth_start_time <= eva_start_time) & (truth_end_time >= eva_end_time)) | (
            (eva_start_time <= truth_start_time) & (eva_end_time >= truth_end_time)
        ):
            overlap_rate = 1
        else:
            if truth_start_time < eva_start_time:
                overlap_rate = (truth_end_time - eva_start_time) / (
                    truth_end_time - truth_start_time
                )
            else:
                overlap_rate = (eva_end_time - truth_start_time) / (
                    truth_end_time - truth_start_time
                )
    return overlap_rate


def calc_iou(ts1, ts2):
    t1 = max(ts1[0], ts2[0])
    t2 = min(ts1[1], ts2[1])
    if t2 <= t1:
        return 0
    union = max(ts1[1], ts2[1]) - min(ts1[0], ts2[0])
    inter = t2 - t1
    # return inter / (ts1[1] - ts1[0])
    return inter / union


def calc_coverage(ts1, ts2):
    t1 = max(ts1[0], ts2[0])
    t2 = min(ts1[1], ts2[1])
    if t2 <= t1:
        return 0
    inter = t2 - t1
    return inter / (ts2[1] - ts2[0])
    # return inter / union


def evaluate_coverage(
    detected_events: List[Event], annotation_events: List[Event], threshold: float = 0.8
):
    C = np.zeros([len(detected_events), len(annotation_events)], dtype=float)
    if C.size == 0:
        return np.zeros(0, dtype=bool)
    for i, det in enumerate(detected_events):
        for j, anno in enumerate(annotation_events):
            C[i, j] = calc_coverage(
                (det.t_start, det.t_end), (anno.t_start, anno.t_end)
            )
    max_coverages = np.max(C, axis=0)
    return max_coverages


def evaluate_iou(
    detected_events: List[Event], annotation_events: List[Event], threshold: float = 0.6
):
    C = np.zeros([len(detected_events), len(annotation_events)], dtype=float)
    if C.size == 0:
        return np.zeros(0, dtype=bool)
    for i, det in enumerate(detected_events):
        for j, anno in enumerate(annotation_events):
            C[i, j] = calc_iou((det.t_start, det.t_end), (anno.t_start, anno.t_end))
    max_coverages = np.max(C, axis=0)
    return max_coverages


def evaluate_detected_events(
    detected_events: List[Event], annotation_events: List[Event], thresh=0.8
) -> Tuple:

    TP = 0
    gt_match = np.zeros([1, len(annotation_events)])
    de_match = np.zeros([1, len(detected_events)])

    if len(detected_events) < 1:
        TP_num = TP
        FP_num = len(detected_events) - TP_num
        FN_num = len(annotation_events) - TP_num
        return (TP_num, FP_num, FN_num)

    for dete_index in range(len(detected_events)):
        for ground_truth_number in range(len(annotation_events)):
            ground_truth_event = annotation_events[ground_truth_number]
            detected_event = detected_events[dete_index]
            # overlap_rate = event_timestamp_overlap_calculation(
            #    ground_truth_event, detected_event
            # )
            overlap_rate = calc_iou(
                (ground_truth_event.t_start, ground_truth_event.t_end),
                (detected_event.t_start, detected_event.t_end),
            )
            if overlap_rate >= thresh:
                TP = TP + 1
                gt_match[0, ground_truth_number] = 1
                de_match[0, dete_index] = 1
                break
        if TP == len(annotation_events):
            break
    TP_num = TP
    FP_num = len(detected_events) - TP_num
    FN_num = len(annotation_events) - TP_num

    return (TP_num, FP_num, FN_num)


def evaluate_detected_events_return_events(
    detected_events: List[Event], annotation_events: List[Event], thresh=0.8
) -> Tuple:

    TP = 0
    gt_match = -1*np.ones([len(annotation_events)])
    de_match = -1*np.ones([len(detected_events)])

    matched_ano = []
    matched_det = []
    ground_truth_IOU_best = -1*np.zeros([len(annotation_events)])
    dete_IOU_best = -1*np.zeros([len(detected_events)])

    if len(detected_events) < 1:
        TP_num = TP
        FP_num = len(detected_events) - TP_num
        FN_num = len(annotation_events) - TP_num
        return (
            (TP_num, FP_num, FN_num),
            (
                [matched_det, matched_ano],
                [],
                [np.arange(len(detected_events))],
                [np.arange(len(annotation_events))],
            ),
            ground_truth_IOU_best,
            dete_IOU_best,
            gt_match,
            de_match
        )

    for dete_index in range(len(detected_events)):
        for ground_truth_number in range(len(annotation_events)):
            ground_truth_event = annotation_events[ground_truth_number]
            detected_event = detected_events[dete_index]
            overlap_rate = calc_iou(
                (ground_truth_event.t_start, ground_truth_event.t_end),
                (detected_event.t_start, detected_event.t_end),
            )

            if overlap_rate > ground_truth_IOU_best[ground_truth_number]:
                ground_truth_IOU_best[ground_truth_number] = overlap_rate
                gt_match[ground_truth_number] = dete_index

            if overlap_rate > dete_IOU_best[dete_index]:
                dete_IOU_best[dete_index] = overlap_rate
                de_match[dete_index] = ground_truth_number

            if overlap_rate >= thresh:
                TP = TP + 1
                matched_ano.append(ground_truth_number)
                matched_det.append(dete_index)
                break

        if TP == len(annotation_events):
            break
    TP_num = TP
    FP_num = len(detected_events) - TP_num
    FN_num = len(annotation_events) - TP_num

    return (
        (TP_num, FP_num, FN_num),
        (
            [matched_det, matched_ano],
            [
                n
                for n in range(len(detected_events))
                if (n not in matched_det) and (dete_IOU_best[n] > 0)
            ],
            [
                n
                for n in range(len(detected_events))
                if (n not in matched_det) and (dete_IOU_best[n] == 0)
            ],
            [n for n in range(len(annotation_events)) if n not in matched_ano],
        ),
        (ground_truth_IOU_best),
        (dete_IOU_best),
        gt_match,
        de_match
    )


def evaluate_overlap_rate(
    detected_events: List[Event], annotation_events: List[Event]
) -> List:
    C = np.zeros([len(detected_events), len(annotation_events)], dtype=float)
    if C.size == 0:
        return np.zeros(0, dtype=bool)
    if len(detected_events) < 1:
        return []
    overlap_r = []
    for i, det in enumerate(detected_events):
        for j, anno in enumerate(annotation_events):
            C[i, j] = event_timestamp_overlap_calculation(anno, det)
    overlap_r = np.max(C, axis=0)
    return overlap_r


@dataclass
class Evaluator:
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
