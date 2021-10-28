from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from inter_det.data.feature import Event, FeaturePoint, TrainSample
import pickle
import glob
from record_dataloader import RecordItem, TimeSeriesLoader
import os
import json

TargetKey = Tuple[str, str]


def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotationMatrixToEulerAnglesZYX(R):
    assert isRotationMatrix(R)

    if R[2, 0] < 1:
        if R[2, 0] > -1:
            y = np.arcsin(-R[2, 0])
            z = np.arctan2(R[1, 0], R[0, 0])
            x = np.arctan2(R[2, 1], R[1, 1])
        else:
            y = np.pi / 2
            z = -np.arctan2(-R[1, 2], R[1, 1])
            x = 0
    else:
        y = -np.pi / 2
        z = np.arctan2(-R[1, 2], R[1, 1])
        x = 0

    return np.array([x, y, z])


def read_annotations(filename: str) -> Dict[TargetKey, List[Event]]:
    # load anotation csv
    # return dict with key (rec, id) and data [event(start, end, gondola=-1, action), ...] and dict with key dataset and values [ids, ...]
    df = pd.read_csv(filename)
    annotations: Dict[TargetKey, List[Event]] = defaultdict(list)
    for _, row in df.iterrows():
        k = (str(row.rec_id), str(row.target_id))
        event = Event(row.t_start, row.t_end, gondola_id=-1, action=row.action)
        annotations[k].append(event)
    annotations = {
        key: sorted(annotations[key], key=lambda x: x.t_start)
        for key in annotations.keys()
    }
    return annotations

def read_annotations_annotation_type(filename: str, annotation_type=None) -> Dict[TargetKey, List[Event]]:
    # load anotation csv
    # return dict with key (rec, id) and data [event(start, end, gondola=-1, action), ...] and dict with key dataset and values [ids, ...]
    df = pd.read_csv(filename)
    df = df[~df['action'].isin(annotation_type)]
    annotations: Dict[TargetKey, List[Event]] = defaultdict(list)

    for _, row in df.iterrows():
        k = (str(row.rec_id), str(row.target_id))
        event = Event(row.t_start, row.t_end, gondola_id=-1, action=row.action)
        annotations[k].append(event)
    annotations = {
        key: sorted(annotations[key], key=lambda x: x.t_start)
        for key in annotations.keys()
    }
    return annotations

def read_annotations_dataset(
    filename: str, select_dataset_name: List
) -> Dict[TargetKey, List[Event]]:
    df = pd.read_csv(filename)
    annotations: Dict[TargetKey, List[Event]] = defaultdict(list)
    for _, row in df.iterrows():
        if row.rec_id not in select_dataset_name:
            continue
        k = (str(row.rec_id), str(row.target_id))
        event = Event(row.t_start, row.t_end, row.action)
        annotations[k].append(event)
    return annotations


def write_fp_events_to_csv(filename: str, event_dict: Dict):
    dataframe = pd.DataFrame(
        {
            "rec_id": event_dict["dataset"],
            "target_id": event_dict["id"],
            "t_start": event_dict["t_start"],
            "t_end": event_dict["t_end"],
            "action": event_dict["action"],
        }
    )
    dataframe.to_csv(filename, index=False, sep=",")


def read_gondolas(filename) -> Dict[int, pd.Series]:
    gondolas = pd.read_csv(filename)
    gondolas_map: Dict[int, pd.Series] = {}
    for _, g in gondolas.iterrows():
        gondolas_map[int(g["id"])] = g
    return gondolas_map


def read_shelves(shelve_file):
    shelf = pd.read_csv(
        shelve_file,
        header=0,
        names=[
            "id",
            "created_at",
            "updated_at",
            "index",
            "gondola_id",
            "dim_x",
            "dim_y",
            "dim_z",
            "rotation_x",
            "rotation_y",
            "rotation_z",
            "translation_x",
            "translation_y",
            "translation_z",
        ],
    )
    return shelf


def read_gondolas_json(filename) -> Dict[int, pd.Series]:
    with open(filename, "r") as f:
        data = json.load(f)

    gondolas_map: Dict[int, pd.Series] = {}
    for gondola in data:
        id = gondola["id_"]
        mat = np.array(gondola["box"]["pose"]["matrix"])
        dim_x = gondola["box"]["dim_x"]
        dim_y = gondola["box"]["dim_y"]
        dim_z = gondola["box"]["dim_z"]
        translation_x = mat[0, 3]
        translation_y = mat[1, 3]
        translation_z = mat[2, 3]
        rotation = mat[:3, :3]
        rotation_x, rotation_y, rotation_z = rotationMatrixToEulerAnglesZYX(rotation)
        rotation = mat[:3, :3]
        series_dict = {
            "id": id,
            "dim_x": dim_x,
            "dim_y": dim_y,
            "dim_z": dim_z,
            "translation_x": translation_x,
            "translation_y": translation_y,
            "translation_z": translation_z,
            "rotation_x": rotation_x,
            "rotation_y": rotation_y,
            "rotation_z": rotation_z,
        }
        gondolas_map[int(id)] = pd.Series(series_dict)

    return gondolas_map


def read_features(feat_path: str) -> Dict[TargetKey, List[FeaturePoint]]:
    all_features = {}
    if os.path.isdir(feat_path):
        files = glob.glob(os.path.join(feat_path, "*.*"))
    else:
        files = [feat_path]
    for filename in files:
        with open(filename, "rb") as f:
            data: Dict[TargetKey, List[Dict]] = pickle.load(f)
            all_features.update(data)
    all_features = {
        k: [FeaturePoint(**v) for v in vv] for k, vv in all_features.items()
    }
    return all_features


def read_features_datasets(
    feat_path: str, dataset_name: str
) -> Dict[TargetKey, List[FeaturePoint]]:
    all_features = {}
    files = glob.glob(os.path.join(feat_path, "*.pk"))
    for filename in files:
        name_sp = filename.split("/")
        name_sp = name_sp[-1]
        set_name = name_sp.split("_")[0].split(".")[0]
        if set_name not in dataset_name:
            continue
        with open(filename, "rb") as f:
            data: Dict[TargetKey, List[Dict]] = pickle.load(f)
            all_features.update(data)
    all_features = {
        k: [FeaturePoint(**v) for v in vv] for k, vv in all_features.items()
    }
    return all_features


class FeatureSeries(TimeSeriesLoader):
    def __init__(self, features: List[FeaturePoint]):
        super().__init__(features)

    def _load_data(self, features: List[FeaturePoint]) -> List[RecordItem]:
        return [
            RecordItem(data, None, data.timestamp, data.timestamp) for data in features
        ]

    def _read_data(self, item: RecordItem) -> FeaturePoint:
        return item.data


def fuse_shelves_to_gondolas(gondolas_map, shelf_map):
    tem_gondolas_map = gondolas_map.copy()
    for gondola_key in gondolas_map.keys():
        # if gondola_key != 10:
        #    continue
        gondola = gondolas_map[gondola_key]
        x2, y2, z2 = gondola.dim_x, gondola.dim_y, gondola.dim_z
        angle = gondola.rotation_z
        trans_x, trans_y = gondola.translation_x, gondola.translation_y

        cs = np.cos(angle)
        sn = np.sin(angle)
        transform = np.array(
            [[cs, -sn, 0, trans_x], [sn, cs, 0, trans_y], [0, 0, 1, 0], [0, 0, 0, 1],]
        )
        gondola_shelve = shelf_map[shelf_map.gondola_id == gondola_key]

        angle = np.mean(gondola_shelve.rotation_z)
        trans_xg, trans_yg, trans_zg = (
            np.min(gondola_shelve.translation_x),
            np.min(gondola_shelve.translation_y),
            gondola_shelve.translation_z,
        )
        x2g, y2g, z2g = (
            np.max(gondola_shelve["dim_x"]),
            np.max(gondola_shelve.dim_y),
            np.max(gondola_shelve.dim_z),
        )

        shelf_points = np.array([[0, 0, 0, 1], [x2g, y2g, 0, 1]])

        csg = np.cos(angle)
        sng = np.sin(angle)
        transformshelf = np.array(
            [
                [csg, -sng, 0, trans_xg],
                [sng, csg, 0, trans_yg],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        shelf_pointsT = transform @ transformshelf @ shelf_points.T
        shelf_pointsT = shelf_pointsT.T

        tem_gondolas_map[gondola_key].translation_x = shelf_pointsT[0, 0]
        tem_gondolas_map[gondola_key].translation_y = shelf_pointsT[0, 1]
        tem_gondolas_map[gondola_key].dim_x = x2g
        tem_gondolas_map[gondola_key].dim_y = y2g

    return tem_gondolas_map


def move_to_gondolas(gondolas_map, points, gid = None):
    add_gondola = []
    if gid is None:
        keys = list(gondolas_map.keys())
    else:
        keys = np.array(list(gondolas_map.keys()))[gid:gid+1]
    for gondola_key in keys:
        gondola = gondolas_map[gondola_key]
        x1, y1, z1, x2, y2, z2 = 0, 0, 0, gondola.dim_x, gondola.dim_y, gondola.dim_z
        angle = gondola.rotation_z
        trans_x, trans_y = gondola.translation_x, gondola.translation_y
        x2 = x2 if x2 != 0.0 else 0.1
        y2 = y2 if y2 != 0.0 else 0.1
        z2 = z2 if z2 != 0.0 else 0.1

        cs = np.cos(angle)
        sn = np.sin(angle)
        transform = np.array(
            [[cs, -sn, 0, trans_x], [sn, cs, 0, trans_y], [0, 0, 1, 0], [0, 0, 0, 1],]
        )
        # part = np.meshgrid(np.arange(-(x2-x1)/2, (x2-x1)/2,0.1), np.arange(z1,z2,0.1))
        part = points
        world = transform @ np.vstack(
            [
                part[:,0].flatten(),
                part[:,1].flatten(),
                part[:,2].flatten(),
                np.ones_like(part[:,0].flatten()),
            ]
        )
        world = world.T[:, :3] * 39.37
        add_gondola.append(world)
    return add_gondola


def move_from_gondolas(gondolas_map, points, gid=None):
    add_gondola = []
    if gid is None:
        keys = list(gondolas_map.keys())
    else:
        keys = np.array(list(gondolas_map.keys()))[gid:gid+1]
    for gondola_key in keys:
        gondola = gondolas_map[gondola_key]
        x1, y1, z1, x2, y2, z2 = 0, 0, 0, gondola.dim_x, gondola.dim_y, gondola.dim_z
        angle = -gondola.rotation_z
        trans_x, trans_y = gondola.translation_x, gondola.translation_y
        x2 = x2 if x2 != 0.0 else 0.1
        y2 = y2 if y2 != 0.0 else 0.1
        z2 = z2 if z2 != 0.0 else 0.1

        cs = np.cos(angle)
        sn = np.sin(angle)
        transform = np.array(
            [[cs, -sn, 0, 0], [sn, cs, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],]
        )
        transform2 = np.array(
            [[1, 0, 0, -trans_x], [0, 1, 0, -trans_y], [0, 0, 1, 0], [0, 0, 0, 1],]
        )
        # part = np.meshgrid(np.arange(-(x2-x1)/2, (x2-x1)/2,0.1), np.arange(z1,z2,0.1))
        part = points
        world = transform2 @ np.vstack(
            [
                part[:,0].flatten(),
                part[:,1].flatten(),
                part[:,2].flatten(),
                np.ones_like(part[:,0].flatten()),
            ]
        )
        world = world.T
        world = transform @ np.vstack(
            [
                world[:,0].flatten(),
                world[:,1].flatten(),
                world[:,2].flatten(),
                np.ones_like(part[:,0].flatten()),
            ]
        )
        world = world.T[:, :3]
        add_gondola.append(world)
    return add_gondola


def make_gondolas(gondolas_map):
    add_gondola = []
    for gondola_key in gondolas_map.keys():
        gondola = gondolas_map[gondola_key]
        x1, y1, z1, x2, y2, z2 = 0, 0, 0, gondola.dim_x, gondola.dim_y, gondola.dim_z
        angle = gondola.rotation_z
        trans_x, trans_y = gondola.translation_x, gondola.translation_y
        x2 = x2 if x2 != 0.0 else 0.1
        y2 = y2 if y2 != 0.0 else 0.1
        z2 = z2 if z2 != 0.0 else 0.1

        cs = np.cos(angle)
        sn = np.sin(angle)
        transform = np.array(
            [[cs, -sn, 0, trans_x], [sn, cs, 0, trans_y], [0, 0, 1, 0], [0, 0, 0, 1],]
        )
        # part = np.meshgrid(np.arange(-(x2-x1)/2, (x2-x1)/2,0.1), np.arange(z1,z2,0.1))
        part = np.meshgrid(
            np.arange(x1, x2, 0.1), np.arange(y1, y2, 0.1), np.arange(z1, z2, 0.2)
        )
        world = transform @ np.vstack(
            [
                part[0].flatten(),
                part[1].flatten(),
                part[2].flatten(),
                np.ones_like(part[0].flatten()),
            ]
        )
        world = world.T[:, :3] * 39.37
        add_gondola.append(world)
    return add_gondola


def make_gondolas_shelves(gondolas_map, shelve_map):
    add_gondola = []
    shelf_gondola = []
    for gondola_key in gondolas_map.keys():
        gondola = gondolas_map[gondola_key]
        x1, y1, z1, x2, y2, z2 = 0, 0, 0, gondola.dim_x, gondola.dim_y, gondola.dim_z
        angle = gondola.rotation_z
        trans_x, trans_y = gondola.translation_x, gondola.translation_y
        x2 = x2 if x2 != 0.0 else 0.1
        y2 = y2 if y2 != 0.0 else 0.1
        z2 = z2 if z2 != 0.0 else 0.1

        cs = np.cos(angle)
        sn = np.sin(angle)
        transform = np.array(
            [[cs, -sn, 0, trans_x], [sn, cs, 0, trans_y], [0, 0, 1, 0], [0, 0, 0, 1],]
        )
        # part = np.meshgrid(np.arange(-(x2-x1)/2, (x2-x1)/2,0.1), np.arange(z1,z2,0.1))
        part = np.meshgrid(
            np.arange(x1, x2 / 2, 0.1), np.arange(y1, y2, 0.1), np.arange(z1, z2, 0.01)
        )
        world = transform @ np.vstack(
            [
                part[0].flatten(),
                part[1].flatten(),
                part[2].flatten(),
                np.ones_like(part[0].flatten()),
            ]
        )
        world = world.T[:, :3] * 39.37
        add_gondola.append(world)
        shelf_stack = []
        for _, gondola_shelves in shelve_map[
            shelve_map["gondola_id"] == gondola_key
        ].iterrows():
            x1, y1, z1, x2, y2, z2 = (
                0,
                0,
                0,
                gondola_shelves["dim_x"],
                gondola_shelves.dim_y,
                gondola_shelves.dim_z,
            )

            angle = gondola_shelves.rotation_z
            trans_x, trans_y, trans_z = (
                gondola_shelves.translation_x,
                gondola_shelves.translation_y,
                gondola_shelves.translation_z,
            )
            x2 = x2 if x2 != 0.0 else 0.1
            y2 = y2 if y2 != 0.0 else 0.1
            z2 = z2 if z2 != 0.0 else 0.1
            cs = np.cos(angle)
            sn = np.sin(angle)
            transformshelf = np.array(
                [
                    [cs, -sn, 0, trans_x],
                    [sn, cs, 0, trans_y],
                    [0, 0, 1, trans_z],
                    [0, 0, 0, 1],
                ]
            )
            # part = np.meshgrid(np.arange(-(x2-x1)/2, (x2-x1)/2,0.1), np.arange(z1,z2,0.1))
            part = np.meshgrid(
                np.arange(x1, x2, 0.1), np.arange(y1, y2, 0.1), np.arange(z1, z2, 0.01)
            )
            world = transformshelf @ np.vstack(
                [
                    part[0].flatten(),
                    part[1].flatten(),
                    part[2].flatten(),
                    np.ones_like(part[0].flatten()),
                ]
            )
            world = transform @ world
            shelfpoints = world.T[:, :3] * 39.37
            if (shelfpoints.shape[0] * shelfpoints.shape[1]) == 0:
                print(1)
            shelf_stack.append(shelfpoints)
        #shelf_gondola.append(np.concatenate(shelf_stack))
        shelf_gondola.extend(shelf_stack)
    return add_gondola, shelf_gondola
