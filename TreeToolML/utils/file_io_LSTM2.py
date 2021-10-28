from typing import Dict, List, Tuple
import pandas as pd
from collections import defaultdict
from inter_det.data.feature import Event, FeaturePoint
import pickle
import os

TargetKey = Tuple[str, str]

##Read batch of annotations
def read_annotations_dl(start, end, filename: str) -> Dict[TargetKey, List[Event]]:
    # load anotation csv
    df = pd.read_csv(filename)
    annotations: Dict[TargetKey, List[Event]] = defaultdict(list)
    
    ##Takes annotations of the current batch
    temp_df = df[start:end] 

    for _, row in temp_df.iterrows():
        k = (str(row.rec_id), str(row.target_id))
        event = Event(row.t_start, row.t_end, gondola_id=-1, action=row.action)
        annotations[k].append(event)
    return annotations


##Read batch of features
def read_features_dl(record_list, feat_path: str) -> Dict[TargetKey, List[FeaturePoint]]:
    all_features = {}
    if os.path.isdir(feat_path):

        ##Gets just the pk files of the recordings in the current batch
        files = []
        for record in record_list:
            temp_file = os.path.join(feat_path, str(record)+".pk")

            if os.path.isfile(temp_file):
                files.append(temp_file)

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
