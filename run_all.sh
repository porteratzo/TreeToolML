#!/bin/bash
#source ~/anaconda3/etc/profile.d/conda.sh
#conda activate TreeTool
sh datapreparation/download_datasets/download_all.sh
python scripts/datapreparation/preprocess_datasets.py
python scripts/datapreparation/make_PDE_dataset.py
