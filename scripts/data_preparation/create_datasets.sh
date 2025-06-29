set -e
bash scripts/data_preparation/download_datasets/download_all.sh
python scripts/data_preparation/preprocess_datasets_1.py --cfg configs/datasets/subconfigs/trunks.yaml
python scripts/data_preparation/preprocess_extract_centers_2.py
python scripts/data_preparation/make_PDE_dataset_3.py --cfg configs/datasets/subconfigs/original.yaml
python scripts/data_preparation/make_PDE_dataset_3.py --cfg configs/datasets/subconfigs/center_filtered.yaml
python scripts/data_preparation/make_PDE_dataset_3.py --cfg configs/datasets/subconfigs/trunks.yaml
