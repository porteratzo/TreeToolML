sh scripts/datapreparation/download_datasets/download_all.sh
python scripts/datapreparation/preprocess_datasets_1.py --cfg configs/datasets/trunks.yaml
python scripts/datapreparation/preprocess_extract_centers_2.py
python scripts/datapreparation/make_PDE_dataset_3.py --cfg configs/datasets/original.yaml
python scripts/datapreparation/make_PDE_dataset_3.py --cfg configs/datasets/center_filtered.yaml
python scripts/datapreparation/make_PDE_dataset_3.py --cfg configs/datasets/trunks.yaml
