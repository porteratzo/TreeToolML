sh scripts/datapreparation/download_datasets/download_all.sh
python scripts/datapreparation/preprocess_datasets.py --cfg configs/datasets/trunks.yaml
python scripts/datapreparation/make_PDE_dataset.py --cfg configs/datasets/trunks.yaml
