set -ex
python scripts/PDE_train.py --cfg configs/datasets/subconfigs/trunks.yaml --amp 1 --resume 0 TRAIN.BATCH_SIZE 10 TRAIN.EPOCHS 1
python scripts/PDE_test.py --cfg configs/datasets/subconfigs/trunks.yaml --amp 1 TRAIN.BATCH_SIZE 24 VALIDATION.PATH 'datasets/custom_data/trunks/testing_data'
python notebooks/benchmarking/benchmark_treetoolml.py --cfg configs/datasets/subconfigs/trunks.yaml
python notebooks/benchmarking/draw_graphs_treetoolml.py --cfg configs/datasets/subconfigs/trunks.yaml