python scripts/PDE_train.py --cfg configs/experimentos_model/subconfigs/distance_out_loss_scale.yaml --amp 1 --resume 0 TRAIN.BATCH_SIZE 10 TRAIN.EPOCHS 150
python scripts/PDE_train.py --cfg configs/experimentos_model/subconfigs/distance_out_loss.yaml --amp 1 --resume 1 TRAIN.BATCH_SIZE 10 TRAIN.EPOCHS 150
#python scripts/PDE_train.py --cfg configs/datasets/original.yaml --amp 1 --resume 1 TRAIN.BATCH_SIZE 10 TRAIN.EPOCHS 150
#python scripts/PDE_train.py --cfg configs/datasets/center_filtered.yaml --amp 1 --resume 1 TRAIN.BATCH_SIZE 10 TRAIN.EPOCHS 150
python scripts/PDE_train.py --cfg configs/experimentos_model/subconfigs/distance_out.yaml --amp 1 --resume 1 TRAIN.BATCH_SIZE 10 TRAIN.EPOCHS 150
python scripts/PDE_train.py --cfg configs/experimentos_model/subconfigs/distance_loss.yaml --amp 1 --resume 1 TRAIN.BATCH_SIZE 10 TRAIN.EPOCHS 150
python scripts/PDE_train.py --cfg configs/datasets/trunks.yaml --amp 1 --resume 1 TRAIN.BATCH_SIZE 10 TRAIN.EPOCHS 150

