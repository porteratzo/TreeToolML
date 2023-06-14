#python scripts/PDE_train.py --cfg configs/experimentos_model/subconfigs/distance_out_loss_scale.yaml --amp 1 --resume 1 TRAIN.BATCH_SIZE 10 TRAIN.EPOCHS 80
#python scripts/PDE_train.py --cfg configs/experimentos_model/subconfigs/distance_out_loss.yaml --amp 1 --resume 1 TRAIN.BATCH_SIZE 10 TRAIN.EPOCHS 150
#python scripts/PDE_train.py --cfg configs/experimentos_model/subconfigs/distance_out.yaml --amp 1 --resume 1 TRAIN.BATCH_SIZE 10 TRAIN.EPOCHS 150
#python scripts/PDE_train.py --cfg configs/experimentos_model/subconfigs/distance_loss.yaml --amp 1 --resume 1 TRAIN.BATCH_SIZE 10 TRAIN.EPOCHS 150
python scripts/PDE_train.py --cfg configs/datasets/subconfigs/original.yaml --amp 1 --resume 1 TRAIN.BATCH_SIZE 10 TRAIN.EPOCHS 150
python scripts/PDE_train.py --cfg configs/datasets/subconfigs/center_filtered.yaml --amp 1 --resume 1 TRAIN.BATCH_SIZE 10 TRAIN.EPOCHS 150
#python scripts/PDE_train.py --cfg configs/datasets/subconfigs/trunks.yaml --amp 1 --resume 1 TRAIN.BATCH_SIZE 10 TRAIN.EPOCHS 150
