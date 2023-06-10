for f in $1/subconfigs/*.yaml
do
  echo $f
  python scripts/PDE_train.py --cfg $f --amp 1 --resume 1 TRAIN.BATCH_SIZE 10 TRAIN.EPOCHS 100
done