python scripts/PDE_test.py --cfg configs/datasets/trunks.yaml TRAIN.BATCH_SIZE 10 VALIDATION.PATH 'datasets/custom_data/center_filtered/testing_data'
python scripts/PDE_test.py --cfg configs/datasets/original.yaml TRAIN.BATCH_SIZE 10 VALIDATION.PATH 'datasets/custom_data/center_filtered/testing_data'
python scripts/PDE_test.py --cfg configs/datasets/center_filtered.yaml TRAIN.BATCH_SIZE 10 VALIDATION.PATH 'datasets/custom_data/center_filtered/testing_data'
#python scripts/PDE_test.py --cfg configs/experimentos_model/subconfigs/distance_out.yaml TRAIN.BATCH_SIZE 10 VALIDATION.PATH 'datasets/custom_data/center_filtered/testing_data'
#python scripts/PDE_test.py --cfg configs/experimentos_model/subconfigs/distance_out_loss.yaml TRAIN.BATCH_SIZE 10 VALIDATION.PATH 'datasets/custom_data/center_filtered/testing_data'
#python scripts/PDE_test.py --cfg configs/experimentos_model/subconfigs/distance_loss.yaml TRAIN.BATCH_SIZE 10 VALIDATION.PATH 'datasets/custom_data/center_filtered/testing_data'
