#!/bin/bash
BASE_DIR=datasets/Paris_Lille3D

mkdir -p $BASE_DIR

export url_train="https://cloud.mines-paristech.fr/index.php/s/JhIxgyt0ALgRZ1O/download?path=%2Ftraining_50_classes"
#export url_test="https://cloud.mines-paristech.fr/index.php/s/JhIxgyt0ALgRZ1O/download?path=%2Ftraining_50_classes"


wget -c -N -O $BASE_DIR'/training_50_classes.zip'  $url_train
#wget -c -N -O $BASE_DIR'/test_10_classes.zip' $url_test

cd $BASE_DIR

#unzip test_10_classes.zip
unzip training_50_classes.zip
