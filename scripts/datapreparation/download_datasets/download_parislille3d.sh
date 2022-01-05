#!/bin/bash
BASE_DIR=datasets/Paris_Lille3D

mkdir -p $BASE_DIR

export url_train="https://cloud.mines-paristech.fr/index.php/s/JhIxgyt0ALgRZ1O/download?path=%2Ftraining_50_classes"
#export url_test="https://cloud.mines-paristech.fr/index.php/s/JhIxgyt0ALgRZ1O/download?path=%2Ftraining_50_classes"

#wget -c -N --user user --password 'Paris-Lille-3D' -O  $BASE_DIR'/training_50_classes.zip'  $url_train
#wget --user user --password 'Paris-Lille-3D' -O  $BASE_DIR'/training_50_classes.zip'  $url_train
curl -o $BASE_DIR'/training_50_classes.zip' -u omontoyac1900@alumno.ipn.mx:Fifauefa1! $url_train
#wget -c -N -O $BASE_DIR'/test_10_classes.zip' $url_test

cd $BASE_DIR

#unzip test_10_classes.zip
unzip training_50_classes.zip

rm training_50_classes.zip
