#!/bin/bash

BASE_DIR=datasets/lucid

export url="lucid.wur.nl/storage/downloads/TLidar_TropicalForest/1_LidarTreePoinCloudData.zip"

mkdir -p $BASE_DIR

wget -c -N -O $BASE_DIR'/1_LidarTreePoinCloudData.zip' $url

cd $BASE_DIR

unzip -j 1_LidarTreePoinCloudData.zip

rm 1_LidarTreePoinCloudData.zip
