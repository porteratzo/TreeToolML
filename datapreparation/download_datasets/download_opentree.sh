#!/bin/bash

BASE_DIR=datasets/open_tree

export url="https://github.com/VUKOZ-OEL/3dforest-data/archive/refs/heads/master.zip"

mkdir -p $BASE_DIR

wget -c -N -O $BASE_DIR'/master.zip' $url

cd $BASE_DIR

unzip -j master.zip

rm master.zip