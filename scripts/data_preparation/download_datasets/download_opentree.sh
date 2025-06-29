#!/bin/bash
set -e

echo "Setting up base directory for OpenTree dataset..."
BASE_DIR=datasets/open_tree

export url="https://github.com/VUKOZ-OEL/3dforest-data/archive/refs/heads/master.zip"

echo "Creating base directory: $BASE_DIR"
mkdir -p $BASE_DIR

echo "Downloading OpenTree dataset..."
wget -c -N -O $BASE_DIR'/master.zip' $url

echo "Navigating to base directory..."
cd $BASE_DIR

echo "Unzipping OpenTree dataset..."
unzip -j master.zip

echo "Cleaning up zip file..."
rm master.zip