#!/bin/bash
set -e

echo "Setting up base directory for Toronto3D dataset..."
BASE_DIR=datasets/Toronto3D

export url="https://xx9lca.sn.files.1drv.com/y4mUm9-LiY3vULTW79zlB3xp0wzCPASzteId4wdUZYpzWiw6Jp4IFoIs6ADjLREEk1-IYH8KRGdwFZJrPlIebwytHBYVIidsCwkHhW39aQkh3Vh0OWWMAcLVxYwMTjXwDxHl-CDVDau420OG4iMiTzlsK_RTC_ypo3z-Adf-h0gp2O8j5bOq-2TZd9FD1jPLrkf3759rB-BWDGFskF3AsiB3g"

echo "Creating base directory: $BASE_DIR"
mkdir -p $BASE_DIR

echo "Downloading Toronto3D dataset..."
wget -c -N -O $BASE_DIR'/Toronto_3D.zip' $url

echo "Navigating to base directory..."
cd $BASE_DIR

echo "Unzipping Toronto3D dataset..."
unzip -j Toronto_3D.zip

echo "Cleaning up zip file..."
rm Toronto_3D.zip