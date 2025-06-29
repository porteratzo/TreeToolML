#!/bin/bash
set -e
set -x

bash scripts/data_preparation/download_datasets/download_lucid.sh
bash scripts/data_preparation/download_datasets/download_opentree.sh
bash scripts/data_preparation/download_datasets/download_toronto3d.sh
bash scripts/data_preparation/download_datasets/download_parislille3d.sh
