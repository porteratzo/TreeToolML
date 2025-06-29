#!/bin/bash
set -e

BASE_DIR="datasets/lucid"
DATASET_URL="https://data.4tu.nl/file/31225173-3a25-4793-bb34-63709e4cf300/0909910c-60ac-48d1-837f-e8094a968b0c"
ARCHIVE_NAME="LidarTreePoinCloudData.7z"

setup_base_directory() {
    echo "Setting up base directory: $BASE_DIR"
    mkdir -p "$BASE_DIR"
}

download_dataset() {
    echo "Downloading Lucid dataset..."
    wget -c -N -O "$BASE_DIR/$ARCHIVE_NAME" "$DATASET_URL"
}

extract_dataset() {
    echo "Extracting Lucid dataset..."
    7z x $BASE_DIR/$ARCHIVE_NAME -o$BASE_DIR
}

cleanup() {
    echo "Cleaning up archive file..."
    rm "$BASE_DIR/$ARCHIVE_NAME"
}

main() {
    setup_base_directory
    download_dataset
    extract_dataset
    #cleanup
    echo "Lucid dataset setup complete."
}

main
