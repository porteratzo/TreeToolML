#!/bin/bash
set -e

echo "Please download the Paris-Lille-3D dataset manually from the following link:"
echo "https://npm3d.fr/paris-lille-3d"
echo "Use the following password to access the dataset: Paris-Lille-3D"
echo "After downloading, please place the dataset in the 'datasets' directory."
echo "The expected directory structure is: datasets/Paris-Lille-3D"
read -p "Have you downloaded and placed the dataset in the correct directory? (y/n): " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Please download and place the dataset before proceeding."
    exit 1
fi