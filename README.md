# TreeToolML

TreeToolML is a toolkit for individual tree extraction from terrestrial laser scanning (TLS) point clouds using deep learning algorithms. It implements the methods described in the paper:

## Features

- Deep learning-based pointwise direction embedding (PDE-net) for tree center prediction
- Voxel-based region growing for instance-level tree separation
- Modular pipeline for training, testing, benchmarking, and visualization

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/porteratzo/TreeToolML.git
   cd TreeToolML
   ```

2. **Create the conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate treetoolml
   ```

3. **Install additional dependencies:**
   - Run the following script to get the correct versions of required tools:
     ```bash
     bash get_Individual_tree_extraction.sh
     ```

## Data Preparation

Prepare datasets by running:
```bash
bash scripts/data_preparation/create_datasets.sh
```
> **Note:** One of the steps requires a manual download. Follow the instructions in the script.

## Usage

To run the full pipeline (training, testing, benchmarking, and plotting), execute:
```bash
bash basic.sh
```

### Individual Tree Extraction

For details on the individual tree extraction module, see [`IndividualTreeExtraction/README.md`](./IndividualTreeExtraction/README.md).

## Repository Structure

- `IndividualTreeExtraction/` - Core implementation for tree extraction
- `scripts/` - Data preparation and utility scripts
- `result/` - Output results
- `data/` - Input and processed datasets

## Citation

Pending

## License

This project is for academic research purposes. For other uses, please contact the authors.

## Contact

For questions or support, contact [Omar Montoya](mailto:omar.alfonso.montoya@hotmail.com).

