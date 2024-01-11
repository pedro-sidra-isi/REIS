> [!WARNING]  
> This repo has migrated to [github.com/pedro-sidra-isi/REIS](https://github.com/pedro-sidra-isi/REIS)
> **This page will not be updated!**

# REIS

Code for the paper "[REIS: A Visual Analytics Tool for Rendering and Exploring Instance Segmentation of Point Clouds](https://ieeexplore.ieee.org/document/10347129)"

This work was presented at SIBGRAPI 2023, see presentation [here](https://docs.google.com/presentation/d/1cweoimZlXPxmpqE3er2ZNStR14ouMLzgxFKx_WkDI48/edit?usp=drivesdk).

## Installation

We recommend using `mamba` and [MambaForge](https://github.com/conda-forge/miniforge). You can also use `conda`, but it will be slower:

```bash
# need to be on repository root folder
cd REIS
mamba env create
# or conda env create
conda activate reis
```

## Usage

We provide an example dashboard for the S3DIS dataset, to run:

```bash
python tools/s3dis_dash.py
```
