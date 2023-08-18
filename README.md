# REIS

Code for the paper "REIS: A Visual Analytics Tool for Rendering and Exploring Instance Segmentation of Point Clouds"

This work started as a final project for CMP596, see presentation [here](https://docs.google.com/presentation/d/1GWUrdDCbHLROz9JThbOdXl3evL4CjDpOlCe82D2A9X8/edit?usp=sharing).

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
