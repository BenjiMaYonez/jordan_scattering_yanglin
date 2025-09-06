# Key Components
- `jordan_scatter`: subpackage for pytorch model definition
- `configs`: subpackage for configuration of hyperparameter
- `scripts`: command-line entry point
- `experiments`: saved experiments and configuration for reprodusibility
- `images`: subpackage for image utility, saved images
# Get Start

## Package Installation
Before running, install the package in your python environment `pip install -e .`

## Running file
Sanity Check, lossless inversion: `python scripts/inverse.py --config configs/full_inverse.yaml`

Invertible: `python scripts/inverse.py --config configs/inverse.yaml`