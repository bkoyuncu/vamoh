# VamoH

This is the official PyTorch implementation of [VaMoH](https://arxiv.org/abs/2302.06223).

# Installation

First, create a conda environment and activate it.

```bash
conda create --name vamoh python=3.8  --no-default-packages
conda activate vamoh
```

Then, install cartopy

```bash
conda install --channel conda-forge cartopy
```

Then, install the requirements

```bash
pip install -r requirements_pip.txt
pip install -e .
```

## Usage

First move to ./run folder, and run experiments for Shapes3D dataset.

To run with cpu

```bash
python main.py --cfg     device "cpu" dataset.name shapes3d_10 dataset.missing_perc 0.0
```

To run with gpu change it wih

```bash
python main.py --cfg ./configs/models/shapes.yaml device "cuda:0" dataset.name shapes3d_10 dataset.missing_perc 0.0
```

If you want to run it will full dataset instead of 10% of it

```bash
python main.py --cfg ./configs/models/shapes.yaml device "cuda:0" device shapes3d dataset.missing_perc 0.0

```

If you want to train the model with point dropout give the amount as

```bash
python main.py --cfg ./configs/models/shapes.yaml device "cpu" dataset.name shapes3d_10 dataset.missing_perc 0.3
```

## Citation

If you use this code in your research, please cite the following paper:

```
@inproceedings{koyuncu2023variational,
      title={Variational Mixture of HyperGenerators for Learning Distributions Over Functions}, 
      author={Batuhan Koyuncu and Pablo Sanchez-Martin and Ignacio Peis and Pablo M. Olmos and Isabel Valera},
      year={2023},
      booktitle={International Conference on Machine Learning}}
 ```

# License

PyTorch VAMoH is licensed under the MIT License.
