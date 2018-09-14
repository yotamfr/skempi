#!/usr/bin/env bash

#### Virtual Env ####

conda create -p skempi2 python=2.7

conda install -p skempi2 pip -c anaconda

#conda install -p skempi visdom -c conda-forge

### activate skempi
source activate skempi2

pip install -r requirements.txt

pip install ipython notebook ipykernel
python -m ipykernel install --user --name skempi2

pip install cogent