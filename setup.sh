#!/usr/bin/env bash

#### Virtual Env ####

conda create -p skempi python=3.6

conda install -p skempi pip -c anaconda

#conda install -p skempi visdom -c conda-forge

### activate skempi
source activate skempi

pip install -r requirements.txt

python -m ipykernel install --user --name skempi