#!/usr/bin/env bash

#### Virtual Env ####
conda create -p skempi2 python=2.7
conda install -p skempi2 pip -c anaconda
conda install -p skempi2  pytorch torchvision cuda80 -c pytorch

### activate skempi2
source activate skempi2
conda --add channels salilab
conda install modeller
pip install --upgrade pip

pip install numpy
pip install --no-binary pandas -I pandas
pip install xlrd
pip install biopython==1.68
pip install tqdm
pip install scipy
pip install sklearn
pip install pymongo
pip install cogent
pip install requests

pip install theano
conda install mkl-service

#python -m ipykernel install --user --name skempi
