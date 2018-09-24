#!/usr/bin/env bash

#### Virtual Env ####
conda create -p skempi2 python=2.7
conda install -p skempi2 pip -c anaconda
conda install -p skempi2  pytorch torchvision cuda80 -c pytorch

### activate skempi2
source activate skempi2
conda --add channels salilab
conda install -c salilab modeller
pip install --upgrade pip

pip install numpy
pip install pandas
pip install xlrd
pip install tensoarboardX --upgrade
pip install biopython==1.68
pip install tqdm
pip install scipy
pip install sklearn
pip install pymongo
pip install sphinx
pip install cogent
pip install requests
pip install futures

pip install theano
conda install mkl-service

#pip install ipython notebook
#python -m ipykernel install --user --name skempi2
