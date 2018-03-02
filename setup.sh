#!/usr/bin/env bash

mkdir data
mkdir stride

#### STRIDE installation ####
wget http://webclu.bio.wzw.tum.de/stride/stride.tar.gz

tar -xvf stride.tar.gz -C stride

cd ./stride

make

cd ../
rm stride.tar.gz


#### Download SKEMPI ####
cd ./data

wget https://life.bsc.es/pid/mutation_database/SKEMPI_1.1.csv

wget https://life.bsc.es/pid/mutation_database/SKEMPI_pdbs.tar.gz

mkdir pdbs
tar -xvf SKEMPI_pdbs.tar.gz -C pdbs

cd ../


