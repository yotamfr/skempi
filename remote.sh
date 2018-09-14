#!/usr/bin/env bash

HOST=$1
PORT=$2

#ssh -N -f -L ${PORT}:${HOST}:22 yotamfra@nova.cs.tau.ac.il
ssh -L ${PORT}:${HOST}:22 yotamfra@nova.cs.tau.ac.il

# bash remote.sh rack-jonathan-g04 4400