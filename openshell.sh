#!/usr/bin/env bash
USER=$1
HOST=$2
ssh yotamfra@gate.tau.ac.il -t ssh yotamfra@nova.cs.tau.ac.il -t ssh ${USER}@${HOST}
# sudo bash openshell.sh yotamfr rack-jonathan-g03
