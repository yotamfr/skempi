#!/usr/bin/env bash
HOST=$1

ssh -L 22:yotamfra@gate.tau.ac.il:22 yotamfra@nova.cs.tau.ac.il -t ssh ${HOST}

# sudo bash openshell.sh yotamfr@rack-jonathan-g03