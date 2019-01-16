#!/usr/bin/env bash

REMOTE_HOST=$1
LOCAL_PORT=$2
REMOTE_PORT=$3

ssh -L 22:yotamfra@gate.tau.ac.il:22 -L ${LOCAL_PORT}:${REMOTE_HOST}:${REMOTE_PORT} yotamfra@nova.cs.tau.ac.il

# sudo bash openport.sh rack-jonathan-g03 8199 8888
# sudo bash openport.sh rack-jonathan-g03 16006 6006
