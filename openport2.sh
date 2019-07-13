#!/usr/bin/env bash

REMOTE_USER=$1
REMOTE_HOST=$2
LOCAL_PORT=$3
REMOTE_PORT=$4

# bash openport2.sh yotamfr rack-jonathan-g03 8199 8888
# bash openport2.sh yotamfr rack-jonathan-g03 16006 6006
# bash openport2.sh yotamfr rack-jonathan-g03 3300 22

ssh -L${LOCAL_PORT}:localhost:${LOCAL_PORT} yotamfra@gate.tau.ac.il -t ssh -L${LOCAL_PORT}:localhost:${LOCAL_PORT} yotamfra@nova.cs.tau.ac.il -t ssh -L${LOCAL_PORT}:localhost:${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_HOST}
