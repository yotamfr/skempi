#!/usr/bin/env bash

#!/usr/bin/env bash

REMOTE_HOST=$1
LOCAL_PORT=$2
REMOTE_PORT=$3

#ssh -N -f -L  ${LOCAL_PORT}:${REMOTE_HOST}:${REMOTE_PORT} yotamfra@nova.cs.tau.ac.il
ssh -L  ${LOCAL_PORT}:${REMOTE_HOST}:${REMOTE_PORT} yotamfra@nova.cs.tau.ac.il

# ./pylab.sh rack-jonathan-g04 8199 8888
