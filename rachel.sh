#!/usr/bin/env bash

HOST=$1
PORT=$2

ssh -L ${PORT}:${HOST}:22 yotamfr@trachel-srv.cs.haifa.ac.il

# bash rachel.sh trachel-srv2.cs.haifa.ac.il 2200
