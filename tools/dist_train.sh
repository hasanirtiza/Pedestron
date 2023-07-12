#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1
GPUS=$2

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=11073\
    $(dirname "$0")/train.py $CONFIG --seed 7 --launcher pytorch ${@:3}
