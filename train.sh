#!/usr/bin/env bash

SECONDS=0

EXP_NAME=semseg_dgcnn_$(date +"%Y-%m-%d_%H-%M-%S")
EXP_DIR=./outputs/$EXP_NAME
LOG_FILE=$EXP_DIR/out.log
mkdir $EXP_DIR
touch $LOG_FILE
echo $@ > $EXP_DIR/hyperparams.txt

/rhome/ge23yeq/.poetry/bin/poetry run python main_semseg.py --exp_name=$EXP_NAME --dataset ScanNet --num_sem_labels=21 $@ 2>&1 | tee $LOG_FILE

duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
