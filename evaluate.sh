#!/usr/bin/env bash

SECONDS=0

if [[ ! -e evaluation_results.csv ]]; then
    touch evaluation_results.csv
    echo -e "loss,acc,avgacc,mIoU,model_root\n" > evaluation_results.csv
fi

/rhome/ge23yeq/.poetry/bin/poetry run python main_semseg.py --eval=True --dataset ScanNet --num_sem_labels=21 $@

duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
