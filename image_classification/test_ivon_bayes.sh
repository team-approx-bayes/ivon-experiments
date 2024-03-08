#!/bin/bash
ts=$(date +"%Y%m%dT%H%M%S")
datadir=../datasets
dataset=$1  # cifar10/cifar100/tinyimagenet
model=$2  # resnet20/resnet18wide/preresnet110/densenet121
optimizer=ivon
testrepeat=64  # mc samples
device=cuda
traindir=../trained/${dataset}/${optimizer}/${model}
seed=0

python -u test.py ${traindir} ${dataset} -tr ${testrepeat} -s ${seed} \
    -dd ${datadir} -sd ${traindir} -d ${device} -pd -so \
    |& tee -a ${traindir}/stdout-${ts}.log