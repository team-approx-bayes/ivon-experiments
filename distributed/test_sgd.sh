#!/bin/bash
ts=$(date +"%Y%m%dT%H%M%S")
datadir=$1
epochs=$2
dataset=imagenet
model=resnet50_imagenet
optimizer=sgd
testrepeat=0
device=cuda
traindir=../trained/${dataset}/${optimizer}_${epochs}/${model}
seed=0

python -u test.py ${traindir} ${dataset} -tr ${testrepeat} -s ${seed} \
    -dd ${datadir} -sd ${traindir} -d ${device} -pd -so \
    |& tee -a ${traindir}/stdout-${ts}.log
