#!/bin/bash
ts=$(date +"%Y%m%dT%H%M%S")
datadir=../datasets  # TODO: specify the dataset folder
testrepeat=64  # mc samples
device=cuda  # cpu/cuda/cuda:X
traindir=../trained/cifar10/ivon/resnet20/
seed=0

python -u test_von.py ${traindir} -tr ${testrepeat} -s ${seed} \
    -dd ${datadir} -sd ${traindir} -d ${device} -pd -so \
    |& tee -a ${traindir}/stdout-${ts}.log
