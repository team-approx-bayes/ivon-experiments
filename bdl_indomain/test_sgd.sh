#!/bin/bash
ts=$(date +"%Y%m%dT%H%M%S")
datadir=../datasets
testsamples=1
testrepeat=1
device=cuda  # cpu/cuda/cuda:X
traindir=../trained/cifar10/sgd/resnet20/

python -u test_sgd.py ${traindir} -ts ${testsamples} -tr ${testrepeat} \
    -dd ${datadir} -sd ${traindir} -d ${device} -pd -so \
    |& tee -a ${traindir}/stdout-${ts}.log
