#!/bin/bash
ts=$(date +"%Y%m%dT%H%M%S")
datadir=../datasets  # TODO: specify the dataset folder
testsamples=1
testrepeat=64
device=cuda  # cpu/cuda/cuda:X
traindir=../trained/cifar10/bbb/resnet20_bbb/

mkdir -p ${traindir}/test-${testsamples}-${testrepeat}

python -u test_sgd.py ${traindir} -ts ${testsamples} -tr ${testrepeat} \
    -dd ${datadir} -sd ${traindir}/test-${testsamples}-${testrepeat} \
    -d ${device} -pd -so \
    |& tee -a ${traindir}/test-${testsamples}-${testrepeat}/stdout-${ts}.log
