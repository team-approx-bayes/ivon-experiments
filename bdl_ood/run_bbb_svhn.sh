#!/bin/bash
ts=$(date +"%Y%m%dT%H%M%S")
datadir=../datasets
testsamples=1
testrepeat=64
device=cuda  # cpu/cuda/cuda:X
traindir=../trained/cifar10/bbb/resnet20_bbb/

savedir=../trained/svhn_ood/bbb/resnet20_bbb/test-${testsamples}-${testrepeat}
mkdir -p ${savedir}

python -u run.py ${traindir} -ts ${testsamples} -tr ${testrepeat} \
    -dd ${datadir} -sd ${savedir} -d ${device} -so \
    |& tee -a ${savedir}/stdout-${ts}.log
