#!/bin/bash
ts=$(date +"%Y%m%dT%H%M%S")
datadir=../datasets
testsamples=32
testrepeat=1
device=cuda  # cpu/cuda/cuda:X
traindir=../trained/cifar10/mcdrop/resnet20_mcdrop/

savedir=../trained/flowers102_ood/mcdrop/resnet20_mcdrop/test-${testsamples}-${testrepeat}
mkdir -p ${savedir}

python -u run.py ${traindir} -ts ${testsamples} -tr ${testrepeat} \
    -dd ${datadir} -sd ${savedir} -d ${device} -so  --ood_dataset flowers102 \
    |& tee -a ${savedir}/stdout-${ts}.log
