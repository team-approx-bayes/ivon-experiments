#!/bin/bash
ts=$(date +"%Y%m%dT%H%M%S")
datadir=../datasets
testsamples=1
testrepeat=1
device=cuda  # cpu/cuda/cuda:X
traindir=../trained/cifar10/sgd/resnet20/

savedir=../trained/flowers102_ood/sgd/resnet20/
mkdir -p ${savedir}
python -u run.py ${traindir} -ts ${testsamples} -tr ${testrepeat} \
  -dd ${datadir} -sd ${savedir} -d ${device} -so --ood_dataset flowers102 \
  |& tee -a ${savedir}/stdout-${ts}.log
