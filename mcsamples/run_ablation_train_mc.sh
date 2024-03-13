#!/bin/bash
ts=$(date +"%Y%m%dT%H%M%S")
testrepeat=64  # test mc samples
device=cuda  # cpu/cuda/cuda:X
traindir=../trained/cifar10/ivon/resnet20/
seed=0
train_mc=$1
datadir=../datasets

savedir=${traindir}/test4mc-${train_mc}
mkdir -p ${savedir}
python -u test_ivon.py ${traindir} -tr ${testrepeat} -s ${seed} \
    -dd ${datadir} -sd "${savedir}" -d ${device} -pd -so \
    --train_mc "${train_mc}" |& tee -a "${savedir}"/stdout-${ts}.log
