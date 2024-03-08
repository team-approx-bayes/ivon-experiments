#!/bin/bash
ts=$(date +"%Y%m%dT%H%M%S")
datadir=../datasets
testrepeat=64  # mc samples
device=cuda  # cpu/cuda/cuda:X
traindir=../trained/cifar10/ivon/resnet20/

savedir=../trained/flowers102_ood/ivon/resnet20/
mkdir -p ${savedir}
python -u run.py ${traindir} -tr ${testrepeat} \
    -dd ${datadir} -sd ${savedir} -d ${device} -so  --ood_dataset flowers102 \
    |& tee -a ${savedir}/stdout-${ts}.log

# - usage examples:
# bash ood_ivon.sh train_ivon/cifar10/preresnet20_frntlu/
