#!/bin/bash
ts=$(date +"%Y%m%dT%H%M%S")
datadir=../datasets  # TODO: specify the dataset folder
swagmodels=64
samplemode=modelwise
device=cuda  # cpu/cuda/cuda:X
traindir=../trained/cifar10/swag/resnet20_swag/
savedir=${traindir}/test-${swagmodels}-${samplemode}

mkdir -p ${savedir}

python -u test_swag.py ${traindir} -sms ${swagmodels} -ssm ${samplemode} \
    -dd ${datadir} -sd ${savedir} -d ${device} -pd -so -sbu \
    |& tee -a ${savedir}/stdout-${ts}.log
