#!/bin/bash
ts=$(date +"%Y%m%dT%H%M%S")
datadir=../datasets
swagmodels=64
samplemode=modelwise  # modelwise / layerwise / channelwise
device=cuda  # cpu/cuda/cuda:X
traindir=../trained/cifar10/swag/resnet20_swag/

savedir=../trained/svhn_ood/swag/resnet20_swag/test-${swagmodels}-${samplemode}
mkdir -p ${savedir}

python -u run.py ${traindir} -sms ${swagmodels} -ssm ${samplemode} \
    -dd ${datadir} -sd ${savedir} -d ${device} -so \
    |& tee -a ${savedir}/stdout-${ts}.log
