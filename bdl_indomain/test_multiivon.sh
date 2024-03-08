#!/bin/bash
ts=$(date +"%Y%m%dT%H%M%S")
datadir=../datasets
dataset=cifar10
ensemblecount=5
ensemblesize=5
testrepeat=64  # mc samples
device=cuda  # cpu/cuda/cuda:X
rootdir=../trained/cifar10/ivon/resnet20/
seed=0

mkdir -p ${rootdir}/de-${ensemblecount}-${ensemblesize}

#python -u test_von.py ${rootdir} -tr ${testrepeat} -s ${seed} \
#    -dd ${datadir} -sd ${rootdir} -d ${device} -pd -so --ensemble \
#    |& tee -a ${rootdir}/de-${ensemblecount}-${ensemblesize}/stdout-${ts}.log

python -u test_ensemble.py ${rootdir} ${dataset} -ec ${ensemblecount} \
    -sd ${rootdir}/de-${ensemblecount}-${ensemblesize} --prefix test_bayes \
    -es ${ensemblesize} -dd ${datadir} -d ${device} -pd -so \
    |& tee -a ${rootdir}/de-${ensemblecount}-${ensemblesize}/stdout-${ts}.log
