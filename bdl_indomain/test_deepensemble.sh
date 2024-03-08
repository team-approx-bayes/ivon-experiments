#!/bin/bash
ts=$(date +"%Y%m%dT%H%M%S")
datadir=../datasets
dataset=cifar10
ensemblecount=5
ensemblesize=5
device=cuda  # cpu/cuda/cuda:X
rootdir=../trained/cifar10/sgd/resnet20/

mkdir -p ${rootdir}/de-${ensemblecount}-${ensemblesize}

python -u test_sgd.py ${rootdir} -dd ${datadir} -sd ${rootdir} -d ${device} \
    --ensemble -pd -so \
    |& tee -a ${rootdir}/de-${ensemblecount}-${ensemblesize}/stdout-${ts}.log

python -u test_ensemble.py ${rootdir} ${dataset} -ec ${ensemblecount} \
    -sd ${rootdir}/de-${ensemblecount}-${ensemblesize} \
    -es ${ensemblesize} -dd ${datadir} -d ${device} -pd -so \
    |& tee -a ${rootdir}/de-${ensemblecount}-${ensemblesize}/stdout-${ts}.log
