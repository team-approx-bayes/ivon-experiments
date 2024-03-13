#!/bin/bash
ts=$(date +"%Y%m%dT%H%M%S")
dataset=cifar10
model=resnet20
epochs=200
device=cuda  # cpu/cuda/cuda:X
lr=0.2
momentum=0.9
momentum_hess=0.99999
wdecay=2e-4
ess=5e4
tbatch=50
vbatch=50
hess_init=0.5
split=1.0
seed=${1}
mc_samples=${2}
datadir=../datasets

savedir="../trained/${dataset}/ivon/${model}/${seed}-${mc_samples}"

mkdir -p "${savedir}"
    python -u train_ivon.py ${model} ${dataset} -s "${seed}" -dd ${datadir} \
        -sd "${savedir}" -lr ${lr} -e ${epochs} --weight-decay ${wdecay} \
        --momentum ${momentum} --momentum_hess ${momentum_hess} --hess_init ${hess_init} \
        --ess ${ess} --device ${device} -pd --tbatch ${tbatch} --vbatch ${vbatch} \
        --tvsplit ${split} --mc_samples "${mc_samples}" |& tee -a "${savedir}/stdout-${ts}.log"
