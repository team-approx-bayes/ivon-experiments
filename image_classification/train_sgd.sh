#!/bin/bash
ts=$(date +"%Y%m%dT%H%M%S")
datadir=../datasets
dataset=$1  # cifar10/cifar100/tinyimagenet
model=$2  # resnet20/resnet18wide/preresnet110/densenet121
optimizer=sgd
epochs=200
device=cuda  # cpu/cuda/cuda:X
lr=0.1
wdecay=2e-4
momentum=0.9
tbatch=50
vbatch=50
split=1.0
seed=$3 

savedir=../trained/${dataset}/${optimizer}/${model}

mkdir -p ${savedir}/${seed}
python -u train.py ${model} ${dataset} -opt ${optimizer} -s $seed -dd ${datadir} \
       -sd ${savedir}/${seed} -lr ${lr} -e ${epochs} --weight-decay ${wdecay} \
       --momentum ${momentum} --device ${device} -pd --tbatch ${tbatch} --vbatch ${vbatch} \
       --tvsplit ${split} |& tee -a ${savedir}/${seed}/stdout-${ts}.log

