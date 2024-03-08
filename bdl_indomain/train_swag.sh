#!/bin/bash
ts=$(date +"%Y%m%dT%H%M%S")
datadir=../datasets  # TODO: specify the dataset folder
dataset=cifar10
model=resnet20_swag
epochs=200
swag_start=160
base_lr=0.05
swag_lr=0.01
device=cuda  # cpu / cuda / cuda:X
split=1.0
wdecay=2e-4
tbatch=50
vbatch=50

savedir=../trained/${dataset}/swag/${model}

for (( seed=0; seed<5; seed++ ))
do
    mkdir -p ${savedir}/${seed}
    python -u train_swag.py  ${model} ${dataset} \
        -s ${seed} -dd ${datadir} -sd ${savedir}/${seed} -d ${device} \
        -e ${epochs} -lr ${base_lr} -sse ${swag_start} -slr ${swag_lr} \
        -sbu -wd ${wdecay} -sp ${split} \
        --tbatch ${tbatch} --vbatch ${vbatch} \
        |& tee -a ${savedir}/${seed}/stdout-${ts}.log
done
