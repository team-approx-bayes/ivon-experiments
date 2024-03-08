#!/bin/bash
ts=$(date +"%Y%m%dT%H%M%S")
datadir=../datasets  # TODO: specify the dataset folder
dataset=cifar10
model=resnet20_mcdrop
epochs=200
device=cuda  # cpu/cuda/cuda:X
lr=0.2
split=1.0
wdecay=2e-4
tbatch=50
vbatch=50

savedir=../trained/${dataset}/mcdrop/${model}

for (( seed=0; seed<5; seed++ ))
do
    mkdir -p ${savedir}/${seed}
    python -u train_sgd.py ${model} ${dataset} -s $seed -dd ${datadir} \
        -lr ${lr} -sd ${savedir}/${seed} -e ${epochs} -d ${device} -pd \
        --tbatch ${tbatch} --vbatch ${vbatch} -wd ${wdecay} -sp ${split} \
        |& tee -a ${savedir}/${seed}/stdout-${ts}.log
done
