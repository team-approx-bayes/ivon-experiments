#!/bin/bash
ts=$(date +"%Y%m%dT%H%M%S")
datadir=../datasets  # TODO: specify the dataset folder
dataset=cifar10
model=resnet20_bbb
epochs=200
device=cuda  # cpu/cuda/cuda:X
lr=0.002
wdecay=2e-4
temperature=1.0
std_init=1.5
tbatch=50
vbatch=50
split=1.0

savedir=../trained/${dataset}/bbb/${model}

for (( seed=0; seed<5; seed++ ))
do
    mkdir -p ${savedir}/${seed}
    python -u train_bbb.py ${model} ${dataset} -s $seed -dd ${datadir} \
        -sd ${savedir}/${seed} -lr ${lr} -e ${epochs} --weight_decay ${wdecay} \
        --std_init ${std_init} --temperature ${temperature} \
         --device ${device} -pd --tbatch ${tbatch} --vbatch ${vbatch} \
        --tvsplit ${split} |& tee -a ${savedir}/${seed}/stdout-${ts}.log
done
