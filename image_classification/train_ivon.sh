#!/bin/bash
ts=$(date +"%Y%m%dT%H%M%S")
datadir=../datasets
dataset=$1  # cifar10/cifar100/tinyimagenet
model=$2  # resnet20/resnet18wide/preresnet110/densenet121
optimizer=ivon
epochs=200
device=cuda  # cpu/cuda/cuda:X
lr=0.2
momentum=0.9
momentum_hess=0.99999
wdecay=2e-4
tbatch=50
vbatch=50
hess_init=0.5
split=1.0
seed=$3 

savedir=../trained/${dataset}/${optimizer}/${model}

case $dataset in

  cifar10 | cifar100)
    ess=50000
    ;;

  tinyimagenet)
    ess=200000
    ;;

  *)
    echo -n "unknown dataset: ${dataset}"
    exit 1
    ;;
esac


mkdir -p ${savedir}/${seed}
python -u train.py ${model} ${dataset} -opt ${optimizer} -s $seed -dd ${datadir} \
       -sd ${savedir}/${seed} -lr ${lr} -e ${epochs} --weight-decay ${wdecay} \
       --momentum ${momentum} --momentum_hess ${momentum_hess} --hess_init ${hess_init} \
       --ess ${ess} --device ${device} -pd --tbatch ${tbatch} --vbatch ${vbatch} \
       --tvsplit ${split} |& tee -a ${savedir}/${seed}/stdout-${ts}.log
