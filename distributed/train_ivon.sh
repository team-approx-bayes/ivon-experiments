#!/bin/bash
ts=$(date +"%Y%m%dT%H%M%S")
dataset=imagenet
model=resnet50_imagenet
optimizer=ivon
lr=2.5
wdecay=5e-5
ess=1281167.0
hess_init=0.05
momentum=0.9
momentum_hess=0.999995 
datadir=$1
ngpus=$2
batchsize=$3
seed=$4
epochs=$5
 
savedir=../trained/${dataset}/${optimizer}_${epochs}/${model}

mkdir -p ${savedir}/${seed}
OMP_NUM_THREADS=12 torchrun --standalone --nproc_per_node=${ngpus} train.py --arch ${model} \
       -opt ${optimizer} -s $seed -dd ${datadir} -j 12 \
       -sd ${savedir}/${seed} -lr ${lr} -e ${epochs} --weight-decay ${wdecay} \
       --momentum_hess ${momentum_hess} --hess_init ${hess_init} --ess ${ess} \
       --momentum ${momentum} --batchsize ${batchsize} |& tee -a ${savedir}/${seed}/stdout-${ts}.log
