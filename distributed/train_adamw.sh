#!/bin/bash
ts=$(date +"%Y%m%dT%H%M%S")
dataset=imagenet
model=resnet50_imagenet
optimizer=adamw
lr=1e-3
wdecay=1e-1
datadir=$1 
ngpus=$2
batchsize=$3
seed=$4
epochs=$5

savedir=../trained/${dataset}/${optimizer}_${epochs}/${model}

mkdir -p ${savedir}/${seed}
OMP_NUM_THREADS=12 torchrun --standalone --nproc_per_node=${ngpus} train.py --arch ${model}  \
       -opt ${optimizer} -s $seed -dd ${datadir} -j 12 \
       -sd ${savedir}/${seed} -lr ${lr} -e ${epochs} --weight-decay ${wdecay} \
       --batchsize ${batchsize} |& tee -a ${savedir}/${seed}/stdout-${ts}.log
