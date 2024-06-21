#!/bin/bash
ts=$(date +"%Y%m%dT%H%M%S")
dataset=imagenet
model=resnet50_imagenet
datadir=$1 
ngpus=$2
batchsize=$3
seed=$4
epochs=$5
optimizer=$6

savedir=../trained/${dataset}/${optimizer}_${epochs}/${model}
resumefile=${savedir}/${seed}/checkpoint.pt

mkdir -p ${savedir}/${seed}
OMP_NUM_THREADS=12 torchrun --standalone --nproc_per_node=${ngpus} train.py --arch ${model}  \
       -opt ${optimizer} -s $seed -dd ${datadir} -j 12 --resume ${resumefile} \
       -sd ${savedir}/${seed} -e ${epochs} \
       --batchsize ${batchsize} |& tee -a ${savedir}/${seed}/stdout-${ts}.log
