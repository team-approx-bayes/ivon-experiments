#!/bin/bash

lr=0.1
lr_final=0.001
beta1=0.9
beta2=0.9999
hess_init=0.005
epochs=120
bs=50 
ess=200000.0
wd=0.0002
mc=1
save_dir=./results_cifar10/

for seed in {1..6}; do
    python train.py cifar10 --learning_rate ${lr} --save_dir ${save_dir} --seed ${seed} \
        --epochs ${epochs} --hess_init ${hess_init} --tbatch ${bs} --momentum_hess ${beta2} \
        --momentum ${beta1} --lr_final ${lr_final} --warmup 0 \
        --ess ${ess} --wd ${wd} --mc_samples ${mc} --device cuda
done 

python test.py cifar10 --load_dir ${save_dir} --device cuda
