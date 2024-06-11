#!/bin/bash

lr=0.1
lr_final=0.02
beta1=0.9
beta2=0.99999
hess_init=0.5
epochs=250
bs=50 
ess=20000.0
wd=0.005
mc=1
save_dir=./results_medmnist/

for seed in {1..6}; do
    python train.py medmnist --learning_rate ${lr} --save_dir ${save_dir} --seed ${seed} \
        --epochs ${epochs} --hess_init ${hess_init} --tbatch ${bs} --momentum_hess ${beta2} \
        --momentum ${beta1} --lr_final ${lr_final} --warmup 0 \
        --ess ${ess} --wd ${wd} --mc_samples ${mc} --device cuda
done 

python test.py medmnist --load_dir ${save_dir} --device cuda
