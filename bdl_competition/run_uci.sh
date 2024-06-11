#!/bin/bash

lr=0.02
lr_final=0.02
beta1=0.9
beta2=0.999
hess_init=0.5
epochs=60
bs=8
ess=6000
wd=0.005
mc=5
save_dir=./results_uci/

for seed in {1..6}; do
    python train.py uci --learning_rate ${lr} --save_dir ${save_dir} --seed ${seed} \
        --epochs ${epochs} --hess_init ${hess_init} --tbatch ${bs} --momentum_hess ${beta2} \
        --momentum ${beta1} --lr_final ${lr_final} --warmup 0 \
        --ess ${ess} --wd ${wd} --mc_samples ${mc} --device cpu
done 

python test.py uci --load_dir ${save_dir}
