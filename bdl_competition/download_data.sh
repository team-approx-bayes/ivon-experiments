#! /bin/bash

mkdir data

curl https://raw.githubusercontent.com/izmailovpavel/neurips_bdl_starter_kit/main/eval-phase/eval_data/cifar_probs.csv -o data/cifar_probs.csv
curl https://raw.githubusercontent.com/izmailovpavel/neurips_bdl_starter_kit/main/eval-phase/eval_data/medmnist_probs.csv -o data/medmnist_probs.csv
curl https://raw.githubusercontent.com/izmailovpavel/neurips_bdl_starter_kit/main/eval-phase/eval_data/uci_samples.csv -o data/uci_samples.csv
curl https://storage.googleapis.com/neurips2021_bdl_competition/evaluation_phase/dermamnist_anon.npz -o data/dermamnist_anon.npz
curl https://storage.googleapis.com/neurips2021_bdl_competition/evaluation_phase/energy_anon.npz -o data/energy_anon.npz
curl https://storage.googleapis.com/neurips2021_bdl_competition/evaluation_phase/cifar_anon.npz -o data/cifar_anon.npz
