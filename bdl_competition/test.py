import argparse
import os
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import einops
import scipy.stats
from scipy.special import softmax

from ivon import IVON
from pytorch_models import get_model

def list_pt_files(directory):
    pt_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".pt"):
                pt_files.append(os.path.join(root, file))
    return pt_files

def agreement(predictions: np.array, reference: np.array):
    """Returns 1 if predictions match and 0 otherwise."""
    return (predictions.argmax(axis=-1) == reference.argmax(axis=-1)).mean()


def total_variation_distance(predictions: np.array, reference: np.array):
    """Returns total variation distance."""
    return np.abs(predictions - reference).sum(axis=-1).mean() / 2.


def w2_distance(predictions: np.array, reference: np.array):
    """Returns W-2 distance """
    NUM_SAMPLES_REQUIRED = 1000
    assert predictions.shape[0] == reference.shape[0], "wrong predictions shape"
    assert predictions.shape[1] == NUM_SAMPLES_REQUIRED, "wrong number of samples"
    return -np.mean([scipy.stats.wasserstein_distance(pred, ref) for
                     pred, ref in zip(predictions, reference)])

cifar_gt = np.genfromtxt("data/cifar_probs.csv")
medmnist_gt = np.genfromtxt("data/medmnist_probs.csv")
uci_gt = np.genfromtxt("data/uci_samples.csv")

parser = argparse.ArgumentParser(description="NeurIPS 2021 BDL Competition")
parser.add_argument(
    "dataset",
    choices=('medmnist', 'cifar10', 'uci'),
)
parser.add_argument(
    "-ld",
    "--load_dir",
    default="",
    type=str,
    required=True
)
parser.add_argument(
    "--device",
    default="cpu",
    type=str,
)
args = parser.parse_args()

device = args.device
testmc = 64

model_names = { 
    'uci' : 'uci_mlp', 
    'cifar10' : 'cifar_alexnet', 
    'medmnist' : 'medmnist_lenet' }

data_names = { 
    'uci' : 'data/energy_anon.npz', 
    'cifar10' : 'data/cifar_anon.npz', 
    'medmnist' : 'data/dermamnist_anon.npz' }

data = np.load(data_names[args.dataset])
x_train = data["x_train"]
y_train = data["y_train"]
x_test = data["x_test"]
y_test = data["y_test"]

if args.dataset == 'cifar10' or args.dataset == 'medmnist':
    x_train_ = einops.rearrange(x_train, "n h w c -> n c h w")
    x_test_ = einops.rearrange(x_test, "n h w c -> n c h w") 

    train_dataset = TensorDataset(torch.from_numpy(x_train_).float() / 255.0,
                            torch.from_numpy(y_train).long())
    test_dataset = TensorDataset(torch.from_numpy(x_test_). float() / 255.0,
                            torch.from_numpy(y_test).long())
    
    nc = y_train.max() + 1

else: 
    train_dataset = (torch.from_numpy(x_train).float(),
                    torch.from_numpy(y_train).float())
    test_dataset = (torch.from_numpy(x_test).float(),
                    torch.from_numpy(y_test).float())
    nc = 0

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

model_filenames = list_pt_files(args.load_dir)

if args.dataset == 'cifar10' or args.dataset == 'medmnist':
    all_probs = [] 
    for fn in model_filenames: 
        print(f'Processing model {fn}...')
        model = get_model(model_names[args.dataset], data_info={"num_classes" : nc}).to(device)
        optimizer = IVON(model.parameters(), lr=0.1, ess=5000)

        modeldict, optdict = torch.load(fn)

        model.load_state_dict(modeldict)
        optimizer.load_state_dict(optdict)

        for _ in range(testmc):
            with optimizer.sampled_params(train=False):
                probs_per_seed = [] 
                for idx, data in enumerate(test_loader):
                    images, targets = data[0].to(device), data[1].to(device)
                    logits = model(images)
                    probs = torch.softmax(logits, axis=1)
                    probs_per_seed.append(probs.detach().cpu().numpy()) 

                probs_per_seed = np.vstack(probs_per_seed)
                all_probs.append(probs_per_seed.reshape(1, *probs_per_seed.shape))

    probs = np.concatenate(all_probs, axis=0)
    predictions = probs.mean(axis=0)

    if args.dataset == 'medmnist':
        agreements = agreement(predictions[:1000], medmnist_gt[:1000]), agreement(predictions, medmnist_gt)
        tvs = (total_variation_distance(predictions[:1000], medmnist_gt[:1000]),
            total_variation_distance(predictions, medmnist_gt))

    else: 
        agreements = agreement(predictions[:10000], cifar_gt[:10000]), agreement(predictions, cifar_gt)
        tvs = (total_variation_distance(predictions[:10000], cifar_gt[:10000]),
            total_variation_distance(predictions, cifar_gt))
        
    with open(pjoin(args.load_dir, 'results.txt'), "w") as file:
        file.write(f"Public Agreement: {agreements[0]}\nFull Agreement: {agreements[1]}\n\n")
        file.write(f"Public TV: {tvs[0]}\nFull TV: {tvs[1]}\n\n")

else: 
    num_samples = 1000
    all_samples = [] 
    models = [] 
    opts = [] 

    for fn in model_filenames: 
        print(f'Processing model {fn}...')
        modeldict, optdict = torch.load(fn)
        models.append(modeldict)
        opts.append(optdict)

    for _ in range(num_samples): 
        component = np.random.randint(0, len(models))
        model = get_model(model_names[args.dataset], data_info={"num_features" : test_dataset[0].shape[1]}).to(device)
        optimizer = IVON(model.parameters(), lr=0.1, ess=5000)

        model.load_state_dict(models[component])
        optimizer.load_state_dict(opts[component])

        with optimizer.sampled_params(train=False):
            probs_per_seed = [] 
            test_preds = model(test_dataset[0]).detach().cpu()
            mu, sigma = test_preds.split([1, 1], dim=-1)
            sigma = F.softplus(sigma)
            mu, sigma = mu[:, 0].numpy(), sigma[:, 0].numpy()
            eps = np.random.randn(test_dataset[0].shape[0], 1)
            samples = mu[:, None] + eps * sigma[:, None]

            all_samples.append(samples) 

    samples = np.hstack(all_samples)
    result_public, result_full = w2_distance(samples[:100, ], uci_gt.T[:100, ]), w2_distance(samples, uci_gt.T)
    print(f"Public: {result_public}\nFull: {result_full}")
    with open(pjoin(args.load_dir, 'results.txt'), 'w') as file:
        file.write(f"Public: {result_public}\nFull: {result_full}")


