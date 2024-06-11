import argparse
import sys
import math 
from os.path import join as pjoin

import numpy as np
import einops
import torch
import torch.nn.functional as nnf
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.append("..")
from common.utils import coro_timer, mkdirp
from common.trainutils import (
    check_cuda,
    deteministic_run,
)
from pytorch_models import get_model 
from ivon import IVON

def get_args():
    parser = argparse.ArgumentParser(description="NeurIPS 2021 BDL Competition")
    parser.add_argument(
        "dataset",
        choices=('medmnist', 'cifar10', 'uci'),
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=0,
        type=int,
        metavar="N",
        help="number of data loading workers",
    )
    parser.add_argument(
        "-tb",
        "--tbatch",
        default=512,
        type=int,
        metavar="N",
        help="train mini-batch size",
    )
    parser.add_argument(
        "-vb",
        "--vbatch",
        default=512,
        type=int,
        metavar="N",
        help="eval mini-batch size",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=400,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=1.0,
        type=float,
        metavar="LR",
        help="initial learning rate",
    )
    parser.add_argument(
        "--lr_final",
        default=0.0,
        type=float,
        metavar="LR",
        help="final learning rate",
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "-d",
        "--device",
        default="cpu",
        type=str,
        metavar="DEV",
        help="run on cpu/cuda",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="if specified, fixes seed for reproducibility",
    )
    parser.add_argument(
        "-sd",
        "--save_dir",
        help="The directory used to save the trained models",
        default="",
        type=str,
        required=True
    )
    parser.add_argument(
        "-dd",
        "--data_dir",
        help="The directory to store dataset",
        default="../data",
        type=str,
    )
    parser.add_argument("--mc_samples", default=1, type=int)
    parser.add_argument("--momentum_hess", default=0.999, type=float)
    parser.add_argument("--hess_init", default=1.0, type=float)
    parser.add_argument("--ess", default=5e4, type=float)
    parser.add_argument("--clip_radius", default=float("inf"), type=float)
    parser.add_argument("--warmup", default=5, type=int)

    return parser.parse_args()

# noinspection PyShadowingNames
def do_trainbatch_ivon(batchinput, model, optimizer, lossfun):
    images, target = batchinput
    loss_samples = []
    prob_samples = []

    for _ in range(args.mc_samples):
        with optimizer.sampled_params(train=True):
            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            loss = lossfun(output, target) 
            loss.backward()
        loss_samples.append(loss.detach())
        prob_samples.append(nnf.softmax(output.detach(), -1))

    optimizer.step()

    loss = torch.mean(torch.stack(loss_samples, dim=0), dim=0)
    prob = torch.mean(torch.stack(prob_samples, dim=0), dim=0)

    return prob, target, loss.item()

def avneg_loglik_gaussian(output, y):
    """Computes the negative log-likelihood.

    The outputs of the network should be two-dimensional.
    The first output is treated as predictive mean. The second output is treated
    as inverse-softplus of the predictive standard deviation.
    """

    predictions_mean, predictions_std = output.split([1, 1], dim=-1)
    predictions_std = F.softplus(predictions_std)

    se = (predictions_mean - y)**2
    log_likelihood = (-0.5 * se / predictions_std**2 -
                      0.5 * torch.log(predictions_std**2 * 2 * math.pi))
    log_likelihood = torch.mean(log_likelihood)

    return -log_likelihood

if __name__ == "__main__":
    timer = coro_timer()
    t_init = next(timer)
    print(f">>> Training initiated at {t_init.isoformat()} <<<\n")

    args = get_args()
    print(args, end="\n\n")

    # if seed is specified, run deterministically
    if args.seed is not None:
        deteministic_run(seed=args.seed)

    # get device for this experiment
    device = torch.device(args.device)

    if device != torch.device("cpu"):
        check_cuda()

    # build train_dir for this experiment
    mkdirp(args.save_dir)
    with open(pjoin(args.save_dir, 'hyperparameters.txt'), 'w') as file:
        file.write(str(args))

    startepoch = 0

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

        if args.dataset == 'medmnist':
            train_dataset = TensorDataset(torch.from_numpy(x_train_).float() / 255.0,
                                    torch.from_numpy(y_train).long())
            test_dataset = TensorDataset(torch.from_numpy(x_test_). float() / 255.0,
                                    torch.from_numpy(y_test).long())
        else: 
            train_dataset = TensorDataset(torch.from_numpy(x_train_).float(),
                                    torch.from_numpy(y_train).long())
            test_dataset = TensorDataset(torch.from_numpy(x_test_). float(),
                                    torch.from_numpy(y_test).long())
        
        nc = y_train.max() + 1
        lossfun = nnf.cross_entropy
        di = {"num_classes" : nc}

    else: 
        train_dataset = TensorDataset(torch.from_numpy(x_train).float(),
                                torch.from_numpy(y_train).float())
        test_dataset = TensorDataset(torch.from_numpy(x_test).float(),
                                torch.from_numpy(y_test).float())
        nc = 0

        lossfun = avneg_loglik_gaussian
        di = {"num_features" : x_train.shape[1]}
    
    train_loader = DataLoader(train_dataset, batch_size=args.tbatch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.vbatch, shuffle=False)

    model = get_model(model_names[args.dataset], data_info=di).to(args.device)

    optimizer = IVON(
        model.parameters(),
        lr=args.learning_rate,
        mc_samples=args.mc_samples,
        beta1=args.momentum,
        beta2=args.momentum_hess,
        weight_decay=args.weight_decay,
        hess_init=args.hess_init,
        ess=args.ess,
        clip_radius=args.clip_radius,
    )

    scheduler = (
        torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0 / args.warmup,
            end_factor=1.0,
            total_iters=args.warmup,
            verbose=False,
        )
        if args.warmup > 0
        else None
    )

    for e in range(startepoch, args.epochs):
        # run training part
        if e == args.warmup:
            # Creating a new scheduler will already change the learning rate
            print(f"End of warmup epochs, starting cosine annealing")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, eta_min=args.lr_final, T_max=args.epochs,
                verbose=False)
        model.train()

        def mytrainfun(batch, model, optimizer):
            return do_trainbatch_ivon(batch, model, optimizer, lossfun)

        total_loss = 0.0
        for idx, data in enumerate(train_loader):
            data = (data[0].to(args.device), data[1].to(args.device))
            prob, target, loss = mytrainfun(data, model, optimizer)
            total_loss += loss 

        scheduler.step() 

        total_loss /= float(idx)
        print(f'Epoch {e}, Loss={total_loss}')

        time_per_epoch = next(timer)[1]

    torch.save((model.state_dict(), optimizer.state_dict()), 
               pjoin(args.save_dir, f'model{args.seed}.pt'))
