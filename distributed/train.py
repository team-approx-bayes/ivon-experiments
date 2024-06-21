import argparse
import os
from os.path import join as pjoin
import torch
torch.backends.cudnn.benchmark = True
torch.autograd.profiler.emit_nvtx(False)
torch.autograd.profiler.profile(False)

import torch.nn.functional as nnf
import torch.distributed as dist
import torch.multiprocessing as mp
from ivon import IVON
import sys

sys.path.append("..")
from common.utils import coro_timer, mkdirp
from common.models import STANDARDMODELS
from common.dataloaders import get_imagenet_test_loader, get_imagenet_train_loader, ImageNetInfo
from common.trainutils import (
    coro_log_timed,
    do_epoch,
    do_trainbatch,
    do_evalbatch,
    check_cuda,
    savecheckpoint,
    loadcheckpoint,
)
from common.adahessian import AdaHessian

def get_args():
    parser = argparse.ArgumentParser(description="Distributed IVON training")
    parser.add_argument(
        "--arch",
        choices=STANDARDMODELS,
        help="model architecture: " + " | ".join(STANDARDMODELS),
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers",
    )
    parser.add_argument(
        "-b",
        "--batchsize",
        default=128,
        type=int,
        metavar="N",
        help="batch-size (gets divided among workers)",
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
        "-pf",
        "--printfreq",
        default=200,
        type=int,
        metavar="N",
        help="print frequency",
    )
    parser.add_argument(
        "-r",
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="resume training from checkpoint",
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
        default="save_temp",
        type=str,
    )
    parser.add_argument(
        "-dd",
        "--data_dir",
        help="The directory to store dataset",
        default="/datasets/",
        type=str,
    )
    parser.add_argument(
        "-nb",
        "--bins",
        default=20,
        type=int,
        help="number of bins for ece & reliability diagram",
    )
    parser.add_argument("--mc_samples", default=1, type=int)
    parser.add_argument("--momentum_hess", default=0.999, type=float)
    parser.add_argument("--hess_init", default=1.0, type=float)
    parser.add_argument("--ess", default=5e4, type=float)
    parser.add_argument("--clip_radius", default=float("inf"), type=float)
    parser.add_argument("--warmup", default=5, type=int)
    parser.add_argument(
        "-opt",
        "--optimizer",
        default="ivon",
        choices=["ivon", "sgd", "adamw", "adahessian"],
        type=str,
        help="optimizer to use",
    )
    
    return parser.parse_args()


# noinspection PyShadowingNames
def do_trainbatch_ivon(batchinput, model, optimizer):
    images, target = batchinput
    loss_samples = []
    prob_samples = []

    for _ in range(optimizer.mc_samples):
        with optimizer.sampled_params(train=True), model.no_sync():
            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            loss = nnf.cross_entropy(output, target)
            loss.backward()

        loss_samples.append(loss.detach())
        prob_samples.append(nnf.softmax(output.detach(), -1))

    optimizer.step()

    loss = torch.mean(torch.stack(loss_samples, dim=0), dim=0)
    prob = torch.mean(torch.stack(prob_samples, dim=0), dim=0)

    return prob, target, loss.item()


def do_trainbatch_adahessian(batchinput, model, optimizer):
    images, target = batchinput
    loss_samples = []
    prob_samples = []

    optimizer.zero_grad(set_to_none=True)
    output = model(images)
    loss = nnf.cross_entropy(output, target)
    loss.backward(create_graph=True)
    loss_samples.append(loss.detach())
    prob_samples.append(nnf.softmax(output.detach(), -1))

    optimizer.step()

    loss = torch.mean(torch.stack(loss_samples, dim=0), dim=0)
    prob = torch.mean(torch.stack(prob_samples, dim=0), dim=0)

    return prob, target, loss.item()


train_functions = {
    "sgd": do_trainbatch,
    "adamw": do_trainbatch,
    "adahessian": do_trainbatch_adahessian,
    "ivon": do_trainbatch_ivon,
}


def get_optimizer(args, model):
    if args.optimizer == "ivon":
        return IVON(
            model.parameters(),
            lr=args.learning_rate,
            mc_samples=args.mc_samples,
            beta1=args.momentum,
            beta2=args.momentum_hess,
            weight_decay=args.weight_decay,
            hess_init=args.hess_init,
            ess=args.ess,
            clip_radius=args.clip_radius,
            sync=True,
            debias=True #debias when training from epoch 0
        )

    elif args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

    elif args.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    elif args.optimizer == "adahessian":
        return AdaHessian(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )


def main(args):
    global_rank = int(os.environ["RANK"])
    gpu = int(os.environ["LOCAL_RANK"])   
    world_size = int(os.environ["WORLD_SIZE"]) 
    local_batchsize = int(args.batchsize / world_size)

    # only print info once
    if gpu == 0: 
        print(args, end="\n\n")
        check_cuda()

        # build train_dir for this experiment
        mkdirp(args.save_dir)

    # if seed is specified, run deterministically
    if args.seed is not None:
        #deteministic_run(seed=args.seed)
        torch.manual_seed(args.seed * world_size + global_rank)
    else: 
        torch.manual_seed(42 * world_size + global_rank)
    
    print("Use GPU: {} for training".format(gpu))
        
    timer = coro_timer()
    t_init = next(timer)
    if gpu == 0: 
        print(f">>> Training initiated at {t_init.isoformat()} <<<\n")
        
    dist.init_process_group(backend='nccl')

    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    torch.cuda.set_device(gpu)
    device = torch.device('cuda:{}'.format(gpu))

    # resume or initialize
    if args.resume:
        startepoch, model, optimizer, scheduler, dic = loadcheckpoint(
            args.resume, device, epochs=args.epochs 
        )
        args.optimizer = type(optimizer).__name__.lower() 
        optimizer.debias = False #dont debias gradients when resuming
        optimizer.sync = True 

        modelargs, modelkwargs = dic["modelargs"], dic["modelkwargs"]
        print(f"resumed from {args.resume}\n")
        print(optimizer.defaults)

        model.cuda(gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
        
    else:
        startepoch = 0
        modelargs, modelkwargs = (
            ImageNetInfo.outclass,
            ImageNetInfo.imgshape,
        ), {}
        model = STANDARDMODELS[args.arch](*modelargs, **modelkwargs).to(
            device
        )
        model.cuda(gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

        optimizer = get_optimizer(args, model)
        print(optimizer.defaults)
        
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
                
    test_loader = get_imagenet_test_loader(
        args.data_dir,
        workers=args.workers,
        batch=local_batchsize,
        device=device,
        distributed=False,
    )

    # load data
    train_loader = get_imagenet_train_loader(
        args.data_dir,
        workers=args.workers,
        tbatch=local_batchsize, 
        device=device, 
        distributed=True,
    )

    # perform training
    log_ece = coro_log_timed(None, args.printfreq, args.bins, args.save_dir, global_rank, append=args.resume)

    if gpu == 0:
        data_size = ImageNetInfo.counts['train']
        print(
            f"datasize {int(data_size)}, paramsize "
            f"{sum(p.nelement() for p in model.parameters())}"
        )

        print(f">>> Training starts at {next(timer)[0].isoformat()} <<<\n")

    for e in range(startepoch, args.epochs):
        # run training part
        log_ece.send((e, "train", len(train_loader), None))

        if e == args.warmup:
            # Creating a new scheduler will already change the learning rate
            print(f"End of warmup epochs, starting cosine annealing")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, eta_min=0.0, T_max=args.epochs, verbose=True
            )
        model.train()

        do_epoch(
            train_loader,
            train_functions[args.optimizer],
            log_ece,
            device,
            model=model,
            optimizer=optimizer,
        )
        scheduler.step()
        log_ece.throw(StopIteration)

        if gpu == 0: 
            # save checkpoint
            savecheckpoint(
                pjoin(args.save_dir, "checkpoint.pt"),
                args.arch,
                modelargs,
                modelkwargs,
                model.module, 
                optimizer,
                scheduler,
            )

            # save intermediate checkpoints 
            checkpoint_epochs = [0, 1, 2, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 250, 300, 350, 400]
            if e in checkpoint_epochs:
                savecheckpoint(
                    pjoin(args.save_dir, "checkpoint%03d.pt" % (e + 1)),
                    args.arch,
                    modelargs,
                    modelkwargs,
                    model.module,
                    optimizer,
                    scheduler,
                )

            time_per_epoch = next(timer)[1]
            print(f">>> Time elapsed: {time_per_epoch} <<<\n")

            # log time per epochs
            with open(pjoin(args.save_dir, "time.csv"), "a+") as file:
                file.write("%d,%f\n" % (e, time_per_epoch.total_seconds()))

        # run evaluation part (only on one GPU)
        if gpu == 0: 
            log_ece.send((e, "test", len(test_loader), None))
            with torch.no_grad():
                model.eval()
                do_epoch(test_loader, do_evalbatch, log_ece, device, model=model)
            bins, _, avgvloss = log_ece.throw(StopIteration)[:3]

        if gpu == 0:
            print(f">>> Time elapsed: {next(timer)[1]} <<<\n")

    log_ece.close()

    if gpu == 0:
        print(f">>> Training completed at {next(timer)[0].isoformat()} <<<\n")

if __name__ == "__main__":
    main(get_args()) 