"""script to run standard/MCDropout cifar10/cifar100 classification training"""
import argparse
from os.path import join as pjoin
import torch
from torch.optim import SGD
import sys

sys.path.insert(0, "..")
from common.models import STANDARDMODELS, MCDROPMODELS
from common.utils import coro_timer, mkdirp
from common.dataloaders import TRAINDATALOADERS, OUTCLASS
from common.trainutils import (
    coro_log,
    do_epoch,
    do_trainbatch,
    do_evalbatch,
    SummaryWriter,
    check_cuda,
    deteministic_run,
    savecheckpoint,
    loadcheckpoint,
)


MODELS = {**STANDARDMODELS, **MCDROPMODELS}


def get_args():
    parser = argparse.ArgumentParser(
        description="CIFAR10/100 standard/MCDropout training"
    )
    parser.add_argument(
        "arch",
        default="vgg16",
        choices=MODELS,
        help="model architecture: " + " | ".join(MODELS),
    )
    parser.add_argument(
        "dataset",
        default="cifar10",
        choices=TRAINDATALOADERS,
        help="datasets: " + " | ".join(TRAINDATALOADERS),
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
        "-sp",
        "--tvsplit",
        default=0.9,
        type=float,
        metavar="RATIO",
        help="ratio of data used for training",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=200,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
    )
    parser.add_argument(
        "-lrr",
        "--lr_ratio",
        default=0.0,
        type=float,
        metavar="LR",
        help="ratio of final / initial lr",
    )
    parser.add_argument("-wd", "--weight_decay", default=2e-4, type=float)
    parser.add_argument(
        "-m",
        "--momentum",
        default=0.9,
        type=float,
        metavar="M",
        help="momentum",
    )
    parser.add_argument(
        "-pf",
        "--printfreq",
        default=100,
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
        default="save_temp",
        type=str,
    )
    parser.add_argument(
        "-dd",
        "--data_dir",
        help="The directory to store dataset",
        default="../data",
        type=str,
    )
    parser.add_argument(
        "-nb",
        "--bins",
        default=20,
        type=int,
        help="number of bins for ece & reliability diagram",
    )
    parser.add_argument(
        "-pd",
        "--plotdiagram",
        action="store_true",
        help="plot reliability diagram for best val",
    )
    parser.add_argument(
        "-tbd",
        "--tensorboard_dir",
        default="",
        type=str,
        help="if specified, record data for tensorboard.",
    )
    parser.add_argument("--warmup", default=5, type=int)

    return parser.parse_args()


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

    # resume or initialize
    if args.resume:
        startepoch, model, optimizer, scheduler, dic = loadcheckpoint(
            args.resume, device
        )
        modelargs, modelkwargs = dic["modelargs"], dic["modelkwargs"]
        print(f"resumed from {args.resume}\n")
    else:
        startepoch = 0
        model = MODELS[args.arch](OUTCLASS[args.dataset]).to(args.device)
        modelargs, modelkwargs = (OUTCLASS[args.dataset],), {}

        optimizer = SGD(
            model.parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        scheduler = (
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0 / args.warmup,
                end_factor=1.0,
                total_iters=args.warmup,
            )
            if args.warmup > 0
            else None
        )

    # prep tensorboard if specified
    if args.tensorboard_dir:
        mkdirp(args.tensorboard_dir)
        sw = SummaryWriter(args.tensorboard_dir)
    else:
        sw = None

    # load data
    train_loader, val_loader = TRAINDATALOADERS[args.dataset](
        args.data_dir,
        args.tvsplit,
        args.workers,
        (device != torch.device("cpu")),
        args.tbatch,
        args.vbatch,
    )

    # perform training
    log_ece = coro_log(sw, args.printfreq, args.bins, args.save_dir)
    print(f">>> Training starts at {next(timer)[0].isoformat()} <<<\n")

    for e in range(startepoch, args.epochs):
        # run training part
        log_ece.send((e, "train", len(train_loader), None))
        if e == args.warmup:
            # Creating a new scheduler will already change the learning rate
            print(f"End of warmup epochs, starting cosine annealing")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, eta_min=0.0, T_max=args.epochs
            )
        model.train()
        do_epoch(
            train_loader,
            do_trainbatch,
            log_ece,
            device,
            model=model,
            optimizer=optimizer,
        )
        log_ece.throw(StopIteration)
        # update lr scheduler and decay
        scheduler.step()
        # save checkpoint
        savecheckpoint(
            pjoin(args.save_dir, "checkpoint.pt"),
            args.arch,
            modelargs,
            modelkwargs,
            model,
            optimizer,
            scheduler,
        )

        print(f">>> Time elapsed: {next(timer)[1]} <<<\n")
        # run evaluation part
        if len(val_loader) == 0:
            continue
        log_ece.send((e, "val", len(val_loader), None))
        with torch.no_grad():
            model.eval()
            do_epoch(val_loader, do_evalbatch, log_ece, device, model=model)
        bins, _, avgvloss = log_ece.throw(StopIteration)[:3]

        print(f">>> Time elapsed: {next(timer)[1]} <<<\n")

    log_ece.close()

    print(f">>> Training completed at {next(timer)[0].isoformat()} <<<\n")
