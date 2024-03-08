import argparse
from os.path import join as pjoin
import torch
from torch.optim import SGD
import sys

sys.path.insert(0, "..")
from common.utils import coro_timer, mkdirp, rm
from common.models import SWAGMODELS
from common.dataloaders import TRAINDATALOADERS, OUTCLASS
from common.trainutils import (
    coro_log,
    do_epoch,
    do_trainbatch,
    do_evalbatch,
    SummaryWriter,
    check_cuda,
    deteministic_run,
    bn_update,
    loadcheckpoint,
    savecheckpoint,
)


def get_args():
    parser = argparse.ArgumentParser(description="CIFAR10/100 SWAG training")
    parser.add_argument(
        "arch",
        default="vgg16_swag",
        choices=SWAGMODELS,
        help="model architecture: " + " | ".join(SWAGMODELS),
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
        default=1,
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
        default=300,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=0.05,
        type=float,
        metavar="LR",
        help="initial learning rate",
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
        "-tbd",
        "--tensorboard_dir",
        default="",
        type=str,
        help="if specified, record data for tensorboard.",
    )
    parser.add_argument(
        "-sse",
        "--swag_start",
        type=int,
        default=160,
        help="SWAG start epoch number",
    )
    parser.add_argument(
        "-slr",
        "--swag_lr",
        type=float,
        default=0.01,
        help="SWAG learning rate",
    )
    parser.add_argument(
        "-scf",
        "--swag_collectfreq",
        type=int,
        default=1,
        help="SWAG model collection frequency",
    )
    parser.add_argument(
        "-svf",
        "--swag_valfreq",
        type=int,
        default=5,
        help="SWAG model validation frequency",
    )
    parser.add_argument(
        "-sdr",
        "--swag_devrank",
        type=int,
        metavar="K",
        default=20,
        help="max rank of SWAG deviation matrix",
    )
    parser.add_argument(
        "-sbu",
        "--swag_bnupdate",
        action="store_true",
        help="update BatchNorm for averaged model",
    )
    parser.add_argument("--warmup", default=5, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    timer = coro_timer()
    t_init = next(timer)
    print(f">>> Training initiated at {t_init.isoformat()} <<<\n")

    args = get_args()
    print(args, end="\n\n")

    assert 0 <= args.swag_start <= args.epochs

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
    swamodel = None
    if args.resume:
        startepoch, swagmodel, optimizer, scheduler, dic = loadcheckpoint(
            args.resume, device
        )
        modelargs, modelkwargs = dic["modelargs"], dic["modelkwargs"]
        model = swagmodel.basemodel
        print(f"resumed from {args.resume}\n")
    else:
        startepoch = 0
        swagmodel = SWAGMODELS[args.arch](
            OUTCLASS[args.dataset], args.swag_devrank
        ).to(args.device)
        modelargs, modelkwargs = (
            OUTCLASS[args.dataset],
            args.swag_devrank,
        ), {}
        model = swagmodel.basemodel
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
                verbose=True,
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

    # standard training
    print("\n\n")
    print(f">>> Base training starts at {next(timer)[0].isoformat()} <<<\n")

    for e in range(startepoch, args.swag_start):
        # run training part
        log_ece.send((e, "train", len(train_loader), None))
        if e == args.warmup:
            # Creating a new scheduler will already change the learning rate
            print(f"End of warmup epochs, starting cosine annealing")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, eta_min=0.0, T_max=args.swag_start, verbose=True
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
            swagmodel,
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

    # SWAG training
    print("\n\n")
    print(f">>> SWAG training starts at {next(timer)[0].isoformat()} <<<\n")

    # adjust learning rate to swag_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = args.swag_lr

    for e in range(max(startepoch, args.swag_start), args.epochs):
        # run training part
        log_ece.send((e, "train", len(train_loader), None))
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
        # update lr scheduler
        # scheduler.step()

        print(f">>> Time elapsed: {next(timer)[1]} <<<\n")

        # collect base  model
        if (e - args.swag_start) % args.swag_collectfreq == 0:
            swagmodel.collect_model()

        # save checkpoint
        savecheckpoint(
            pjoin(args.save_dir, "checkpoint.pt"),
            args.arch,
            modelargs,
            modelkwargs,
            swagmodel,
            optimizer,
            scheduler,
        )

        # run evaluation part
        if len(val_loader) == 0:
            continue
        if (e - args.swag_start) % args.swag_valfreq == args.swag_valfreq - 1:
            with torch.no_grad():
                # validate base model
                log_ece.send((e, "val", len(val_loader), None))
                model.eval()
                do_epoch(
                    val_loader, do_evalbatch, log_ece, device, model=model
                )
                log_ece.throw(StopIteration)
                # validate swa model
                log_ece.send((e, "swaval", len(val_loader), None))
                swamodel = swagmodel.averaged_model(swamodel)
                if args.swag_bnupdate:
                    print("updating BatchNorm ...", end="")
                    bn_update(train_loader, swamodel, device=device)
                    print(" Done.")
                swamodel.eval()
                do_epoch(
                    val_loader, do_evalbatch, log_ece, device, model=swamodel
                )
                log_ece.throw(StopIteration)

        print(f">>> Time elapsed: {next(timer)[1]} <<<\n")
    log_ece.close()

    print(f">>> Training completed at {next(timer)[0].isoformat()} <<<\n")
