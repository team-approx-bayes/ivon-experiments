import argparse
from os.path import join as pjoin
import torch
import torch.nn.functional as nnf
from ivon import IVON
import sys

sys.path.append("..")
from common.utils import coro_timer, mkdirp, rm
from common.models import STANDARDMODELS
from common.dataloaders import (
    TRAINDATALOADERS,
    TESTDATALOADER,
    NTRAIN,
    OUTCLASS,
    INSIZE,
)
from common.trainutils import (
    coro_log_timed,
    do_epoch,
    do_trainbatch,
    do_evalbatch,
    SummaryWriter,
    check_cuda,
    deteministic_run,
    savecheckpoint,
    loadcheckpoint,
)
from common.adahessian import AdaHessian


def get_args():
    parser = argparse.ArgumentParser(description="CIFAR10/100 IVON training")
    parser.add_argument(
        "arch",
        choices=STANDARDMODELS,
        help="model architecture: " + " | ".join(STANDARDMODELS),
    )
    parser.add_argument(
        "dataset",
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

    for _ in range(args.mc_samples):
        with optimizer.sampled_params(train=True):
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
        modelargs, modelkwargs = (
            OUTCLASS[args.dataset],
            INSIZE[args.dataset],
        ), {}
        model = STANDARDMODELS[args.arch](*modelargs, **modelkwargs).to(
            args.device
        )
        data_size = int(NTRAIN[args.dataset] * args.tvsplit)

        optimizer = get_optimizer(args, model)

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

    # try compile
    # model = torch.compile(model)

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

    test_loader = TESTDATALOADER[args.dataset](
        args.data_dir,
        args.workers,
        (device != torch.device("cpu")),
        args.tbatch,
    )

    # perform training
    log_ece = coro_log_timed(sw, args.printfreq, args.bins, args.save_dir)

    print(
        f"datasize {int(data_size * args.tvsplit)}, paramsize "
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
                optimizer, eta_min=args.lr_final, T_max=args.epochs,
                verbose=True)
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
        # update lr scheduler and decay
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

        checkpoint_epochs = [0, 1, 2, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200]
        if e in checkpoint_epochs:
            savecheckpoint(
                pjoin(args.save_dir, "checkpoint%03d.pt" % (e + 1)),
                args.arch,
                modelargs,
                modelkwargs,
                model,
                optimizer,
                scheduler,
            )
        print(f'Max memory usage {torch.cuda.max_memory_allocated()}')
        time_per_epoch = next(timer)[1]
        print(f">>> Time elapsed: {time_per_epoch} <<<\n")

        # log time per epochs
        with open(pjoin(args.save_dir, "time.csv"), "a+") as file:
            file.write("%d,%f\n" % (e, time_per_epoch.total_seconds()))

        # run evaluation part
        log_ece.send((e, "test", len(test_loader), None))
        with torch.no_grad():
            model.eval()
            do_epoch(test_loader, do_evalbatch, log_ece, device, model=model)
        bins, _, avgvloss = log_ece.throw(StopIteration)[:3]

        if len(val_loader) == 0:
            continue

        # run evaluation part
        log_ece.send((e, "val", len(val_loader), None))
        with torch.no_grad():
            model.eval()
            do_epoch(val_loader, do_evalbatch, log_ece, device, model=model)
        bins, _, avgvloss = log_ece.throw(StopIteration)[:3]

        print(f">>> Time elapsed: {next(timer)[1]} <<<\n")

    log_ece.close()

    print(f">>> Training completed at {next(timer)[0].isoformat()} <<<\n")
