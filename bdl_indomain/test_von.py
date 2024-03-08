import argparse
from os.path import join as pjoin, exists
import torch
import torch.nn.functional as nnf
import sys

sys.path.append("..")
from common import models
from common.utils import coro_timer, mkdirp
from common.calibration import bins2diagram
from common.dataloaders import (
    TRAINDATALOADERS,
    TESTDATALOADER,
    OUTCLASS,
    NTRAIN,
    NTEST,
)
from common.trainutils import (
    coro_log_auroc,
    do_epoch,
    check_cuda,
    deteministic_run,
    SummaryWriter,
    loadcheckpoint,
    get_outputsaver,
    summarize_csv,
)


# if compile fails, fall back to eager
# torch._dynamo.config.suppress_errors = True


# standard load model function
def loadmodel(fromfile, device=torch.device("cpu")):
    dic = torch.load(fromfile, map_location=device)
    model = models.__dict__[dic["modelname"]](
        *dic["modelargs"], **dic.get("modelkwargs", {})
    ).to(device)
    model.load_state_dict(dic.pop("modelstates"))
    return model, dic


def get_args():
    parser = argparse.ArgumentParser(description="CIFAR10/100 IVON test")
    parser.add_argument(
        "traindir", type=str, help="path that collects all trained runs."
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
        "-b",
        "--batch",
        default=512,
        type=int,
        metavar="N",
        help="test mini-batch size",
    )
    parser.add_argument(
        "-tr",
        "--testrepeat",
        default=1,
        type=int,
        help="create test samples via process repeat",
    )
    parser.add_argument(
        "-vd",
        "--valdata",
        action="store_true",
        help="use validation instead of test data",
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
        "-pf",
        "--printfreq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency",
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
        default=0,
        help="fixes seed for reproducibility",
    )
    parser.add_argument(
        "-sd",
        "--save_dir",
        help="The directory used to save test results",
        default="save_temp",
        type=str,
    )
    parser.add_argument(
        "-so",
        "--saveoutput",
        action="store_true",
        help="save output probability",
    )
    parser.add_argument(
        "-dd",
        "--data_dir",
        help="The directory to find/store dataset",
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
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="run through seeds 5-24 instead of 0-4",
    )

    return parser.parse_args()


def get_dataloader(outclass: int, args):
    dataset = {v: k for k, v in OUTCLASS.items()}[outclass]
    # load data
    if args.valdata:
        _, data_loader = TRAINDATALOADERS[dataset](
            args.data_dir,
            args.tvsplit,
            args.workers,
            (device != torch.device("cpu")),
            args.batch,
            args.batch,
        )
    else:
        data_loader = TESTDATALOADER[dataset](
            args.data_dir,
            args.workers,
            (device != torch.device("cpu")),
            args.batch,
        )
    return data_loader


# generic boilerplate to eval/test a minibatch
# should be wrapped within torch.no_grad()
def do_evalbatch(batchinput, model, optimizer, repeat: int = 1):
    inputs, gt = batchinput[:-1], batchinput[-1]
    cumloss = 0.0
    cumprob = torch.zeros([])
    for _ in range(repeat):
        with optimizer.sampled_params():
            output = model(*inputs)
        ll = nnf.log_softmax(output, 1)  # get log-likelihood
        loss = nnf.nll_loss(ll, gt) / repeat
        cumloss += loss.item()
        prob = nnf.softmax(output, 1)  # get likelihood
        cumprob = cumprob + prob / repeat
    return cumprob, gt, cumloss


def do_evalbatch_mean(batchinput, model):
    inputs, gt = batchinput[:-1], batchinput[-1]
    output = model(*inputs)
    ll = nnf.log_softmax(output, 1)  # get log-likelihood
    loss = nnf.nll_loss(ll, gt)
    prob = nnf.softmax(output, 1)  # get likelihood
    return prob, gt, loss.item()


if __name__ == "__main__":
    timer = coro_timer()
    t_init = next(timer)
    print(f">>> Test initiated at {t_init.isoformat()} <<<\n")

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

    # prep tensorboard if specified
    if args.tensorboard_dir:
        mkdirp(args.tensorboard_dir)
        sw = SummaryWriter(args.tensorboard_dir)
    else:
        sw = None

    # distinguish between runs on validation data and test data
    prefix = "val_bayes" if args.valdata else "test_bayes"
    dataset = "cifar10"
    ndata = (
        NTRAIN[dataset] - int(args.tvsplit * NTRAIN[dataset])
        if args.valdata
        else NTEST[dataset]
    )

    log_ece = coro_log_auroc(
        sw, args.printfreq, args.bins, '' if args.ensemble else args.save_dir)

    # iterate over trained runs
    runs_range = range(5, 25) if args.ensemble else range(5)
    for runfolder in [str(i) for i in runs_range]:
        model_path = pjoin(args.traindir, runfolder, "checkpoint.pt")
        if not exists(model_path):
            print(f"skipping {pjoin(args.traindir, runfolder)}\n")
            continue
        print(f"loading model from {model_path} ...\n")
        # resume model
        _, model, optimizer = loadcheckpoint(model_path, device)[:3]
        print(optimizer.defaults)
        optimizer.mc_samples = args.testrepeat
        outclass = 10
        data_loader = get_dataloader(outclass, args)
        print(f">>> Test starts at {next(timer)[0].isoformat()} <<<\n")

        if args.saveoutput:
            outputsaver = get_outputsaver(
                args.save_dir,
                ndata,
                outclass,
                f"predictions_{prefix}_{runfolder}.npy",
            )
        else:
            outputsaver = None

        log_ece.send((runfolder, prefix, len(data_loader), outputsaver))
        with torch.no_grad():
            model.eval()
            do_epoch(
                data_loader,
                do_evalbatch,
                log_ece,
                device,
                model=model,
                optimizer=optimizer,
                repeat=args.testrepeat,
            )
        bins, _, avgvloss = log_ece.throw(StopIteration)[:3]
        if args.saveoutput:
            outputsaver.close()
        del model

        if args.plotdiagram:
            bins2diagram(
                bins,
                False,
                pjoin(args.save_dir, f"calibration_{prefix}_{runfolder}.pdf"),
            )

        print(f">>> Time elapsed: {next(timer)[1]} <<<\n")

    summarize_csv(pjoin(args.save_dir, f"{prefix}.csv"))

    # iterate over all trained runs (0-4), do mean predictions
    prefix = "val_map" if args.valdata else "test_map"
    for runfolder in [str(i) for i in range(5)]:
        model_path = pjoin(args.traindir, runfolder, "checkpoint.pt")
        if not exists(model_path):
            print(f"skipping {pjoin(args.traindir, runfolder)}\n")
            continue
        print(f"loading model from {model_path} ...\n")
        _, model, optimizer = loadcheckpoint(model_path, device)[:3]
        optimizer.mc_samples = args.testrepeat
        outclass = 10
        data_loader = get_dataloader(outclass, args)
        print(f">>> Test starts at {next(timer)[0].isoformat()} <<<\n")

        if args.saveoutput:
            outputsaver = get_outputsaver(
                args.save_dir,
                ndata,
                outclass,
                f"predictions_{prefix}_{runfolder}.npy",
            )
        else:
            outputsaver = None

        log_ece.send((runfolder, prefix, len(data_loader), outputsaver))
        with torch.no_grad():
            model.eval()
            do_epoch(
                data_loader, do_evalbatch_mean, log_ece, device, model=model
            )
        bins, _, avgvloss = log_ece.throw(StopIteration)[:3]
        if args.saveoutput:
            outputsaver.close()
        del model

        if args.plotdiagram:
            bins2diagram(
                bins,
                False,
                pjoin(args.save_dir, f"calibration_{prefix}_{runfolder}.pdf"),
            )

        print(f">>> Time elapsed: {next(timer)[1]} <<<\n")

    summarize_csv(pjoin(args.save_dir, f"{prefix}.csv"))

    log_ece.close()

    print(f">>> Test completed at {next(timer)[0].isoformat()} <<<\n")
