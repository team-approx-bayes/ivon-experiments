"""script to run swag cifar10/cifar100 classification test"""
import argparse
from os import listdir
from os.path import join as pjoin, isdir, exists
import torch
import torch.nn.functional as nnf
import sys

sys.path.insert(0, "..")
from common.swag import SWAG
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
    loadcheckpoint,
    coro_log_auroc,
    do_epoch,
    do_evalbatch,
    check_cuda,
    deteministic_run,
    SummaryWriter,
    bn_update,
    get_outputsaver,
    summarize_csv,
)


def get_args():
    parser = argparse.ArgumentParser(description="CIFAR10/100 swag test")
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
        default=128,
        type=int,
        metavar="N",
        help="test mini-batch size",
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
        "-sms",
        "--swag_modelsamples",
        type=int,
        default=1,
        help="number of swag model samples",
    )
    parser.add_argument(
        "-ssm",
        "--swag_samplemode",
        default="modelwise",
        choices=SWAG.sample_mode,
        help=f"specify at which level sampling will happen",
    )
    parser.add_argument(
        "-srr",
        "--swag_reducerank",
        type=int,
        help="if specified, limit rank of off-diagonal part",
    )
    parser.add_argument(
        "-srs",
        "--swag_reducestep",
        type=int,
        default=1,
        help="if reduce rank, step size for thinning",
    )
    parser.add_argument(
        "-sbu",
        "--swag_bnupdate",
        action="store_true",
        help="update BatchNorm for averaged model",
    )

    return parser.parse_args()


def get_swagloader(outclass: int, args):
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


def get_swaloaders(outclass: int, args):
    dataset = {v: k for k, v in OUTCLASS.items()}[outclass]
    # load data
    train_loader, data_loader = TRAINDATALOADERS[dataset](
        args.data_dir,
        args.tvsplit,
        args.workers,
        (device != torch.device("cpu")),
        args.batch,
        args.batch,
    )
    if not args.valdata:
        data_loader = TESTDATALOADER[dataset](
            args.data_dir,
            args.workers,
            (device != torch.device("cpu")),
            args.batch,
        )
    return train_loader, data_loader


def do_swagevalbatch(batchinput, models):
    inputs, gt = batchinput[:-1], batchinput[-1]
    cumloss = 0.0
    cumprob = torch.zeros([])
    nmodel = len(models)
    for model in models:
        output = model(*inputs)
        loss = nnf.nll_loss(nnf.log_softmax(output, 1), gt) / nmodel
        cumloss += loss.item()
        cumprob = cumprob + nnf.softmax(output, 1) / nmodel
    return cumprob, gt, cumloss


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
    dataset = "cifar10"
    ndata = (
        NTRAIN[dataset] - int(args.tvsplit * NTRAIN[dataset])
        if args.valdata
        else NTEST[dataset]
    )

    log_ece = coro_log_auroc(sw, args.printfreq, args.bins, args.save_dir)
    prefix = ""

    # iterate over all trained runs, assume model name best_model.pt
    for runfolder in sorted(
        [d for d in listdir(args.traindir) if isdir(pjoin(args.traindir, d))]
    ):
        model_path = pjoin(args.traindir, runfolder, "checkpoint.pt")
        if not exists(model_path):
            print(f"skipping {pjoin(args.traindir, runfolder)}\n")
            continue
        print(f"loading model from {model_path} ...\n")
        # resume model
        _, swagmodel, _, _, dic = loadcheckpoint(model_path, device)
        if args.swag_reducerank is not None:
            swagmodel.reduce_rank(args.swag_reducerank, args.swag_reducestep)
        outclass = dic["modelargs"][0]
        train_loader, swa_loader = get_swaloaders(outclass, args)
        swag_loader = get_swagloader(outclass, args)

        print(f">>> Test starts at {next(timer)[0].isoformat()} <<<\n")

        # do SWAG sampled model evaluation

        prefix = "val_bayes" if args.valdata else "test_bayes"

        if args.saveoutput:
            outputsaver = get_outputsaver(
                args.save_dir,
                ndata,
                outclass,
                f"predictions_{prefix}_{runfolder}.npy",
            )
        else:
            outputsaver = None

        log_ece.send((runfolder, prefix, len(swag_loader), outputsaver))
        with torch.no_grad():
            # sample models from swag
            sampledmodels = [
                swagmodel.sampled_model(mode=args.swag_samplemode)
                for _ in range(args.swag_modelsamples)
            ]
            # prepare them for evaluation
            for i, model in enumerate(sampledmodels):
                if args.swag_bnupdate:
                    print(
                        (
                            "updating BatchNorm for SWAG model sample "
                            f"{i+1}/{len(sampledmodels)} ..."
                        ),
                        end="",
                    )
                    bn_update(train_loader, model, device=device)
                    print(" Done.")
                model.eval()
            print()

            do_epoch(
                swag_loader,
                do_swagevalbatch,
                log_ece,
                device,
                models=sampledmodels,
            )

        bins, _, avgvloss = log_ece.throw(StopIteration)[:3]

        if args.saveoutput:
            outputsaver.close()

        del sampledmodels
        del swag_loader

        if args.plotdiagram:
            bins2diagram(
                bins,
                False,
                pjoin(args.save_dir, f"calibration_{prefix}_{runfolder}.pdf"),
            )

        print(f">>> Time elapsed: {next(timer)[1]} <<<\n")

        # do SWA evaluation

        prefix = "val_map" if args.valdata else "test_map"

        if args.saveoutput:
            outputsaver = get_outputsaver(
                args.save_dir,
                ndata,
                outclass,
                f"predictions_{prefix}_{runfolder}.npy",
            )
        else:
            outputsaver = None

        log_ece.send((runfolder, prefix, len(swa_loader), outputsaver))
        with torch.no_grad():
            swamodel = swagmodel.averaged_model()
            if args.swag_bnupdate:
                print("updating BatchNorm ...", end="")
                bn_update(train_loader, swamodel, device=device)
                print(" Done.")
            swamodel.eval()
            do_epoch(swa_loader, do_evalbatch, log_ece, device, model=swamodel)
        bins, _, avgvloss = log_ece.throw(StopIteration)[:3]

        if args.saveoutput:
            outputsaver.close()

        del swamodel

        if args.plotdiagram:
            bins2diagram(
                bins,
                False,
                pjoin(args.save_dir, f"calibration_{prefix}_{runfolder}.pdf"),
            )

        print(f">>> Time elapsed: {next(timer)[1]} <<<\n")

    log_ece.close()
    if prefix:
        print("- SWA results:")
        summarize_csv(pjoin(args.save_dir, f"{prefix.split('_')[0]}_map.csv"))
        print()
        print("- SWAG results:")
        summarize_csv(
            pjoin(args.save_dir, f"{prefix.split('_')[0]}_bayes.csv")
        )
        print()

    print(f">>> Test completed at {next(timer)[0].isoformat()} <<<\n")
