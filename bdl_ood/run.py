import argparse
from os.path import join as pjoin, exists
import torch
from ivon import IVON
import sys

sys.path.append("..")
from common.vogn import VOGN
from common.swag import SWAG
from common.utils import coro_timer, mkdirp, coro_dict2csv
from common.trainutils import (
    do_epoch,
    check_cuda,
    deteministic_run,
    loadcheckpoint,
)
from common.dataloaders import get_cifar10_test_loader
from common.ood_utils import (
    get_svhn_loader,
    get_flowers102_loader,
    do_evalbatch,
    do_evalbatch_von,
    do_evalbatch_swag,
    auroc,
    get_outputsaver,
    summarize_csv,
    coro_log,
    SVHNInfo,
    Flowers102Info,
    confidence_from_prediction_npy,
    OODMetrics,
)


OODInfo = {
    'svhn': SVHNInfo,
    'flowers102': Flowers102Info,
}


def get_in_out_confs(test_folder: str, idx: int, starts_with="predictions"):
    in_name = pjoin(test_folder, f"{starts_with}_indomain_test_{idx}.npy")
    out_name = pjoin(test_folder, f"{starts_with}_ood_test_{idx}.npy")
    in_conf = confidence_from_prediction_npy(in_name)
    out_conf = confidence_from_prediction_npy(out_name)
    return in_conf, out_conf


def compute_and_save_metrics(test_folder: str, wamode: str = "", runs=()):
    starts_with = "predictions" if not wamode else f"predictions_{wamode}"
    csv_name = (
        "metrics_test.csv" if not wamode else f"metrics_{wamode}_test.csv"
    )
    csvcorolog = coro_dict2csv(
        pjoin(test_folder, csv_name), ("epoch",) + OODMetrics.metric_names
    )
    for e in runs:
        metrics = OODMetrics(
            *get_in_out_confs(test_folder, e, starts_with)
        ).get_all()
        print(
            ", ".join(
                [f"epoch: {e}"] + [f"{k}: {v:.4f}" for k, v in metrics.items()]
            )
        )
        csvcorolog.send({"epoch": e, **metrics})


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "traindir", type=str, help="path that collects all trained runs."
    )
    parser.add_argument(
        "--ood_dataset",
        default="svhn",
        choices=OODInfo,
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
        "-ts",
        "--testsamples",
        default=1,
        type=int,
        help="create test samples via duplicating batch",
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

    return parser.parse_args()


def get_ood_loader(args):
    if args.ood_dataset == 'svhn':
        return get_svhn_loader(
            args.data_dir,
            args.workers,
            (args.device != "cpu"),
            args.batch,
            'test',
            args.testsamples,
        )
    elif args.ood_dataset == 'flowers102':
        return get_flowers102_loader(
            args.data_dir,
            args.workers,
            (args.device != "cpu"),
            args.batch,
            'test',
            args.testsamples,
        )


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
    log_ece = coro_log(None, args.printfreq, args.save_dir)

    # iterate over all trained runs, assume model name best_model.pt
    indomain_prefix = "indomain_test"
    indomain_loader = get_cifar10_test_loader(
        args.data_dir,
        args.workers,
        (device != torch.device("cpu")),
        args.batch,
        args.testsamples,
    )
    ood_prefix = "ood_test"
    ood_loader = get_ood_loader(args)
    aucroc_scores = []

    runs = sorted([str(i) for i in range(5)])
    valid_runs = []

    for runfolder in runs:
        model_path = pjoin(args.traindir, runfolder, "checkpoint.pt")
        if not exists(model_path):
            print(f"skipping {pjoin(args.traindir, runfolder)}\n")
            continue
        else:
            valid_runs.append(runfolder)

        # resume model
        _, model, optimizer, _, ddat = loadcheckpoint(model_path, device)
        optimizer.mc_samples = args.testrepeat
        outclass = 10

        print(f">>> Test starts at {next(timer)[0].isoformat()} <<<\n")

        # In-domain test run
        if args.saveoutput:
            outputsaver = get_outputsaver(
                args.save_dir,
                10000,
                outclass,
                f"predictions_{indomain_prefix}_{runfolder}.npy",
            )
        else:
            outputsaver = None

        log_ece.send(
            (runfolder, indomain_prefix, len(indomain_loader), outputsaver)
        )
        with torch.no_grad():
            if isinstance(optimizer, IVON) or isinstance(optimizer, VOGN):
                model.eval()
                do_epoch(
                    indomain_loader,
                    do_evalbatch_von,
                    log_ece,
                    device,
                    model=model,
                    optimizer=optimizer,
                    repeat=args.testrepeat,
                )
            elif isinstance(model, SWAG):
                sampledmodels = [
                    model.sampled_model(mode=args.swag_samplemode)
                    for _ in range(args.swag_modelsamples)
                ]
                for m in sampledmodels:
                    m.eval()
                do_epoch(
                    indomain_loader,
                    do_evalbatch_swag,
                    log_ece,
                    device,
                    models=sampledmodels,
                )
            else:
                model.eval()
                do_epoch(
                    indomain_loader,
                    do_evalbatch,
                    log_ece,
                    device,
                    model=model,
                    dups=args.testsamples,
                    repeat=args.testrepeat,
                )
        log_ece.throw(StopIteration)
        if args.saveoutput:
            outputsaver.close()

        # OOD test run
        if args.saveoutput:
            outputsaver = get_outputsaver(
                args.save_dir,
                OODInfo[args.ood_dataset].count["test"],
                outclass,
                f"predictions_{ood_prefix}_{runfolder}.npy",
            )
        else:
            outputsaver = None

        log_ece.send((runfolder, ood_prefix, len(ood_loader), outputsaver))
        with torch.no_grad():
            if isinstance(optimizer, IVON) or isinstance(optimizer, VOGN):
                model.eval()
                do_epoch(
                    ood_loader,
                    do_evalbatch_von,
                    log_ece,
                    device,
                    model=model,
                    optimizer=optimizer,
                    repeat=args.testrepeat,
                )
            elif isinstance(model, SWAG):
                sampledmodels = [
                    model.sampled_model(mode=args.swag_samplemode)
                    for _ in range(args.swag_modelsamples)
                ]
                for m in sampledmodels:
                    m.eval()
                do_epoch(
                    ood_loader,
                    do_evalbatch_swag,
                    log_ece,
                    device,
                    models=sampledmodels,
                )
            else:
                model.eval()
                do_epoch(
                    ood_loader,
                    do_evalbatch,
                    log_ece,
                    device,
                    model=model,
                    dups=args.testsamples,
                    repeat=args.testrepeat,
                )
        log_ece.throw(StopIteration)
        if args.saveoutput:
            outputsaver.close()
        del model

        indomain_conf = confidence_from_prediction_npy(
            pjoin(
                args.save_dir, f"predictions_{indomain_prefix}_{runfolder}.npy"
            )
        )
        ood_conf = confidence_from_prediction_npy(
            pjoin(args.save_dir, f"predictions_{ood_prefix}_{runfolder}.npy")
        )
        aucroc = auroc(indomain_conf, ood_conf)
        print(f"AUC-ROC score: {aucroc}")
        aucroc_scores.append(aucroc)

        print(f">>> Time elapsed: {next(timer)[1]} <<<\n")

    # print(f'{indomain_prefix}:')
    # summarize_csv(pjoin(args.save_dir, f'{indomain_prefix}.csv'))
    # print(f'\n{ood_prefix}:')
    # summarize_csv(pjoin(args.save_dir, f'{ood_prefix}.csv'))
    # mean, std = mean_std(aucroc_scores)
    # print(f'\nAUC-ROC score:\tmean {mean:.4f}, std={std:.4f} \n')
    compute_and_save_metrics(args.save_dir, "", valid_runs)
    summarize_csv(pjoin(args.save_dir, "metrics_test.csv"))

    print(f">>> Test completed at {next(timer)[0].isoformat()} <<<\n")

    log_ece.close()
