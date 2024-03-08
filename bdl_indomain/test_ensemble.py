"""script to run deep ensemble cifar10/cifar100 classification test"""
from typing import List
import argparse
from os import listdir
from os.path import join as pjoin
import numpy as np
import torch
import sys

sys.path.insert(0, "..")
from common.utils import coro_timer, mkdirp, npybatchiterator
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
    get_outputsaver,
    summarize_csv,
)


def get_args():
    parser = argparse.ArgumentParser(
        description='CIFAR10/100 deep ensemble test')
    parser.add_argument('rootdir', type=str,
                        help='path that collects all predictions.')
    parser.add_argument('dataset', default='cifar10', choices=TRAINDATALOADERS,
                        help='datasets: ' + ' | '.join(TRAINDATALOADERS))
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('-b', '--batch', default=512, type=int,
                        metavar='N', help='test mini-batch size')
    parser.add_argument('-ec', '--ensemblecount', default=5, type=int,
                        help='number of deep ensembles')
    parser.add_argument('-es', '--ensemblesize', default=5, type=int,
                        help='number of models in each deep ensemble')
    parser.add_argument('--prefix', default='test')
    parser.add_argument('-sp', '--tvsplit', default=0.9, type=float,
                        metavar='RATIO',
                        help='ratio of data used for training')
    parser.add_argument('-pf', '--printfreq', default=10, type=int,
                        metavar='N', help='print frequency')
    parser.add_argument('-d', '--device', default='cpu', type=str,
                        metavar='DEV', help='run on cpu/cuda')
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='fixes seed for reproducibility')
    parser.add_argument('-sd', '--save_dir',
                        help='The directory used to save test results',
                        default='save_temp', type=str)
    parser.add_argument('-so', '--saveoutput', action='store_true',
                        help='save output probability')
    parser.add_argument('-dd', '--data_dir',
                        help='The directory to find/store dataset',
                        default='../data', type=str)
    parser.add_argument('-nb', '--bins', default=20, type=int,
                        help='number of bins for ece & reliability diagram')
    parser.add_argument('-pd', '--plotdiagram', action='store_true',
                        help='plot reliability diagram for best val')
    parser.add_argument('-tbd', '--tensorboard_dir', default='', type=str,
                        help='if specified, record data for tensorboard.')

    return parser.parse_args()


def predfiles_per_model(args):
    # collect all prediction files
    if args.prefix.startswith("val"):
        predfiles = sorted([
            pjoin(args.rootdir, f) for f in listdir(args.rootdir)
            if f.endswith('.npy') and f.startswith('predictions_val')])
    else:
        predfiles = sorted([
            pjoin(args.rootdir, f) for f in listdir(args.rootdir)
            if f.endswith('.npy') and f.startswith('predictions_test')])
    # deliver per model
    ec, es = args.ensemblecount, args.ensemblesize
    assert len(predfiles) >= ec * es
    for c in range(ec):
        yield predfiles[c*es:(c+1)*es]


def get_predictionloader(args, predfilenames: List[str]):
    dataset = args.dataset
    device = torch.device(args.device)
    # load data
    if args.prefix.startswith("val"):
        _, data_loader = TRAINDATALOADERS[dataset](
            args.data_dir, args.tvsplit, args.workers,
            (device != torch.device('cpu')), args.batch, args.batch)
    else:
        data_loader = TESTDATALOADER[dataset](
            args.data_dir, args.workers, (device != torch.device('cpu')),
            args.batch)
    gts = (gt for _, gt in data_loader)
    prediters = [npybatchiterator(f, args.batch) for f in predfilenames]
    yield len(data_loader)
    yield from zip(*prediters, gts)


def do_devalbatch(batchinput):
    preds, gt = batchinput[:-1], batchinput[-1]
    meanpred = torch.from_numpy(np.mean(np.stack(preds, 0), 0)).to(gt.device)
    return meanpred, gt, 0.0


if __name__ == '__main__':
    timer = coro_timer()
    t_init = next(timer)
    print(f'>>> Test initiated at {t_init.isoformat()} <<<\n')

    args = get_args()
    print(args, end='\n\n')

    # if seed is specified, run deterministically
    if args.seed is not None:
        deteministic_run(seed=args.seed)

    # get device for this experiment
    device = torch.device(args.device)

    if device != torch.device('cpu'):
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
    ndata = NTRAIN[args.dataset]-int(args.tvsplit*NTRAIN[args.dataset]) \
        if (args.prefix.startswith("val")) else NTEST[args.dataset]

    log_ece = coro_log_auroc(sw, args.printfreq, args.bins, args.save_dir)
    outclass = OUTCLASS[args.dataset]
    prefix = args.prefix

    # iterate over saved predictions per each deep ensemble
    for modelid, predfiles in enumerate(predfiles_per_model(args)):
        print(f'ensembling from following {args.ensemblesize} files:')
        for f in predfiles:
            print(f'- {f}')

        print(f'>>> Test starts at {next(timer)[0].isoformat()} <<<\n')

        # do deep ensemble evaluation

        predloader = get_predictionloader(args, predfiles)
        nbatch = next(predloader)

        if args.saveoutput:
            outputsaver = get_outputsaver(
                args.save_dir, ndata, outclass,
                f'predictions_{prefix}_{modelid}.npy')
        else:
            outputsaver = None

        log_ece.send((modelid, prefix, nbatch, outputsaver))
        with torch.no_grad():
            do_epoch(predloader, do_devalbatch, log_ece, device)

        bins, _, avgvloss = log_ece.throw(StopIteration)[:3]

        if args.saveoutput:
            outputsaver.close()

        if args.plotdiagram:
            bins2diagram(
                bins, False,
                pjoin(args.save_dir, f'calibration_{prefix}_{modelid}.pdf'))

        print(f'>>> Time elapsed: {next(timer)[1]} <<<\n')

    log_ece.close()

    if prefix:
        print('- results:')
        summarize_csv(pjoin(args.save_dir, f'{prefix}.csv'))
        print()

    print(f'>>> Test completed at {next(timer)[0].isoformat()} <<<\n')
