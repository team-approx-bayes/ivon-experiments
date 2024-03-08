from typing import Iterable, Tuple, List, Optional
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("pdf")  # for remote machines without GUI
from .utils import autoinitcoroutine


BINSTYPE = Tuple[List[int], List[int], List[float]]


def bins2ece(bins: BINSTYPE) -> float:
    num = sum(bins[0])
    ece = 0.0
    for corr, cconf in zip(bins[1], bins[2]):
        ece += abs(corr - cconf) / num
    return ece


def bins2acc(bins: BINSTYPE) -> float:
    return float(sum(bins[1])) / sum(bins[0])


def bins2conf(bins: BINSTYPE) -> float:
    return sum(bins[2]) / sum(bins[0])


# rightconfs should be an iterable of pairs that indicate if each prediction is
# correct and how confident the prediction is.
def data2bins(
    rightconfs: Iterable[Tuple[bool, float]], nbin: int = 10
) -> BINSTYPE:
    bincounts = [0] * nbin
    corrects = [0] * nbin
    cumconf = [0.0] * nbin

    for isright, conf in rightconfs:
        b = max(0, min(nbin - 1, int(conf * nbin)))
        bincounts[b] += 1
        corrects[b] += isright
        cumconf[b] += conf

    return bincounts, corrects, cumconf


@autoinitcoroutine
def coro_binsmerger():
    bins = None
    try:
        bins = yield
        while True:
            nbins = yield bins
            bins = joinbins(bins, nbins)
    except StopIteration:
        return bins


def joinbins(*binses):
    lb = len(binses[0][0])
    for bins in binses:
        if len(bins[0]) != lb:
            raise ValueError("cannot join bins with different lengths")
    bincountses, correctses, cumconfs = tuple(zip(*binses))
    return (
        [sum(cs) for cs in zip(*bincountses)],
        [sum(cs) for cs in zip(*correctses)],
        [sum(cs) for cs in zip(*cumconfs)],
    )


# plot and save reliability diagram
def bins2diagram(
    bins: BINSTYPE, displays: bool = False, saveas: Optional[str] = None
) -> None:
    nbin = len(bins[0])
    binvals = [float(i) / nbin for i in range(nbin + 1)]
    accconfs = [
        (float(corr) / bc, cconf / bc) if bc > 0 else (0.0, 0.0)
        for bc, corr, cconf in zip(*bins)
    ]
    weights = (
        [acc for acc, _ in accconfs],
        [conf - acc for acc, conf in accconfs],
    )
    fig = plt.figure(figsize=(5, 5))
    a1 = fig.add_subplot(111)
    a2 = a1.twinx()
    a1.set_xlim(0, 1)
    a1.set_ylim(0, 1)
    a1.set_xlabel("Confidence")
    a1.set_ylabel("Accuracy")
    _, _, ps = a1.hist(
        [binvals[:-1], binvals[:-1]],
        binvals,
        weights=weights,
        color=[(0, 0, 1, 1), (1, 0, 0, 0.5)],
        label=("Empirical", "Gap"),
        stacked=True,
    )
    total = sum(bins[0])
    freqs = [float(c) / total for c in bins[0]]
    a2.set_ylim(0, 1)
    a2.set_ylabel("Frequency")
    a2.hist(
        [binvals[:-1]],
        binvals,
        weights=[freqs],
        rwidth=0.3,
        color=[(0.5, 0.5, 0.5, 1)],
        label=("Frequency",),
    )
    fig.set_tight_layout(True)
    fig.legend(loc="upper left", bbox_to_anchor=a1.get_position())

    hatches = ["", "/"]
    linewidths = [1.0, 1.0]
    edgecolors = [(0, 0, 0.5, 1), (1, 0, 0, 1)]
    for pset, h, lw, ec in zip(ps, hatches, linewidths, edgecolors):
        for patch in pset.patches:
            patch.set_hatch(h)
            patch.set_lw(lw)
            patch.set_edgecolor(ec)
    if saveas:
        fig.savefig(saveas)
    if displays:
        fig.show()
    plt.close(fig)
