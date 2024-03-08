import os
import pathlib
import shutil
from urllib.request import urlretrieve
import zipfile
import csv
from datetime import datetime
from itertools import zip_longest
import numpy as np


def autoinitcoroutine(coro):
    def initcoro(*args, **kwargs):
        cr = coro(*args, **kwargs)
        next(cr)
        return cr

    return initcoro


def coro_timer():
    now = datetime.now()
    yield now
    while True:
        now, past = datetime.now(), now
        yield (now, now - past)


@autoinitcoroutine
def coro_trackavg():
    total, count = 0.0, 0
    try:
        num = yield
        while True:
            total, count = total + num, count + 1
            num = yield total / count
    except StopIteration:
        return 0.0 if count == 0 else total / count


def div0(a, b):
    return 0.0 if b == 0 else a / b


@autoinitcoroutine
def coro_trackavg_weighted():
    total, totalweights = 0.0, 0.0
    try:
        val, weight = yield
        while True:
            total, totalweights = total + val, totalweights + weight
            val, weight = yield div0(total, totalweights)
    except StopIteration:
        return div0(total, totalweights)


@autoinitcoroutine
def coro_dict2csv(to, fieldnames, append: bool = False, **kwargs):
    with open(to, "a+" if append else "w", buffering=1) as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, **kwargs)
        if not append:
            writer.writeheader()
        d = yield
        while True:
            writer.writerow(d)
            d = yield


def rm(filename: str) -> bool:
    try:
        os.remove(filename)
        return True
    except OSError:
        return False


def mkdir(dirpath: str, parents: bool = False, exist_ok: bool = False) -> None:
    pathlib.Path(dirpath).mkdir(parents=parents, exist_ok=exist_ok)


def mkdirp(dirpath: str) -> None:
    mkdir(dirpath, parents=True, exist_ok=True)


def cp(source: str, destination: str) -> None:
    shutil.copy(source, destination)


def download(fromurl: str, saveas: str) -> None:
    print(f"downloading {fromurl} to {saveas} ...")
    urlretrieve(fromurl, saveas)
    print("done.")


def unzip(zipped: str, unzipto: str = ".") -> None:
    print(f"unzipping {zipped} to {unzipto} ...")
    with zipfile.ZipFile(zipped, "r") as zipfp:
        zipfp.extractall(unzipto)
    print("done.")


def asnpbatchiter(entryiter, batchsize: int, droplast: bool = False):
    if droplast:
        yield from (np.stack(b) for b in zip(*[entryiter] * batchsize))
    else:
        yield from (
            np.concatenate(b).reshape(-1, *b[0].shape)
            for b in zip_longest(
                *[entryiter] * batchsize, fillvalue=np.array(())
            )
        )


# low ram usage minibatch gatherer on disk
# peak data ram usage: input batchsize * entry_nelem * sizeof(dtype)
@autoinitcoroutine
def coro_npybatchgatherer(
    filepath, entrycount: int, entryshape=(), overwrites=False, dtype="float"
):
    if (not overwrites) and os.path.exists(filepath):
        raise FileExistsError(f"{filepath} already exists.")
    # create/overwrite on disk
    array = np.empty((entrycount, *entryshape), dtype)
    np.save(filepath, array)
    # fill in gathered values
    pos = 0
    try:
        batch = yield
        while pos < entrycount:
            ll = len(batch)
            mm = np.load(filepath, mmap_mode="r+")
            mm[pos : pos + ll] = batch
            pos += ll
            batch = yield
    except StopIteration:  # return effective written entrycount
        return pos


# limited ram iterative decoder for gathered npy file
# peak data ram usage: cache * entry_nelem * sizeof(dtype)
def npyiterator(filepath, transform=None, cache=1000000):
    ll = np.load(filepath, mmap_mode="r").shape[0]
    pos = 0
    while pos * cache <= ll:
        chunk = np.load(filepath, mmap_mode="r")[
            pos * cache : (pos + 1) * cache
        ]
        if transform is None:
            yield from chunk
        else:
            yield from (transform(d) for d in chunk)
        pos += 1


# limited ram iterative decoder for gathered npy file in batches
# peak data ram usage: cache * entry_nelem * sizeof(dtype)
def npybatchiterator(
    filepath,
    batchsize: int,
    droplast: bool = False,
    transform=None,
    cache=1000000,
):
    yield from asnpbatchiter(
        npyiterator(filepath, transform, cache), batchsize, droplast
    )
