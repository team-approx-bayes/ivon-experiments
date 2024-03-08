"""CIFAR-10-C and CIFAR-100-C datasets: https://arxiv.org/abs/1903.12261"""
from typing import Optional, Union, List, Iterable, Dict
from os import listdir, rename, rmdir
from os.path import join as pjoin, exists as pexists
from collections import OrderedDict
import numpy as np
import numpy.lib.format as npformat
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive
import torchvision.transforms as transforms
from .utils import mkdirp


CORRUPTIONS = OrderedDict(
    (
        ("gaussian_noise", "noise"),
        ("shot_noise", "noise"),
        ("impulse_noise", "noise"),
        ("defocus_blur", "blur"),
        ("glass_blur", "blur"),
        ("motion_blur", "blur"),
        ("zoom_blur", "blur"),
        ("snow", "weather"),
        ("frost", "weather"),
        ("fog", "weather"),
        ("brightness", "weather"),
        ("contrast", "digital"),
        ("elastic_transform", "digital"),
        ("pixelate", "digital"),
        ("jpeg_compression", "digital"),
    )
)  # type: OrderedDict[str, str]
CORRUPTIONS_TYPES = OrderedDict(
    (
        ("noise", ("gaussian_noise", "shot_noise", "impulse_noise")),
        ("blur", ("defocus_blur", "glass_blur", "motion_blur", "zoom_blur")),
        ("weather", ("snow", "frost", "fog", "brightness")),
        (
            "digital",
            ("contrast", "elastic_transform", "pixelate", "jpeg_compression"),
        ),
    )
)
CORRUPTIONS_EXTRA = OrderedDict(
    (
        ("speckle_noise", "noise"),
        ("gaussian_blur", "blur"),
        ("spatter", "weather"),
        ("saturate", "digital"),
    )
)
CORRUPTIONS_TYPES_EXTRA = OrderedDict(
    (
        ("noise", "speckle_noise"),
        ("blur", "gaussian_blur"),
        ("weather", "spatter"),
        ("digital", "saturate"),
    )
)
CORRUPTIONS_ALL = OrderedDict(
    list(CORRUPTIONS.items()) + list(CORRUPTIONS_EXTRA.items())
)
CORRUPTIONS_TYPES_ALL = OrderedDict(
    (
        (
            "noise",
            ("gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise"),
        ),
        (
            "blur",
            (
                "defocus_blur",
                "glass_blur",
                "motion_blur",
                "zoom_blur",
                "gaussian_blur",
            ),
        ),
        ("weather", ("snow", "frost", "fog", "brightness", "spatter")),
        (
            "digital",
            (
                "contrast",
                "elastic_transform",
                "pixelate",
                "jpeg_compression",
                "saturate",
            ),
        ),
    )
)


SEVERITY_LEVELS = (1, 2, 3, 4, 5)


CORRUPTION_CHOICES = {
    "main": tuple(CORRUPTIONS),
    "extra": tuple(CORRUPTIONS_EXTRA),
    "all": tuple(CORRUPTIONS_ALL),
    **{f"{t}_main": cs for t, cs in CORRUPTIONS_TYPES.items()},
    **{f"{t}_extra": cs for t, cs in CORRUPTIONS_TYPES_EXTRA.items()},
    **{f"{t}_all": cs for t, cs in CORRUPTIONS_TYPES_ALL.items()},
    **{c: c for c in CORRUPTIONS_ALL},
}


# number of classes
OUTCLASS = {"cifar10": 10, "cifar100": 100}


def parse_npy_header(path):
    with open(path, "rb") as fp:
        version = npformat.read_magic(fp)
        npformat._check_version(version)
        shape, fortran_order, dtype = npformat._read_array_header(fp, version)
        return shape, ("C", "F")[fortran_order], dtype, fp.tell()


def _download_cifar10c(root) -> None:
    url = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1"
    filname = "CIFAR-10-C.tar"
    if pexists(pjoin(root, filname)):
        print("dataset already downloaded.")
        return
    mkdirp(root)
    download_and_extract_archive(url, download_root=root, filename=filname)
    npy_dir = pjoin(root, "CIFAR-10-C")
    for npy in listdir(npy_dir):
        rename(pjoin(npy_dir, npy), pjoin(root, npy))
    rmdir(npy_dir)


class CIFAR10C(VisionDataset):
    def __init__(
        self,
        root,
        corruption: str,
        severity: Optional[int] = None,
        download: bool = False,
        transform=None,
        target_transform=None,
    ):
        super().__init__(
            root, transform=transform, target_transform=target_transform
        )
        if corruption not in CORRUPTIONS_ALL:
            raise RuntimeError(
                f"corruption must be one of {tuple(CORRUPTIONS_ALL)},"
                f" found {corruption}"
            )
        if severity not in (None,) + SEVERITY_LEVELS:
            raise RuntimeError(
                f"severity can only range from 1~5, found {severity}"
            )
        if download:
            _download_cifar10c(root)
        self._dpath = ""
        self._doff = self._check_and_get_offset(root, corruption, severity)
        self._dshape = (
            (50000, 32, 32, 3) if severity is None else (10000, 32, 32, 3)
        )
        self.corruption = corruption
        self.severity = severity
        self.targets = np.load(pjoin(root, "labels.npy"))
        if severity is not None:
            self.targets = self.targets[
                10000 * (severity - 1) : 10000 * severity
            ]

    def _download(self, root) -> None:
        if pexists(pjoin(root, "CIFAR-10-C.tar")):
            print("dataset already downloaded.")
            return
        mkdirp(root)
        download_and_extract_archive(self.url, download_root=root)
        npy_dir = pjoin(root, "CIFAR-10-C")
        for npy in listdir(npy_dir):
            rename(pjoin(npy_dir, npy), pjoin(root, npy))
        # remove the empty original numpy array dir
        rmdir(npy_dir)

    def _check_and_get_offset(self, root, corruption, severity):
        assert parse_npy_header(pjoin(root, f"{corruption}.npy")) == (
            (50000, 32, 32, 3),
            "C",
            np.uint8,
            128,
        )
        assert parse_npy_header(pjoin(root, "labels.npy")) == (
            (50000,),
            "C",
            np.uint8,
            128,
        )
        self._dpath = pjoin(root, f"{corruption}.npy")
        return (
            128
            if severity is None
            else 128 + (severity - 1) * 10000 * 32 * 32 * 3
        )

    def _get_data(self, index: Union[int, List[int]]):
        data = np.frombuffer(
            np.memmap(
                self._dpath,
                dtype=np.uint8,
                mode="r",
                shape=self._dshape,
                offset=self._doff,
            )[index],
            dtype=np.uint8,
        )
        if isinstance(index, int):
            img = Image.fromarray(data.reshape(32, 32, 3))
            if self.transform is None:
                return img
            else:
                return self.transform(img)
        else:  # batch process, assume transforms to torch.Tensor
            self_transform = self.transform
            imgfromarray = Image.fromarray
            return torch.stack(
                [
                    self_transform(imgfromarray(d))
                    for d in data.reshape(-1, 32, 32, 3)
                ],
                dim=0,
            )

    def _get_target(self, index: Union[int, List[int]]):
        target = self.targets[index]
        if isinstance(index, int):
            if self.target_transform is not None:
                target = self.target_transform(target)
            return target
        else:
            if self.target_transform is not None:
                self_target_transform = self.target_transform
                return torch.as_tensor(
                    [self_target_transform(t) for t in target],
                    dtype=torch.long,
                )
            else:
                return torch.as_tensor(target, dtype=torch.long)

    # use list when possible to minimize disk I/O
    def __getitem__(self, index: Union[int, List[int]]):
        return self._get_data(index), self._get_target(index)

    def __len__(self):
        return 50000 if self.severity is None else 10000


class MultiCIFAR10C(VisionDataset):
    def __init__(
        self,
        root,
        corruptions: Iterable[str],
        severity: Optional[int] = None,
        download: bool = False,
        transform=None,
        target_transform=None,
    ):
        super().__init__(
            root, transform=transform, target_transform=target_transform
        )
        self.corruptions = list(corruptions)
        if not set(self.corruptions).issubset(set(CORRUPTIONS_ALL)):
            raise RuntimeError(
                f"corruptions must be chosen from {tuple(CORRUPTIONS_ALL)}."
            )
        if severity not in (None,) + SEVERITY_LEVELS:
            raise RuntimeError(
                f"severity can only range from 1~5, found {severity}"
            )
        if download:
            _download_cifar10c(root)
        self._dpaths = []  # type: List[str]
        self._doff = self._check_and_get_offset(root, severity)
        self._dshape = (
            (50000, 32, 32, 3) if severity is None else (10000, 32, 32, 3)
        )

        self.severity = severity
        self.targets = np.load(pjoin(root, "labels.npy"))
        if severity is not None:
            self.targets = self.targets[
                10000 * (severity - 1) : 10000 * severity
            ]
        self._ccount = len(self.corruptions)
        self._eachlen = 50000 if self.severity is None else 10000

    def _check_and_get_offset(self, root, severity):
        assert parse_npy_header(pjoin(root, "labels.npy")) == (
            (50000,),
            "C",
            np.uint8,
            128,
        )
        for corruption in self.corruptions:
            assert parse_npy_header(pjoin(root, f"{corruption}.npy")) == (
                (50000, 32, 32, 3),
                "C",
                np.uint8,
                128,
            )
            self._dpaths.append(pjoin(root, f"{corruption}.npy"))
        return (
            128
            if severity is None
            else 128 + (severity - 1) * 10000 * 32 * 32 * 3
        )

    def _get_data(self, dindex: int, index: Union[int, List[int]]):
        dpath = self._dpaths[dindex]
        data = np.frombuffer(
            np.memmap(
                dpath,
                dtype=np.uint8,
                mode="r",
                shape=self._dshape,
                offset=self._doff,
            )[index],
            dtype=np.uint8,
        )
        if isinstance(index, int):  # assume transforms to torch.Tensor
            return self.transform(Image.fromarray(data.reshape(32, 32, 3)))
        else:  # batch process, assume transforms to torch.Tensor
            self_transform = self.transform
            imgfromarray = Image.fromarray
            return torch.stack(
                [
                    self_transform(imgfromarray(d))
                    for d in data.reshape(-1, 32, 32, 3)
                ],
                dim=0,
            )

    def _get_multidata(self, lidx: List[int]):
        di = dict()  # type: Dict[int, List[int]]
        de = dict()  # type: Dict[int, List[int]]
        _parse_index = self._parse_index
        for e, i in enumerate(lidx):
            idat, ipos = _parse_index(i)
            di[idat] = di.get(idat, []) + [ipos]
            de[idat] = de.get(idat, []) + [e]
        _get_data = self._get_data
        multidata = [(de[d], _get_data(d, li)) for d, li in di.items()]
        _d = multidata[0][1]
        out = torch.empty(
            (len(lidx), *_d.size()[1:]), dtype=_d.dtype, device=_d.device
        )
        for es, data in multidata:
            out[es] = data
        return out

    def _get_target(self, index: Union[int, List[int]]):
        target = self.targets[index]
        if isinstance(index, int):
            if self.target_transform is not None:
                target = self.target_transform(target)
            return target
        else:
            self_target_transform = self.target_transform
            if self_target_transform is not None:
                return torch.as_tensor(
                    [self_target_transform(t) for t in target],
                    dtype=torch.long,
                )
            else:
                return torch.as_tensor(target, dtype=torch.long)

    def _parse_index(self, idx: int):
        _eachlen = self._eachlen
        return idx // _eachlen, idx % _eachlen

    # use list when possible to minimize disk I/O
    def __getitem__(self, idx: Union[int, List[int]]):
        if isinstance(idx, int):
            didx, iidx = self._parse_index(idx)
            return self._get_data(didx, iidx), self._get_target(iidx)
        else:
            _eachlen = self._eachlen
            return self._get_multidata(idx), self._get_target(
                [i % _eachlen for i in idx]
            )

    def __len__(self):
        return self._eachlen * self._ccount


# manual collate function for dataloader to duplicate batch
def dup_collate_fn(dups: int):
    # assume a list of img, result pairs
    def collate_fn(data):
        imgs, gt = data
        return imgs.repeat(dups, *(1,) * (imgs.ndim - 1)), gt

    return collate_fn


def get_cifar10c_loader(
    data_dir: str,
    workers: int,
    pin_memory: bool,
    batch: int,
    dups: int = 1,
    corruption: Union[str, Iterable[str]] = tuple(CORRUPTIONS),
    severity: Optional[int] = None,
    download: bool = True,
) -> DataLoader:
    # statistics from normal cifar10 dataset
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    if isinstance(corruption, str):
        dataset = CIFAR10C(
            pjoin(data_dir, "cifar10c"),
            corruption,
            severity,
            download=download,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )
    else:
        dataset = MultiCIFAR10C(
            pjoin(data_dir, "cifar10c"),
            corruption,
            severity,
            download=download,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )

    loader = (
        DataLoader(
            dataset,
            batch_size=None,
            sampler=BatchSampler(
                SequentialSampler(dataset), batch_size=batch, drop_last=False
            ),
            num_workers=workers,
            pin_memory=pin_memory,
            collate_fn=dup_collate_fn(dups),
        )
        if dups > 1
        else DataLoader(
            dataset,
            batch_size=None,
            sampler=BatchSampler(
                SequentialSampler(dataset), batch_size=batch, drop_last=False
            ),
            num_workers=workers,
            pin_memory=pin_memory,
        )
    )

    return loader
