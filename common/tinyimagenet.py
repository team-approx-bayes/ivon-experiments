from typing import Optional, Callable
from os import rename, rmdir
from os.path import join as opjoin, exists as opexists
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.transforms import ToTensor
from .utils import mkdirp


class TinyImageNet(ImageFolder):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    dataset_folder = "tiny-imagenet-200"
    splits = ("train", "val", "test")

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        # using val split for evaluation
        split = "train" if train else "val"
        if download:
            self._download(root)
            self._process_val(opjoin(root, self.dataset_folder, "val"))

        super().__init__(
            opjoin(root, self.dataset_folder, split),
            transform=transform,
            target_transform=target_transform,
        )

    def _download(self, root) -> None:
        if opexists(opjoin(root, "tiny-imagenet-200.zip")):
            print("dataset already downloaded.")
            return
        mkdirp(root)
        download_and_extract_archive(self.url, download_root=root)

    @staticmethod
    def _process_val(val_dir) -> None:
        val_img_dir = opjoin(val_dir, "images")
        if not opexists(val_img_dir):
            return
        # read image labels
        with open(opjoin(val_dir, "val_annotations.txt"), "r") as fp:
            data = fp.readlines()
            val_img_dict = {}
            for line in data:
                words = line.strip().split("\t")
                val_img_dict[words[0]] = words[1]
        # sort images by labels
        for img, folder in val_img_dict.items():
            newpath = opjoin(val_dir, folder)
            mkdirp(newpath)
            if opexists(opjoin(val_img_dir, img)):
                rename(opjoin(val_img_dir, img), opjoin(newpath, img))
        # remove the empty original image dir
        rmdir(val_img_dir)


if __name__ == "__main__":

    def compute_stats(dataset):
        mean = 0
        std = 0
        for i, (img, _) in enumerate(dataset):
            if i % 1000 == 0:
                print(".", end="")
            mean += img.mean((1, 2))
            std += img.std((1, 2))
        print()
        mean /= len(dataset)
        std /= len(dataset)
        return mean.tolist(), std.tolist()

    trainset = TinyImageNet(
        "../datasets", True, transform=ToTensor(), download=True
    )
    print(f'Size {len(trainset)}')
    print("mean and std of training set:")
    print(compute_stats(trainset))
