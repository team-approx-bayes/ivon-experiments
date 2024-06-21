import os 
import argparse 
import ffcv
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField
from torchvision.datasets import ImageFolder
import sys
sys.path.append("..")
from common.utils import mkdirp

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'imagenetdir', type=str, help='Path to ImageNet dataset')
    parser.add_argument(
        'ffcvdir', type=str, help='Save directory for ffcv format of ImageNet')
    args = parser.parse_args()

    imagenet_dir = args.imagenetdir 
    imagenet_traindir = os.path.join(imagenet_dir, 'train')
    imagenet_valdir = os.path.join(imagenet_dir, 'val')

    train_dataset = ImageFolder(imagenet_traindir)
    val_dataset = ImageFolder(imagenet_valdir)

    def write_dataset(write_path, dataset):
        writer = DatasetWriter(write_path, {
            'image': RGBImageField(write_mode='proportion',
                                max_resolution=500,
                                compress_probability=0.50,
                                jpeg_quality=90),
            'label': IntField(),
        }, num_workers=16)

        writer.from_indexed_dataset(dataset, chunksize=100)

    mkdirp(args.ffcvdir)
    write_dataset(os.path.join(args.ffcvdir, 'train.ffcv'), train_dataset)
    write_dataset(os.path.join(args.ffcvdir, 'val.ffcv'), val_dataset)

if __name__ == '__main__':
    main()
