# Distributed ImageNet training with IVON 

This covers the ImageNet experiments mentioned in Section 4.1.2 and Figure 1. 

## Download ImageNet and prepare in ffcv format 
First we need to download the ImageNet-1k dataset, for example, you can get it on [Hugging Face](https://huggingface.co/datasets/ILSVRC/imagenet-1k). Then, we need to create the ffcv datasets which enable fast dataloading: `python prepare_ffcv.py --imagenetdir [your_imagenet_directory] --ffcvdir [directory_of_ffcvdataset]`; 

## Train ImageNet with SGD, AdamW and IVON 
( ... )

## Test the trained models
( ... )
