# Distributed ImageNet training with IVON 
This covers the ImageNet experiments mentioned in Section 4.1.2 and Figure 1. 

## Download ImageNet and prepare in ffcv format 
First we need to download the ImageNet-1k dataset, for example, you can get it on [Hugging Face](https://huggingface.co/datasets/ILSVRC/imagenet-1k). Then, create the ffcv datasets which enables quick dataloading: `python prepare_ffcv.py <imagenet_directory> <target_directory>`; 

## Train ImageNet with SGD, AdamW and IVON 
The script supports single node training with `<ngpus>` GPUs. Set `<datadir>` to the target directory where the FFCV dataset is stored. Select `<batchsize>` (we used batchsize 1024 in the paper), random seed `seed` (we used 0-4) and number of epochs `<epochs>` and then run: 
- SGD: `bash train_sgd.sh <datadir> <ngpus> <batchsize> <seed> <epochs>`
- AdamW: `bash train_adamw.sh <datadir> <ngpus> <batchsize> <seed> <epochs>`
- IVON: `bash train_ivon.sh <datadir> <ngpus> <batchsize> <seed> <epochs>`

## Test the trained models
To test the ImageNet models trained for `<epochs>` epochs, run: 
- SGD: `bash test_sgd.sh <datadir> <epochs>`
- AdamW: `bash test_adamw.sh <datadir> <epochs>`
- IVON (mean): `bash test_ivon_mean.sh <datadir> <epochs>`
- IVON (Bayes): `bash test_ivon_bayes.sh <datadir> <epochs>`
