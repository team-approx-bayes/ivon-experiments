# Image Classification (Single GPU)

Image classification experiments in Section 4.1.2 are split into two parts, here we cover single GPU experiments, i.e. everything except ImageNet experiments.

## Run training

Select `<dataset>` from "cifar10"/"cifar100"/"tinyimagenet",  `<model>` from "resnet20"/"densenet121"/"resnet18wide"/"preresnet110" and `seed` from 0-4:

- SGD: `bash train_sgd.sh <dataset> <model> <seed>`
- AdamW: `bash train_adamw.sh <dataset> <model> <seed>`
- AdaHessian: `bash train_adahessian.sh <dataset> <model> <seed>`
- IVON: `bash train_ivon.sh <dataset> <model> <seed>`

## Run test

- SGD: `bash test_sgd.sh <dataset> <model>`
- AdamW: `bash test_adamw.sh <dataset> <model>`
- AdaHessian: `bash test_adahessian.sh <dataset> <model>`
- IVON (mean): `bash test_ivon_mean.sh <dataset> <model>`
- IVON (Bayes): `bash test_ivon_bayes.sh <dataset> <model>`

Note that each test script works only if the respective 5 training runs with seed 0-4 have been executed!
