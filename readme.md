# IVON Experiments

This repository contains the source code for the experiments presented in the paper

----

__Variational Learning is Effective for Large Deep Networks__  
_Y. Shen\*, N. Daheim\*, B. Cong, P. Nickl, G.M. Marconi, C. Bazan, R. Yokota, I. Gurevych, D. Cremers, M.E. Khan, T. Möllenhoff_ \
Paper: https://arxiv.org/abs/2402.17641

----

This repository mainly consists of Python source code and Bash scripts to run the experiments. Our source code is based on the [PyTorch](https://pytorch.org/) deep learning framework.

 This code base depends on an implementation of the IVON optimizer which is released in a separate repo ([https://github.com/team-approx-bayes/ivon](https://github.com/team-approx-bayes/ivon)) and as a pip installable package [`ivon-opt`](https://pypi.org/project/ivon-opt/).

## Conda setup

Here is a simple one-line command to build the conda environment:

`conda env create --file environment.yml`

This will create a conda environment called `ivon-experiments` with necessary dependencies. Especially, it will install the [`ivon-opt`](https://pypi.org/project/ivon-opt/) package which implements the IVON optimizer.
 
You can inspect the yaml file [`environment.yml`](./environment.yml) to understand the config and customize the conda environment.

### FFCV dependencies for ImageNet experiments

For ImageNet experiments we use `ffcv` to speed up the dataloading. If you wish to run ImageNet experiments, you need to additionally install ffcv as follows

`conda env update --file addon_ffcv.yml`

## Content organization

This repository organizes experimental code into separate folders:
- [`image_classification/`](./image_classification): contains the image classification experiments in Section 4.1.2 that can be run on single GPU, i.e. everything except ImageNet experiments. Follow [these instructions](./image_classification/readme.md) to run the experiments;
- [`bdl_indomain/`](./bdl_indomain): covers the in-domain Bayesian deep learning experiments in Section 4.2.1. Follow [these instructions](./bdl_indomain/readme.md) to run the experiments;
- [`bdl_ood/`](./bdl_ood): covers the OOD experiments from Section 4.2.2. Follow [these instructions](./bdl_ood/readme.md) to run the experiments;
- [`mcsamples/`](./mcsamples): covers the multi MC sample ablation studies from Section 4.2.3. Follow [these instructions](./mcsamples/readme.md) to run the experiments;
- [`common/`](./common): common utility folder used by the source code in other folders.

## How to cite

If this code base helps your research, please consider citing

```
@article{shen2024variational,
      title={Variational Learning is Effective for Large Deep Networks}, 
      author={Yuesong Shen and Nico Daheim and Bai Cong and Peter Nickl and Gian Maria Marconi and Clement Bazan and Rio Yokota and Iryna Gurevych and Daniel Cremers and Mohammad Emtiyaz Khan and Thomas Möllenhoff},
      journal={arXiv:2402.17641},
      year={2024},
      url={https://arxiv.org/abs/2402.17641}
}
```