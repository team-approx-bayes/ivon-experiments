# IVON Experiments

This repository contains the source code for the experiments presented in the following paper:

----

__Variational Learning is Effective for Large Deep Networks__  
_Y. Shen\*, N. Daheim\*, B. Cong, P. Nickl, G.M. Marconi, C. Bazan, R. Yokota, I. Gurevych, D. Cremers, M.E. Khan, T. Möllenhoff_ \
Paper: https://arxiv.org/abs/2402.17641

----

This repository mainly consists of Python source code and Bash scripts to run the experiments. Our source code is based on the [PyTorch](https://pytorch.org/) deep learning framework.

 This code base depends on an implementation of the IVON optimizer which is released in a separate repo ([https://github.com/team-approx-bayes/ivon](https://github.com/team-approx-bayes/ivon)) and as a pip installable package [`ivon-opt`](https://pypi.org/project/ivon-opt/).

## Pip setup

Use the following command to install the necessary dependencies using pip:

`pip install -r requirements.txt`

You can inspect the [`requirements.txt`](./requirements.txt) file to understand the dependencies and customize them as needed. By default, we use ivon-opt-0.1.2 which is the version of the IVON optimizer we used in the paper, but using newer (and optimized) versions of IVON should give comparable results.

### FFCV dependencies for ImageNet experiments

For ImageNet experiments we use `ffcv` to speed up the dataloading. If you wish to run ImageNet experiments, you need to additionally install ffcv as follows

`conda env update --file addon_ffcv.yml`

## Content organization

This repository organizes experimental code into separate folders:
- [`gpt2/`](./gpt2): contains the GPT-2 experiments in Section 4.1.1. Follow [these instructions](./gpt2/readme.md) to run the experiments;
- [`distributed/`](./distributed): contains the ImageNet experiments in Section 4.1.2. Follow [these instructions](./distributed/readme.md) to run the experiments;
- [`image_classification/`](./image_classification): contains the image classification experiments in Section 4.1.2 that can be run on single GPU, i.e. everything except ImageNet experiments. Follow [these instructions](./image_classification/readme.md) to run the experiments;
- [`bdl_indomain/`](./bdl_indomain): covers the in-domain Bayesian deep learning experiments in Section 4.2.1. Follow [these instructions](./bdl_indomain/readme.md) to run the experiments;
- [`bdl_ood/`](./bdl_ood): covers the OOD experiments from Section 4.2.2. Follow [these instructions](./bdl_ood/readme.md) to run the experiments;
- [`mcsamples/`](./mcsamples): covers the multi MC sample ablation studies from Section 4.2.3. Follow [these instructions](./mcsamples/readme.md) to run the experiments;
- [`bdl_competiton/`](./bdl_competition): covers the [NeurIPS 2021 Bayesian deep learning competition](https://izmailovpavel.github.io/neurips_bdl_competition/) mentioned in Section 4.2.4 and Appendix D.3. Follow [these instructions](./bdl_competition/readme.md) to run the experiments;
- [`common/`](./common): common utility folder used by the source code in other folders.

## How to cite

If this code base helps your research, please consider citing

```
@inproceedings{shen2024variational,
      title={Variational Learning is Effective for Large Deep Networks}, 
      author={Yuesong Shen and Nico Daheim and Bai Cong and Peter Nickl and Gian Maria Marconi and Clement Bazan and Rio Yokota and Iryna Gurevych and Daniel Cremers and Mohammad Emtiyaz Khan and Thomas Möllenhoff},
      booktitle={International Conference on Machine Learning (ICML)},
      year={2024},
      url={https://arxiv.org/abs/2402.17641}
}
```
