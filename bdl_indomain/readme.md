# In-domain Benchmark (CIFAR-10)

This covers the in-domain Bayesian deep learning experiments in Section 4.2.1.

## Run training

- SGD (MAP): `bash train_sgd.sh` (5 runs, seed 0-4)
- BBB: `bash train_bbb.sh` (5 runs, seed 0-4)
- MC dropout: `bash train_mcdrop.sh` (5 runs, seed 0-4)
- SWAG: `bash train_swag.sh` (5 runs, seed 0-4)
- IVON: `bash train_ivon.sh` (5 runs, seed 0-4)
- Deep ensemble: `bash train_for_deep_ensemble.sh` (runs SGD for seeds 5-24)
- Multi-IVON: `bash train_for_multi_ivon.sh` (runs IVON for seeds 5-24)

## Run test

- SGD (MAP): `bash test_sgd.sh`
- BBB: `bash test_bbb.sh`
- MC dropout: `bash test_mcdrop.sh`
- SWAG: `bash test_swag.sh`
- IVON: `bash test_ivon.sh`
- Deep ensemble: `bash test_deepensemble.sh`
- Multi-IVON: `bash test_multiivon.sh`