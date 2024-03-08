# Out-Of-Distribution Benchmark with SVHN and Flowers102

This covers the OOD experiments from Section 4.2.2.

Before running the scripts in this folder, make sure the respective training script from [`bdl_indomain/`](../bdl_indomain) are already executed, since this folder depends on the in-domain models trained on CIFAR-10.

## Run evaluation on SVHN

- SGD (MAP): `bash run_sgd_svhn.sh`
- MC dropout: `bash run_mcdrop_svhn.sh`
- SWAG: `bash run_swag_svhn.sh`
- IVON: `bash run_ivon_svhn.sh`

## Run evaluation on Flowers102

- SGD (MAP): `bash run_sgd_flowers102.sh`
- MC dropout: `bash run_mcdrop_flowers102.sh`
- SWAG: `bash run_swag_flowers102.sh`
- IVON: `bash run_ivon_flowers102.sh`