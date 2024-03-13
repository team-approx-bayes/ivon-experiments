# Training/inference MC samples ablation

This folder implements the MC samples ablation studies in Section 4.2.3 and covers 2 cases:
- Training with various MC samples, in this case we always do posterior averaging with 64 MC samples during test
- Inference with various MC samples, in this case we always test on the models trained using only 1 MC sample.

## Run training with different MC samples

Select `<seed>` from 0-4 and `<train_mc>` in {1, 2, 4, 8, 16, 32} and run

`bash train_ivon.sh <seed> <train_mc>`

## Evaluate varying training MC samples

Select `<train_mc>` in {1, 2, 4, 8, 16, 32}, make sure the respective 5 training runs with `<seed>` 0-4 have been executed, and run

`bash run_ablation_train_mc.sh <train_mc>`

## Run tests with varying test MC samples

Select `<test_mc>` in {1, 2, 4, 8, 16, 32, 64, 128, 256, 512}, make sure the 5 training runs with `<seed>` 0-4 and `<train_mc>` 1 have been executed, and run

`bash run_ablation_test_mc.sh <test_mc>`