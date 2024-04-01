wandb_log = True
wandb_project = 'ivon-gpt2'
wandb_run_name='small-ivon'

batch_size = 20
block_size = 1024
gradient_accumulation_steps = 3 * 8

# this makes total number of tokens be 50B
max_iters = 100000
lr_decay_iters = 100000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-6

learning_rate = 300.0
min_lr = 0
