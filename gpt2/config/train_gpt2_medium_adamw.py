wandb_log = True
wandb_project = 'ivon-gpt2'
wandb_run_name='medium-adamw'

n_layer = 24
n_head = 16
n_embd = 1024

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# this makes total number of tokens be 50B
max_iters = 100000
lr_decay_iters = 100000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

learning_rate = 3e-4
min_lr = 0
