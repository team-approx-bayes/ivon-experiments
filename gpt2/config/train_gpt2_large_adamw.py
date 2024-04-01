wandb_log = True
wandb_project = 'ivon-gpt2'
wandb_run_name='large-adamw'

n_layer = 36
n_head = 20
n_embd = 1280

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 4
block_size = 1024
gradient_accumulation_steps = 120

# this makes total number of tokens be 50B
max_iters = 100000
lr_decay_iters = 100000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

learning_rate = 2e-4
min_lr = 0
