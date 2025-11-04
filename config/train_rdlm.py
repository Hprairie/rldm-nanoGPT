# config for training RDLM model
# launch as the following (e.g. in a screen session):
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_rdlm.py

# model config
model_type = 'rdlm'  # Use RDLM model
n_preulude_layers = 3
n_recurrent_layers = 3
n_coda_layers = 3
n_head = 12
n_embd = 768
n_intermediate_embed = None  # Will default to n_embd * 4
norm = 'rms'  # 'rms' or 'layer'
mlp_type = 'gated'  # 'gated' or 'mlp'
norm_style = 'post'  # 'post', 'pre', or 'both'
norm_eps = 1e-6
qk_bias = False
n_kv_heads = None  # Will default to n_head
dropout = 0.0  # Dropout is not supported for RDLM (must be 0.0)
bias = True
rope_base = 10000.0
# RDLM-specific training config
mean_recurrence = 32  # Mean number of recurrent steps
mean_backprop_depth = 8  # Mean backprop depth for randomized iteration sampler
init_values_std = 0.02  # Standard deviation for state initialization

wandb_log = True
wandb_project = 'owt'
wandb_run_name='rdlm'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

