# General information
cuda = True
sanity = False
load_model = False
# Data information
input_size = 784
nclasses = 10
# Training parameters
lr = 0.001
weight_decay = 0.0001
batch_size = 32
test_batch_size = 64
max_epoch = 40
start_epoch = 0

# scheduler
patience = 3
factor = 0.01

# model prefix
model_prefix = 'baseline_s2s_'

# Network parameters
hidden = [64, 32]
dropout = 0.2
