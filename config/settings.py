import os
import logging

aug_data = False # Set as False for fair comparison

# hyper-parameter
batch_size = 1
patch_size = 512
lr = 1e-3

# dir
data_dir = '../dataset/OTS_BETA/'
input_dir = 'hazy'
target_dir = 'clear'
trans_dir = 'trans'
real_dir = 'real'
real_output_dir = 'real_output'
log_dir = '../logdir'
show_dir = '../showdir'
model_dir = '../models'
trans_root_dir = '../dataset/OTS_BETA/test/hazy/'
trans_target_dir = '../trans/'

# other training settings 
log_level = 'info'
model_path = os.path.join(model_dir, 'latest')
save_steps = 1000
iteration = 600000

num_workers = 0
num_GPU = 1
device_id = 0

# loss weight
perceptual_weight = 0
loss_weight = 2e-1
total_variation_weight = 1e-5
ssim_loss_weight = 8e-1
adversarial_weight = 0

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
