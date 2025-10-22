import torch
import torch.nn as nn

from laq_model import LAQTrainer
from laq_model import LatentActionQuantization

# Define your model class (must be the same architecture as the saved model)
laq = LatentActionQuantization(
    dim = 1024,
    quant_dim=32,
    codebook_size = 8,
    image_size = 256,
    patch_size = 32,
    spatial_depth = 4, #8
    temporal_depth = 5, #8
    dim_head = 64,
    heads = 16,
    code_seq_len=4,
).cuda()

# Load the state_dict
# If saved on GPU and loading on CPU:
# state_dict = torch.load('model_weights.pt', map_location=torch.device('cpu'))
# If saved on GPU and loading on GPU:
# state_dict = torch.load('model_weights.pt')
# If saved on CPU and loading on GPU:
state_dict = torch.load('/n/fs/cat10301/projects/latent_action_task_decmoposition/minigrid_example/results/low_level/vae.10000.pt') # Or your desired GPU device

# TODO: test this...
laq.load_state_dict(state_dict)
laq.eval()

# Instantiate the model
trainer = LAQTrainer(
    laq,
    folder_train = 'minigrid_example/data_images/train',
    folder_val = 'minigrid_example/data_images/test',
    offsets = 1,
    batch_size = 10,
    grad_accum_every = 1,
    train_on_images = False, 
    use_ema = False,          
    num_train_steps = 10001,
    results_folder='minigrid_example/results/low_level',
    lr=1e-4,
    save_model_every=1000,
    save_results_every=1000,
)

trainer.recon_every_code()