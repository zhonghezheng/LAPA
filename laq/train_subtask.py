from laq_model import LAQTrainer
from laq_model import LatentActionQuantization

laq = LatentActionQuantization(
    dim = 1024,
    quant_dim=32,
    codebook_size = 8,
    image_size = 256,
    patch_size = 32, #no spatial patches, learn action for the whole image
    spatial_depth = 8, #8
    temporal_depth = 8, #8
    dim_head = 64,
    heads = 16,
    code_seq_len=1,
).cuda()

trainer = LAQTrainer(
    laq,
    folder_train = 'minigrid_example/data_images/train',
    folder_val = 'minigrid_example/data_images/test',
    offsets = 4,
    batch_size = 10,
    grad_accum_every = 1,
    train_on_images = False, 
    use_ema = False,          
    num_train_steps = 100001,
    results_folder='minigrid_example/results/high_level_step_4_100k',
    lr=1e-4,
    save_model_every=10000,
    save_results_every=10000,
    subtask=True
)

trainer.train()

