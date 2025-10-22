from laq_model import LAQTrainer
from laq_model import LatentActionQuantization

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

trainer = LAQTrainer(
    laq,
    folder_train = 'minigrid_example/data_images/train',
    folder_val = 'minigrid_example/data_images/test',
    offsets = 1,
    batch_size = 100,
    grad_accum_every = 1,
    train_on_images = False, 
    use_ema = False,          
    num_train_steps = 10001,
    results_folder='minigrid_example/results',
    lr=1e-4,
    save_model_every=1000,
    save_results_every=1000,
)

trainer.train()

