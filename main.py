import argparse
from pathlib import Path
import numpy as np
import itertools
import random

from models import *
from utils import *
# from data import get_data_loaders, get_unpaired_blood
from data_PET import get_data_loaders, read_data
from MCSUVR import load_weights, cal_MCSUVR_torch, cal_correlation
# from model1D_new import define_G
# from pool import ImagePool

import torch

import logging

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="PET", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adamw: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adamw: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--sample_interval", type=int, default=10, help="interval between saving generator outputs")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
parser.add_argument("--lambda_mc", type=float, default=5.0, help="MCSUVR loss weight")
parser.add_argument("--pool_size", type=int, default=50, help="size of image buffer that stores previously generated images")
parser.add_argument("--patch_size", type=int, default=85, help="size of patch for patchGAN discriminator")
parser.add_argument("--num_patch", type=int, default=1, help="number of patch for patchGAN discriminator")

parser.add_argument("--baseline", type=int, default=0, help="whether baseline model")

parser.add_argument("--resample", type=int, default=0, help="resample unpaired data")
parser.add_argument("--generator_width", type=int, default=512, help="width of the generator")
parser.add_argument("--num_residual_blocks_generator", type=int, default=8, help="number of residual blocks in the generator")
parser.add_argument("--discriminator_width", type=int, default=128, help="width of the discriminator")
parser.add_argument("--num_residual_blocks_discriminator", type=int, default=2, help="number of residual blocks in the discriminator")
parser.add_argument("--log_path", type=str, default='./training_logs3', help="path to save log file")
opt = parser.parse_args()
print(opt)

# set random seed
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
# Ensure deterministic behavior in PyTorch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

######################
MCSUVR_WEIGHT, _, REGION_INDEX = load_weights()
######################


def save_model(model_save_path, suffix):
    torch.save(G_AB.state_dict(), model_save_path / f"G_AB_{suffix}.pth")
    torch.save(G_BA.state_dict(), model_save_path / f"G_BA_{suffix}.pth")
    torch.save(D_A.state_dict(), model_save_path / f"D_A_{suffix}.pth")
    torch.save(D_B.state_dict(), model_save_path / f"D_B_{suffix}.pth")
    
def sample_images(paired, uPiB_scaler=None, uFBP_scaler=None):
    """Saves a generated sample from the test set"""
    
    assert paired[0].shape[0] == paired[1].shape[0] == 46
    G_AB.eval()
    G_BA.eval()
    
    with torch.no_grad():
        real_A = paired[0].to(device) # FBP
        fake_B = G_AB(real_A) # Fake PiB
        real_B = paired[1].to(device) # PiB
        fake_A = G_BA(real_B) # Fake FBP
        
        if uPiB_scaler is not None and uFBP_scaler is not None:
            real_A = torch.from_numpy(uFBP_scaler.inverse_transform(real_A.cpu().numpy()))
            real_B = torch.from_numpy(uPiB_scaler.inverse_transform(real_B.cpu().numpy()))
            fake_A = torch.from_numpy(uFBP_scaler.inverse_transform(fake_A.cpu().numpy()))
            fake_B = torch.from_numpy(uPiB_scaler.inverse_transform(fake_B.cpu().numpy()))
    
        REAL_MCSUVR_B = cal_MCSUVR_torch(real_B, REGION_INDEX, MCSUVR_WEIGHT)
        FAKE_MCSUVR_B = cal_MCSUVR_torch(fake_B, REGION_INDEX, MCSUVR_WEIGHT)
        cor_B = cal_correlation(REAL_MCSUVR_B.cpu().numpy(), FAKE_MCSUVR_B.cpu().numpy())
        
        REAL_MCSUVR_A = cal_MCSUVR_torch(real_A, REGION_INDEX, MCSUVR_WEIGHT)
        FAKE_MCSUVR_A = cal_MCSUVR_torch(fake_A, REGION_INDEX, MCSUVR_WEIGHT)
        cor_A = cal_correlation(REAL_MCSUVR_A.cpu().numpy(), FAKE_MCSUVR_A.cpu().numpy())
        
        # calculate relative error
        error_A = torch.mean(torch.abs(fake_A - real_A) / (real_A+1e-8)) 
        error_B = torch.mean(torch.abs(fake_B - real_B) / (real_B+1e-8))
        
        
    return error_A, error_B, cor_A, cor_B, fake_A, fake_B

def MCSUVR_loss(fake_A, real_A, fake_B, real_B, REGION_INDEX):
    mcsuvr_loss_A  = [criterion_MCSUVR(fake_A[:,i], real_A[:,i]) for i in REGION_INDEX.values()]
    mcsuvr_loss_B  = [criterion_MCSUVR(fake_B[:,i], real_B[:,i]) for i in REGION_INDEX.values()]
    mcsuvr_loss = torch.stack(mcsuvr_loss_A + mcsuvr_loss_B)
    assert mcsuvr_loss.shape[0] == 7*2, f'Error in MCSUVR loss shape: {mcsuvr_loss.shape[0]}'
    return torch.mean(mcsuvr_loss)

# path
log_path = Path(opt.log_path)
training_setup = f'{opt.generator_width}_{opt.discriminator_width}_{opt.num_residual_blocks_generator}_{opt.num_residual_blocks_discriminator}_{opt.lambda_cyc}_{opt.lambda_id}_{opt.lambda_mc}'

log_setup_path = log_path / 'log' / training_setup
model_save_path = log_path / 'saved_model' / training_setup
data_save_path = log_path / 'data' / training_setup
if not log_setup_path.exists():
    log_setup_path.mkdir(parents=True, exist_ok=True)
if not model_save_path.exists():
    model_save_path.mkdir(parents=True, exist_ok=True)
if not data_save_path.exists():
    data_save_path.mkdir(parents=True, exist_ok=True)

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# File handler
file_handler = logging.FileHandler(log_setup_path / "training.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.info('Training started')
    
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

# Losses
criterion_GAN =GANLoss('vanilla')
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# L2 or L1?
criterion_MCSUVR = torch.nn.MSELoss()
# criterion_MCSUVR = torch.nn.L1Loss()


# input_shape = (opt.channels, opt.img_height, opt.img_width)
input_dim = 85
latent_dim = 85
# Initialize generator and discriminator
G_AB = Generater_MLP_Skip(input_dim, opt.generator_width, latent_dim, opt.num_residual_blocks_generator)
G_BA = Generater_MLP_Skip(input_dim, opt.generator_width, latent_dim, opt.num_residual_blocks_generator)   

D_A = PatchMLPDiscriminator_1D_Res(opt.num_patch, patch_size=opt.patch_size, hidden_size=opt.discriminator_width, num_residual_blocks=opt.num_residual_blocks_discriminator)
D_B = PatchMLPDiscriminator_1D_Res(opt.num_patch, patch_size=opt.patch_size, hidden_size=opt.discriminator_width, num_residual_blocks=opt.num_residual_blocks_discriminator)


G_AB = G_AB.to(device)
G_BA = G_BA.to(device)
D_A = D_A.to(device)
D_B = D_B.to(device)


# Optimizers
optimizer_G = torch.optim.AdamW(
    itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D_A = torch.optim.AdamW(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.AdamW(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)

# data
baseline = True if opt.baseline == 1 else False
uPiB, uFBP, pPiB, pFBP, uPiB_CL, uFBP_CL, uPiB_scaler, uFBP_scaler = read_data(normalize=False, separate=False, baseline=baseline)

# pool
fake_A_pool = ReplayBuffer(opt.pool_size)
fake_B_pool = ReplayBuffer(opt.pool_size)

# ----------
#  Training
# ----------
max_cor = -1e8
resample ={0:False, 1:'matching', 2:'resample_to_n'}

for epoch in range(opt.epoch, opt.n_epochs):

    # data loader
    # resample for each epoch
    paired_data, unpaired_loader = get_data_loaders(uPiB, uFBP, pPiB, pFBP, uPiB_CL, uFBP_CL, 
                                                 opt.batch_size, resample=resample[opt.resample])
  
    for i, batch in enumerate(unpaired_loader):

        # Set model input
        real_A = batch[0].to(device)
        real_B = batch[1].to(device)
        
        # ------------------
        #  Train Generators
        # ------------------
        G_AB.train()
        G_BA.train()
        optimizer_G.zero_grad()

        # Identity loss
        loss_id_A = criterion_identity(G_BA(real_A), real_A)
        loss_id_B = criterion_identity(G_AB(real_B), real_B)
        loss_identity = (loss_id_A + loss_id_B) / 2

        # GAN loss
        fake_B = G_AB(real_A)
        loss_GAN_AB = criterion_GAN(D_B(fake_B), True)
        fake_A = G_BA(real_B)
        loss_GAN_BA = criterion_GAN(D_A(fake_A), True)
        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        # Cycle loss
        recov_A = G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        
        recov_B = G_AB(fake_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)
        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
        
        # MCSUVR loss
        loss_MCSUVR = MCSUVR_loss(recov_A, real_A, recov_B, real_B, REGION_INDEX)
        
        # Total loss
        loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity + opt.lambda_mc * loss_MCSUVR
        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------
        optimizer_D_A.zero_grad()
        fake_A = fake_A_pool.push_and_pop(fake_A)
        # Real loss
        loss_real = criterion_GAN(D_A(real_A), True)
        # Fake loss (on batch of previously generated samples)
        loss_fake = criterion_GAN(D_A(fake_A.detach()), False)
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2
        loss_D_A.backward()
        optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------
        optimizer_D_B.zero_grad()
        fake_B = fake_B_pool.push_and_pop(fake_B)
        # Real loss
        loss_real = criterion_GAN(D_B(real_B), True)
        # Fake loss (on batch of previously generated samples)
        loss_fake = criterion_GAN(D_B(fake_B.detach()), False)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2
        loss_D_B.backward()
        optimizer_D_B.step()
        
        loss_D = (loss_D_A + loss_D_B) / 2

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(unpaired_loader) + i
        batches_left = opt.n_epochs * len(unpaired_loader) - batches_done

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            error_A, error_B, cor_A, cor_B, fake_A, fake_B = sample_images(paired_data, uPiB_scaler, uFBP_scaler)
            if cor_B > max_cor:
                max_cor = cor_B
                cor_cor_A = cor_A
                cor_error_B = error_B
                cor_error_A = error_A
                # save tensors
                torch.save(fake_B, f'./{data_save_path}/fake_B_Best.pt')
                torch.save(fake_A, f'./{data_save_path}/fake_A_Best.pt')
                # save model
                save_model(model_save_path, 'Best')
            
            logger.info("+" * 30)
            logger.info(f"Epoch: {epoch}, Batch: {i}, D loss: {loss_D.item():.4f}, G loss: {loss_G.item():.4f}, adv: {loss_GAN.item():.4f}, cycle: {loss_cycle.item():.4f}, identity: {loss_identity.item():.4f}, error_A: {error_A:.4f}, error_B: {error_B:.4f},  cor_A: {cor_A:.4f}, cor_B: {cor_B:.4f} ")
            logger.info("-" * 30)
            logger.info(f"max_cor_B: {max_cor:.4f}, cor_cor_A: {cor_cor_A:.4f}, cor_error_A: {cor_error_A:.4f}, cor_error_B: {cor_error_B:.4f}")
            logger.info("+" * 30)
            logger.info("")
           
    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()
    
    
   