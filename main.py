import argparse
import os
from pathlib import Path
import numpy as np
import math
import itertools
import datetime
import time

from models import *
from utils import *
# from data import get_data_loaders, get_unpaired_blood
from data_PET import get_data_loaders
from MCSUVR import get_pID_df, cal_MCSUVR, tensor_to_df, cal_correlation

import torch.nn as nn
import torch.nn.functional as F
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
parser.add_argument("--checkpoint_interval", type=int, default=20, help="interval between saving model checkpoints")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")

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

######################
PPIB_RAW, PPIB_DF, PID = get_pID_df()
######################


def save_model(model_save_path, suffix):
    torch.save(G_AB.state_dict(), model_save_path / f"G_AB_{suffix}.pth")
    torch.save(G_BA.state_dict(), model_save_path / f"G_BA_{suffix}.pth")
    torch.save(D_A.state_dict(), model_save_path / f"D_A_{suffix}.pth")
    torch.save(D_B.state_dict(), model_save_path / f"D_B_{suffix}.pth")
    
def sample_images(val_dataloader):
    """Saves a generated sample from the test set"""
    paired = next(iter(val_dataloader))
    assert paired[0].shape[0] == paired[1].shape[0] == 46
    G_AB.eval()
    G_BA.eval()
    real_A = paired[0].to(device) # FBP
    fake_B = G_AB(real_A) # Fake PiB
    real_B = paired[1].to(device) # PiB
    fake_A = G_BA(real_B) # Fake FBP
    
    fake_B_df = tensor_to_df(fake_B, PID, PPIB_DF)
    REAL_MCSUVR, FAKE_MCSUVR = cal_MCSUVR(PPIB_RAW, fake_B_df)
    cor = cal_correlation(REAL_MCSUVR, FAKE_MCSUVR)
    
    # calculate relative error
    error_A = torch.mean(torch.abs(fake_A - real_A) / (real_A+1e-8)) 
    error_B = torch.mean(torch.abs(fake_B - real_B) / (real_B+1e-8)) 
    return error_A, error_B, cor, fake_A, fake_B

# path
log_path = Path(opt.log_path)
training_setup = f'{opt.generator_width}_{opt.discriminator_width}_{opt.num_residual_blocks_generator}_{opt.num_residual_blocks_discriminator}_{opt.resample}_{opt.lambda_cyc}_{opt.lambda_id}'

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
    

# Losses
# criterion_GAN = torch.nn.MSELoss()
criterion_GAN = torch.nn.BCELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
# input_shape = (opt.channels, opt.img_height, opt.img_width)
input_dim = 85
latent_dim = 85
# Initialize generator and discriminator
G_AB = Generater_MLP_Skip(input_dim, opt.generator_width, latent_dim, opt.num_residual_blocks_generator)
G_BA = Generater_MLP_Skip(input_dim, opt.generator_width, latent_dim, opt.num_residual_blocks_generator)   
D_A = Discriminator_MLP_Skip(input_dim, hidden_size=opt.discriminator_width, num_layers=opt.num_residual_blocks_discriminator)
D_B = Discriminator_MLP_Skip(input_dim, hidden_size=opt.discriminator_width, num_layers=opt.num_residual_blocks_discriminator)
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


# ----------
#  Training
# ----------
max_cor = -1e8
resample ={0:False, 1:'matching', 2:'resample_to_n'}

for epoch in range(opt.epoch, opt.n_epochs):

    # data loader
    # resample for each epoch
    val_dataloader, dataloader = get_data_loaders(128, 46, resample=resample[opt.resample])
  
    
    for i, batch in enumerate(dataloader):

        # Set model input
        real_A = batch[0].to(device)
        real_B = batch[1].to(device)
        # Adversarial ground truths
        valid = torch.full((real_A.shape[0],), 1, dtype=torch.float, device=device)
        fake = torch.full((real_A.shape[0],), 0, dtype=torch.float, device=device)

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
        loss_GAN_AB = criterion_GAN(D_B(fake_B).view(-1), valid)
        fake_A = G_BA(real_B)
        loss_GAN_BA = criterion_GAN(D_A(fake_A).view(-1), valid)
        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        # Cycle loss
        recov_A = G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        recov_B = G_AB(fake_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)
        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Total loss
        loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity
        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------
        optimizer_D_A.zero_grad()
        # Real loss
        loss_real = criterion_GAN(D_A(real_A).view(-1), valid)
        # Fake loss (on batch of previously generated samples)
        loss_fake = criterion_GAN(D_A(fake_A.detach()).view(-1), fake)
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2
        loss_D_A.backward()
        optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------
        optimizer_D_B.zero_grad()
        # Real loss
        loss_real = criterion_GAN(D_B(real_B).view(-1), valid)
        # Fake loss (on batch of previously generated samples)
        loss_fake = criterion_GAN(D_B(fake_B.detach()).view(-1), fake)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2
        loss_D_B.backward()
        optimizer_D_B.step()
        
        loss_D = (loss_D_A + loss_D_B) / 2

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        # time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        # prev_time = time.time()

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            error_A, error_B, cor, fake_A, fake_B = sample_images(val_dataloader)
            if cor > max_cor:
                max_cor = cor
                cor_error_B = error_B
                cor_error_A = error_A
                # save tensors
                torch.save(fake_B, f'./{data_save_path}/fake_B_Best.pt')
                torch.save(fake_A, f'./{data_save_path}/fake_A_Best.pt')
                # save model
                save_model(model_save_path, 'Best')
                
            logger.info(f"Epoch: {epoch}, Batch: {i}, D loss: {loss_D.item():.4f}, G loss: {loss_G.item():.4f}, adv: {loss_GAN.item():.4f}, cycle: {loss_cycle.item():.4f}, identity: {loss_identity.item():.4f}, cor_error_A: {cor_error_A:.4f}, cor_error_B: {cor_error_B:.4f},  max_cor: {max_cor:.4f}")
           
    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
    #     # Save model checkpoints
    #     torch.save(G_AB.state_dict(), model_save_path / "/G_AB_%d.pth" % (epoch,))
    #     torch.save(G_BA.state_dict(), model_save_path / "./saved_models_PET/G_BA_%d.pth" % (epoch,))
    #     torch.save(D_A.state_dict(), model_save_path / "./saved_models_PET/D_A_%d.pth" % (epoch,))
    #     torch.save(D_B.state_dict(), model_save_path / "./saved_models_PET/D_B_%d.pth" % (epoch,))