import argparse
from pathlib import Path
import numpy as np
import itertools
import random

import pandas as pd
from models import *
from utils import *
# from data import get_data_loaders, get_unpaired_blood
from data_PET import get_data_loaders, read_data
from MCSUVR import load_weights, cal_MCSUVR_torch, cal_correlation
# from model1D_new import define_G
# from pool import ImagePool

import torch
from sklearn.linear_model import LinearRegression
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

parser.add_argument("--finetune", type=int, default=0, help="whether finetune model")
parser.add_argument("--finetune_lr", type=float, default=0.00001, help="finetune lr")
parser.add_argument("--model_path", type=str, default='./model', help="path to saved model")
parser.add_argument("--data_path", type=str, default='./data', help="path to saved data")
parser.add_argument("--max_cor", type=float, default=-1, help="initialization of max correlation")

parser.add_argument("--SUVR", type=int, default=1, help="whether SUVR")

parser.add_argument("--seed", type=int, default=0, help="random seed, default=0") 
parser.add_argument("--shuffle", type=int, default=1, help="shuffle data, default=1")


opt = parser.parse_args()
print(opt)

# set random seed
torch.manual_seed(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)
# Ensure deterministic behavior in PyTorch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

######################
REGION_INDEX = load_weights()
######################

# pCL = pd.read_excel('./data_PET/Centioid_Summary.xlsx', sheet_name='Sheet1')

def save_model(model_save_path, suffix):
    torch.save(G_AB.state_dict(), model_save_path / f"G_AB_{suffix}.pth")
    torch.save(G_BA.state_dict(), model_save_path / f"G_BA_{suffix}.pth")
    torch.save(D_A.state_dict(), model_save_path / f"D_A_{suffix}.pth")
    torch.save(D_B.state_dict(), model_save_path / f"D_B_{suffix}.pth")
    
def sample(paired, uPiB_scaler=None, uFBP_scaler=None, fake_PiB_nft=None, ft=False, paired_test=None):
    """Saves a generated sample from the test set"""
    
    # assert paired[0].shape[0] == paired[1].shape[0] == paired[2].shape[0] == 79
    MCSUVR_WEIGHT_PAIRED = paired[2]
    MCSUVR_WEIGHT_PAIRED_TEST = paired_test[2]
    G_AB.eval()
    G_BA.eval()
    
    with torch.no_grad():
        real_A = paired[0].to(device) # FBP
        fake_B = G_AB(real_A) # Fake PiB
        real_B = paired[1].to(device) # PiB
        fake_A = G_BA(real_B) # Fake FBP
        
        real_A_test = paired_test[0].to(device) # FBP
        fake_B_test = G_AB(real_A_test)
        real_B_test = paired_test[1].to(device) # PiB
        fake_A_test = G_BA(real_B_test)
            
        mask_PiB = None

        # PiB
        REAL_MCSUVR_B = cal_MCSUVR_torch(real_B, REGION_INDEX, MCSUVR_WEIGHT_PAIRED, mask=mask_PiB)
        FAKE_MCSUVR_B = cal_MCSUVR_torch(fake_B, REGION_INDEX, MCSUVR_WEIGHT_PAIRED, mask=mask_PiB)
        cor_B = cal_correlation(REAL_MCSUVR_B.cpu().numpy(), FAKE_MCSUVR_B.cpu().numpy())
        
        REAL_MCSUVR_B_TEST = cal_MCSUVR_torch(real_B_test, REGION_INDEX, MCSUVR_WEIGHT_PAIRED_TEST, mask=mask_PiB)
        FAKE_MCSUVR_B_TEST = cal_MCSUVR_torch(fake_B_test, REGION_INDEX, MCSUVR_WEIGHT_PAIRED_TEST, mask=mask_PiB)
        cor_B_test = cal_correlation(REAL_MCSUVR_B_TEST.cpu().numpy(), FAKE_MCSUVR_B_TEST.cpu().numpy())
    
        # calculate relative error
        error_A = torch.mean(torch.abs(fake_A - real_A) / (real_A+1e-8)) 
        error_B = torch.mean(torch.abs(fake_B - real_B) / (real_B+1e-8))
        error_A_test = torch.mean(torch.abs(fake_A_test - real_A_test) / (real_A_test+1e-8))
        error_B_test = torch.mean(torch.abs(fake_B_test - real_B_test) / (real_B_test+1e-8))
         
    return error_A, error_B, error_A_test, error_B_test, cor_B, cor_B_test, fake_A, fake_B, fake_A_test, fake_B_test


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

input_dim = 85
latent_dim = 85
# Initialize generator and discriminator
G_AB = Generater_MLP_Skip(input_dim, opt.generator_width, latent_dim, opt.num_residual_blocks_generator)
G_BA = Generater_MLP_Skip(input_dim, opt.generator_width, latent_dim, opt.num_residual_blocks_generator)   

D_A = PatchMLPDiscriminator_1D_Res(opt.num_patch, patch_size=opt.patch_size, hidden_size=opt.discriminator_width, num_residual_blocks=opt.num_residual_blocks_discriminator)
D_B = PatchMLPDiscriminator_1D_Res(opt.num_patch, patch_size=opt.patch_size, hidden_size=opt.discriminator_width, num_residual_blocks=opt.num_residual_blocks_discriminator)

if opt.finetune == 1:
    logger.info(f'Finetune model from saved model {opt.model_path}')
    model_path = Path(opt.model_path)
    G_AB.load_state_dict(torch.load(model_path / 'G_AB_Best_Cor.pth'))
    G_BA.load_state_dict(torch.load(model_path / 'G_BA_Best_Cor.pth'))
    D_A.load_state_dict(torch.load(model_path / 'D_A_Best_Cor.pth'))
    D_B.load_state_dict(torch.load(model_path / 'D_B_Best_Cor.pth'))
    
    opt.lr = opt.finetune_lr
    fake_PiB_nft = None
else:
    fake_PiB_nft = None


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

# Learning rate update schedulers (original)
# lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
#     optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
# )
# lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
#     optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
# )
# lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
#     optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
# )

lr_scheduler_G = CosineAnnealingLR_with_Restart_WeightDecay(optimizer_G, T_max=5, T_mult=2, eta_min=0.00001, eta_max=opt.lr, decay=0.8)
lr_scheduler_D_A = CosineAnnealingLR_with_Restart_WeightDecay(optimizer_D_A, T_max=5, T_mult=2, eta_min=0.00001, eta_max=opt.lr, decay=0.8)
lr_scheduler_D_B = CosineAnnealingLR_with_Restart_WeightDecay(optimizer_D_B, T_max=5, T_mult=2, eta_min=0.00001, eta_max=opt.lr, decay=0.8)

# data
uPiB, uFBP, pPiB, pFBP, uPiB_CL, uFBP_CL, pPiB_CL, pFBP_CL, uPiB_scaler, uFBP_scaler, pWeight, uWeight = read_data(normalize=False, separate=False, baseline=opt.baseline, SUVR=opt.SUVR)

# split val/test data
bins = np.histogram_bin_edges(pFBP_CL, bins=5)
bin_indices = np.digitize(pFBP_CL, bins)
val_idx = []
test_idx = []
for bin_num in range(1, len(bins)):
    bin_data = pFBP_CL[bin_indices == bin_num]
    bin_data_indices = np.where(bin_indices == bin_num)[0]
    # Shuffle the data and indices within the bin
    shuffled_indices = np.random.permutation(len(bin_data))
    bin_data_indices = bin_data_indices[shuffled_indices]
    half_size = len(bin_data) // 2  
    val_idx.append(bin_data_indices[:half_size])
    test_idx.append(bin_data_indices[half_size:])
    
# Concatenate the parts
val_idx = np.concatenate(val_idx)
test_idx = np.concatenate(test_idx)
np.save(data_save_path / 'val_idx.npy', val_idx)
np.save(data_save_path / 'test_idx.npy', test_idx)
logger.info(f'Val indices: {val_idx}')
logger.info(f'Test indices: {test_idx}')


# pool
fake_A_pool = ReplayBuffer(opt.pool_size)
fake_B_pool = ReplayBuffer(opt.pool_size)

# ----------
#  Training
# ----------
max_cor_B = opt.max_cor
min_error_B = 100
min_error_AB = 100


resample ={0:False, 1:'matching', 2:'resample_to_n', 3:'resample_tail', 4:'resample_CL_threshold'}