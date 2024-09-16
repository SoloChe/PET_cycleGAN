import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, Dataset

import pandas as pd
import numpy as np

def to_np_float(x):
    return x.to_numpy().astype(float)

# Resample sequences to match the largest bin size
def resample_to_max_bin_size(sequence1, sequence2, hist1, hist2, bin_edges):
    resampled_sequence1 = []
    resampled_indices1 = []
    resampled_sequence2 = []
    resampled_indices2 = []
    for count1, count2, (start, end) in zip(hist1, hist2, zip(bin_edges[:-1], bin_edges[1:])):
        if count1 > 0 and count2 > 0:
            bin_indices1 = np.where((sequence1 >= start) & (sequence1 < end))[0]
            bin_indices2 = np.where((sequence2 >= start) & (sequence2 < end))[0]
            
            if count1 > count2:
                indices = np.random.choice(bin_indices2, count1-count2, replace=True)
                resampled_sequence2.extend(sequence2[indices])
                resampled_indices2.extend(indices)
            elif count2 > count1:
                indices = np.random.choice(bin_indices1, count2-count1, replace=True)
                resampled_sequence1.extend(sequence1[indices])
                resampled_indices1.extend(indices)
    return np.array(resampled_indices1), np.array(resampled_indices2)


def read_data(numpy=True):
    uPiB1_path = '/home/yche14/epi_cycleGAN/data_PET/AIBL_PIB_PUP.xlsx'
    uPiB2_path = '/home/yche14/epi_cycleGAN/data_PET/OASIS_PIB_PUP.xlsx'
    uFBP_path = '/home/yche14/epi_cycleGAN/data_PET/ALL-AV45-PUP-BAI-SUVR-11162023.xlsx'

    pPiB_path = '/data/amciilab/processedDataset/Centiloid/FBP_PIB/PUP_PIB/runExtract/list_id_SUVR.csv'
    pFBP_path = '/data/amciilab/processedDataset/Centiloid/FBP_PIB/PUP_FBP/runExtract/list_id_SUVR.csv'
    
    uPiB1 = pd.read_excel(uPiB1_path, sheet_name='AIBL_PIB_PUP')
    uPiB1_CL = pd.read_excel(uPiB1_path, sheet_name='Sheet1')
    uPiB2 = pd.read_excel(uPiB2_path, sheet_name='OASIS_PIB_PUP')
    uPiB2_CL = pd.read_excel(uPiB2_path, sheet_name='Summary')

    uFBP = pd.read_excel(uFBP_path, sheet_name='ALL_AV45_PUP_BAI_SUVR')
    uFBP_CL = pd.read_excel(uFBP_path, sheet_name='Demo')

    pPiB = pd.read_csv(pPiB_path)
    pFBP = pd.read_csv(pFBP_path)
    
    uPiB1 = uPiB1.iloc[:, 1:90]
    uPiB2 = uPiB2.iloc[:, 1:90]
    uFBP = uFBP.iloc[:, 1:90]
    pPiB = pPiB.iloc[:, 1:90]
    pFBP = pFBP.iloc[:, 1:90]
    
    uPiB = pd.concat([uPiB1, uPiB2], axis=0)
    uPiB_CL = pd.concat([uPiB1_CL, uPiB2_CL], axis=0)['CL']
    uFBP_CL = uFBP_CL['CL']
    
    names = []
    for i, name in enumerate(uFBP.columns):
        if len(uPiB[name].unique()) == 1:
            print(i, name) 
            names.append(name)
    
    uPiB = uPiB.drop(columns=names)
    uFBP = uFBP.drop(columns=names)
    pPiB = pPiB.drop(columns=names)
    pFBP = pFBP.drop(columns=names)
    
    if numpy:
        return to_np_float(uPiB), to_np_float(uFBP), to_np_float(pPiB), to_np_float(pFBP), to_np_float(uPiB_CL), to_np_float(uFBP_CL)
    else:
        return uPiB, uFBP, pPiB, pFBP, uPiB_CL, uFBP_CL
    
class PairedDataset(Dataset):
    def __init__(self, pFBP, pPiB):
        self.pFBP = pFBP
        self.pPiB = pPiB
    def __len__(self):
        return len(self.pFBP)
    def __getitem__(self, idx):
        return self.pFBP[idx], self.pPiB[idx]
    
class UnpairedDataset(Dataset):
    def __init__(self, uFBP, uPiB, uPiB_CL=None, uFBP_CL=None, resample=True):
        self.uFBP = uFBP
        self.uPiB = uPiB
       
        self.resample = resample
        if self.resample:
            assert uPiB_CL is not None and uFBP_CL is not None
            # Calculate histograms
            bins = np.histogram_bin_edges(np.concatenate([uPiB_CL, uFBP_CL]), bins=20)
            hist1, bin_edges1 = np.histogram(uPiB_CL, bins=bins)
            hist2, _ = np.histogram(uFBP_CL, bins=bins)
            
            idx_PiB, _ = resample_to_max_bin_size(uPiB_CL, uFBP_CL, hist1, hist2, bin_edges1)
            
            # select the indices that are in both resampled sequences
            self.resample_FBP = self.uFBP
            resample_PiB = self.uPiB[idx_PiB]
            self.resample_PiB = np.concatenate([self.uPiB, resample_PiB], axis=0)    
        else:
            self.resample_FBP = self.uFBP
            self.resample_PiB = self.uPiB
            
        self.len = max(len(self.resample_FBP), len(self.resample_PiB))    
        self.len1 = len(self.resample_FBP)
        self.len2 = len(self.resample_PiB)
        print(self.len1)
        print(self.len2)
    
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        return self.resample_FBP[idx%self.len1], self.resample_PiB[idx%self.len2]
    
def get_data_loaders(batch_size_p, batch_size_u, resample=True):
    uPiB, uFBP, pPiB, pFBP, uPiB_CL, uFBP_CL = read_data(numpy=True)
    uPiB = torch.from_numpy(uPiB).float()
    uFBP = torch.from_numpy(uFBP).float()
    pPiB = torch.from_numpy(pPiB).float()
    pFBP = torch.from_numpy(pFBP).float()
    paired_dataset = PairedDataset(pFBP, pPiB)
    unpaired_dataset = UnpairedDataset(uFBP, uPiB, uPiB_CL, uFBP_CL, resample=resample)
    paired_loader = DataLoader(paired_dataset, batch_size=batch_size_p, shuffle=False)
    unpaired_loader = DataLoader(unpaired_dataset, batch_size=batch_size_u, shuffle=True)
    return paired_loader, unpaired_loader

if __name__ == "__main__":
    paired_loader, unpaired_loader = get_data_loaders(2, 2)
    for i, (brain, blood) in enumerate(paired_loader):
        print(brain.shape, blood.shape)
        print(brain.dtype, blood.dtype)
        if i == 0:
            break
    for i, (brain, blood) in enumerate(unpaired_loader):
        print(brain.shape, blood.shape)
        print(brain.dtype, blood.dtype)
        if i == 0:
            break
