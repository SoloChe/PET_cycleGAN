import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np

def to_np_float(x):
    return x.to_numpy().astype(float)

# Resample sequences to match the largest bin size
def distribution_matching(sequence1, sequence2, hist1, hist2, bin_edges):
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

# resample 500 for each bin
def resample_to_n(sequence1, sequence2, hist1, hist2, bin_edges, n=600):
    resampled_sequence1 = []
    resampled_indices1 = []
    resampled_sequence2 = []
    resampled_indices2 = []
    for count1, count2, (start, end) in zip(hist1, hist2, zip(bin_edges[:-1], bin_edges[1:])):
        if count1 > 0 and count2 > 0:
            bin_indices1 = np.where((sequence1 >= start) & (sequence1 < end))[0]
            bin_indices2 = np.where((sequence2 >= start) & (sequence2 < end))[0]
            
            indices = np.random.choice(bin_indices2, n, replace=True)
            resampled_sequence2.extend(sequence2[indices])
            resampled_indices2.extend(indices)
        
            indices = np.random.choice(bin_indices1, n, replace=True)
            resampled_sequence1.extend(sequence1[indices])
            resampled_indices1.extend(indices)
    return np.array(resampled_indices1), np.array(resampled_indices2)


def read_data(normalize=False, separate=False, baseline=False):
    
    
    uPiB1_path = '/home/yche14/PET_cycleGAN/data_PET/AIBL_PIB_PUP.xlsx'
    uPiB2_path = '/home/yche14/PET_cycleGAN/data_PET/OASIS_PIB_PUP.xlsx'
    uFBP_path = '/home/yche14/PET_cycleGAN/data_PET/ALL-AV45-PUP-BAI-SUVR-11162023.xlsx'

    pPiB_path = '/data/amciilab/processedDataset/Centiloid/FBP_PIB/PUP_PIB/runExtract/list_id_SUVR.csv'
    pFBP_path = '/data/amciilab/processedDataset/Centiloid/FBP_PIB/PUP_FBP/runExtract/list_id_SUVR.csv'
    
    uFBP_sep_path = '/data/amciilab/processedDataset/ADNI/ADNI-PUP/results/FBP/list_ADNI_FBP_SUVRLR.csv'
    uPiB1_sep_path = '/data/amciilab/processedDataset/OASIS/Oasis-PUP/results/PiB_IDS_SUVRLR.csv'
    uPiB2_sep_path = '/data/amciilab/processedDataset/AIBL/AIBL-PUP/results/PiB_IDS_SUVRLR.csv'

    pFBP_sep_path = '/data/amciilab/processedDataset/Centiloid/FBP_PIB/PUP_FBP/runExtract/list_id_SUVRLR.csv'
    pPiB_sep_path = '/data/amciilab/processedDataset/Centiloid/FBP_PIB/PUP_PIB/runExtract/list_id_SUVRLR.csv'
    
    
    # CL are same for both sep. and non-sep. datasets
    uPiB1_CL = pd.read_excel(uPiB1_path, sheet_name='Sheet1')
    uPiB2_CL = pd.read_excel(uPiB2_path, sheet_name='Summary')
    uFBP_CL = pd.read_excel(uFBP_path, sheet_name='Demo')
    
    if not separate:
        uPiB1 = pd.read_excel(uPiB1_path, sheet_name='AIBL_PIB_PUP')
        uPiB2 = pd.read_excel(uPiB2_path, sheet_name='OASIS_PIB_PUP')
        uFBP = pd.read_excel(uFBP_path, sheet_name='ALL_AV45_PUP_BAI_SUVR')
        pPiB = pd.read_csv(pPiB_path)
        pFBP = pd.read_csv(pFBP_path)
        
        if baseline:
            # baseline selection for uPiB2
            sorted_uPiB2 = uPiB2.sort_values(by='ID')
            sorted_uPiB2['ID_prefix'] = uPiB2['ID'].str.extract(r'^(OAS\d+_PIB)')
            sorted_uPiB2 = sorted_uPiB2.drop_duplicates(subset='ID_prefix', keep='first')
            uPiB2 = sorted_uPiB2.drop(columns=['ID_prefix'])

            # baseline selection for uPiB1
            uPiB1 = uPiB1[uPiB1['ID'].str.endswith('bl')]

            # baseline selection for uFBP
            sorted_uFBP = uFBP.sort_values(by=['RID', 'Actual Date'])
            uFBP = sorted_uFBP.drop_duplicates(subset='RID', keep='first')
            
            uFBP_CL = uFBP_CL[uFBP_CL['PUP ID'].isin(uFBP['PUP ID'])]['CL']
            uPiB1_CL_baseline = uPiB1_CL[uPiB1_CL['SID'].isin(uPiB1['ID'])]
            uPiB2_CL_baseline = uPiB2_CL[uPiB2_CL['PIB_ID'].isin(uPiB2['ID'])]
            uPiB_CL = pd.concat([uPiB1_CL_baseline, uPiB2_CL_baseline])['CL']
        else:
            uPiB_CL = pd.concat([uPiB1_CL, uPiB2_CL], axis=0)['CL']
            uFBP_CL = uFBP_CL['CL']
        
        # remove ID column and other columns that are not needed
        uPiB1 = uPiB1.iloc[:, 1:90]
        uPiB2 = uPiB2.iloc[:, 1:90]
        uFBP = uFBP.iloc[:, 1:90]
        pPiB = pPiB.iloc[:, 1:90]
        pFBP = pFBP.iloc[:, 1:90]
    else:
        uFBP = pd.read_csv(uFBP_sep_path) 
        uPiB1 = pd.read_csv(uPiB1_sep_path)
        uPiB2 = pd.read_csv(uPiB2_sep_path)
        uPiB = pd.concat([uPiB1, uPiB2])
        pFBP = pd.read_csv(pFBP_sep_path)
        pPiB = pd.read_csv(pPiB_sep_path)
        
        uPiB_CL = pd.concat([uPiB1_CL, uPiB2_CL], axis=0)['CL']
        uFBP_CL = uFBP_CL['CL']
        
        # remove ID column 
        uPiB1 = uPiB1.iloc[:, 1:]
        uPiB2 = uPiB2.iloc[:, 1:]
        uFBP = uFBP.iloc[:, 1:]
        pPiB = pPiB.iloc[:, 1:]
        pFBP = pFBP.iloc[:, 1:]
        
    uPiB = pd.concat([uPiB1, uPiB2], axis=0)
    
        
    # Drop columns with only one unique value
    names = []
    for i, name in enumerate(uFBP.columns):
        if len(uPiB[name].unique()) == 1:
            # print(i, name) 
            names.append(name)
    # remove cerebellum as requested by Dr. Su for sep. dataset    
    if separate:
        names = names + ['Left-Cerebellum-Cortex', 'Right-Cerebellum-Cortex']  
    
    uPiB = uPiB.drop(columns=names)
    uFBP = uFBP.drop(columns=names)
    pPiB = pPiB.drop(columns=names)
    pFBP = pFBP.drop(columns=names)
    
    # df to numpy
    uPiB = to_np_float(uPiB)
    uFBP = to_np_float(uFBP)
    pPiB = to_np_float(pPiB)
    pFBP = to_np_float(pFBP)
    uPiB_CL = to_np_float(uPiB_CL)
    uFBP_CL = to_np_float(uFBP_CL)
    
    # normalize
    if normalize:
        scaler = MinMaxScaler()
        uPiB_scaler = scaler.fit(uPiB)
        uFBP_scaler = scaler.fit(uFBP)
        uPiB = uPiB_scaler.transform(uPiB)
        uFBP = uFBP_scaler.transform(uFBP)  
        pPiB = uPiB_scaler.transform(pPiB)
        pFBP = uFBP_scaler.transform(pFBP)
    else:
        uPiB_scaler = None
        uFBP_scaler = None
    return uPiB, uFBP, pPiB, pFBP, uPiB_CL, uFBP_CL, uPiB_scaler, uFBP_scaler
    
    
class PairedDataset(Dataset):
    def __init__(self, pFBP, pPiB):
        self.pFBP = pFBP
        self.pPiB = pPiB
    def __len__(self):
        return len(self.pFBP)
    def __getitem__(self, idx):
        return self.pFBP[idx], self.pPiB[idx]
    
class UnpairedDataset(Dataset):
    def __init__(self, uFBP, uPiB, uPiB_CL=None, uFBP_CL=None, resample='matching'):
        self.uFBP = uFBP
        self.uPiB = uPiB
       
        self.resample = resample
        if self.resample:
            assert uPiB_CL is not None and uFBP_CL is not None
            # Calculate histograms
            bins = np.histogram_bin_edges(np.concatenate([uPiB_CL, uFBP_CL]), bins=20)
            hist1, bin_edges1 = np.histogram(uPiB_CL, bins=bins)
            hist2, _ = np.histogram(uFBP_CL, bins=bins)
            
            assert self.resample in ['matching', 'resample_to_n']
            
            if self.resample == 'matching':
                idx_PiB, idx_FBP = distribution_matching(uPiB_CL, uFBP_CL, hist1, hist2, bin_edges1)
                # select the indices that are in both resampled sequences
                self.resample_FBP = self.uFBP
                self.resample_PiB = self.uPiB
                
                if len(idx_PiB) > 0:
                    resample_PiB = self.uPiB[idx_PiB]
                    self.resample_PiB = np.concatenate([self.uPiB, resample_PiB], axis=0) 
                
                if len(idx_FBP) > 0:
                    resample_FBP = self.uFBP[idx_FBP]
                    self.resample_FBP = np.concatenate([self.uFBP, resample_FBP], axis=0)
                    
            elif self.resample == 'resample_to_n':
                idx_PiB, idx_FBP = resample_to_n(uPiB_CL, uFBP_CL, hist1, hist2, bin_edges1)
                self.resample_FBP = self.uFBP[idx_FBP]
                self.resample_PiB = self.uPiB[idx_PiB]
            else:
                raise ValueError('Invalid resample method')
        else:
            self.resample_FBP = self.uFBP
            self.resample_PiB = self.uPiB
            
        self.len = max(len(self.resample_FBP), len(self.resample_PiB))    
        self.len1 = len(self.resample_FBP)
        self.len2 = len(self.resample_PiB)
    
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        return self.resample_FBP[idx%self.len1], self.resample_PiB[idx%self.len2]
    
def get_data_loaders(uPiB, uFBP, pPiB, pFBP, uPiB_CL, uFBP_CL, 
                     batch_size_u, resample='matching'):
    # uPiB = torch.from_numpy(uPiB).float().unsqueeze(1)
    # uFBP = torch.from_numpy(uFBP).float().unsqueeze(1)
    # pPiB = torch.from_numpy(pPiB).float().unsqueeze(1)
    # pFBP = torch.from_numpy(pFBP).float().unsqueeze(1)
    
    uPiB = torch.from_numpy(uPiB).float()
    uFBP = torch.from_numpy(uFBP).float()
    pPiB = torch.from_numpy(pPiB).float()
    pFBP = torch.from_numpy(pFBP).float()
   
    unpaired_dataset = UnpairedDataset(uFBP, uPiB, uPiB_CL, uFBP_CL, resample=resample)
    unpaired_loader = DataLoader(unpaired_dataset, batch_size=batch_size_u, shuffle=True)
    paired_dataset = (pFBP, pPiB)

    return paired_dataset, unpaired_loader

if __name__ == "__main__":
    uPiB, uFBP, pPiB, pFBP, uPiB_CL, uFBP_CL, uPiB_scaler, uFBP_scaler = read_data(normalize=True, separate=False)
    paired, unpaired = get_data_loaders(uPiB, uFBP, pPiB, pFBP, uPiB_CL, uFBP_CL, 16)
    
    for i, (fbp, pib) in enumerate(unpaired):
        print(fbp.shape, pib.shape)
        if i == 0:
            break
        
    fbp, pib = paired
    print(fbp.shape, pib.shape)

