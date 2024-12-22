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

# resample 1000 for each bin
def resample_to_n(sequence1, sequence2, hist1, hist2, bin_edges, n=1000):
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

# resample CL > 25 for fine-tuning
def resample_tail(sequence1, sequence2, hist1, hist2, bin_edges, n1=1000, n2=200):
    # 1000 200
    resampled_sequence1 = []
    resampled_indices1 = []
    resampled_sequence2 = []
    resampled_indices2 = []
    for count1, count2, (start, end) in zip(hist1, hist2, zip(bin_edges[:-1], bin_edges[1:])):
        if count1 > 0 and count2 > 0:
            bin_indices1 = np.where((sequence1 >= start) & (sequence1 < end))[0]
            bin_indices2 = np.where((sequence2 >= start) & (sequence2 < end))[0]
            
            if start > 50:
                indices1 = np.random.choice(bin_indices1, n1, replace=True)
                indices2 = np.random.choice(bin_indices2, n1, replace=True)
            else:
                indices1 = np.random.choice(bin_indices1, n2, replace=True)
                indices2 = np.random.choice(bin_indices2, n2, replace=True)
            
            resampled_sequence1.extend(sequence1[indices1])
            resampled_indices1.extend(indices1)
            resampled_sequence2.extend(sequence2[indices2])
            resampled_indices2.extend(indices2)
    return np.array(resampled_indices1), np.array(resampled_indices2)

def resample_CL_threshold(sequence1, sequence2, CL_threshold, greater=True):
    
    if greater:
        mask1 = sequence1 > CL_threshold
        mask2 = sequence2 > CL_threshold
    else:
        mask1 = sequence1 <= CL_threshold
        mask2 = sequence2 <= CL_threshold
    return np.where(mask1)[0], np.where(mask2)[0]

def get_vox_weight():
    # non sep. vox weight dataset
    uPiB1_weight = pd.read_csv('/home/yche14/PET_cycleGAN/data_PET/unpaired/standard/voxel/AIBL_vox.csv')
    uPiB2_weight = pd.read_csv('//home/yche14/PET_cycleGAN/data_PET/unpaired/standard/voxel/OASIS_vox_removed.csv')
    uPiB3_weight = pd.read_csv('/home/yche14/PET_cycleGAN/data_PET/unpaired/standard/voxel/WRAP_vox.csv')
    uPiB4_weight = pd.read_csv('/home/yche14/PET_cycleGAN/data_PET/unpaired/standard/voxel/ADRC_vox.csv')
    uPiB5_weight  = pd.read_csv('/home/yche14/PET_cycleGAN/data_PET/unpaired/standard/voxel/CLPIB_vox.csv')
    weight_unpaied = pd.concat([uPiB1_weight, uPiB2_weight, uPiB3_weight, uPiB4_weight, uPiB5_weight], ignore_index=True)
    
    pPiB1_paired = pd.read_excel('/home/yche14/PET_cycleGAN/data_PET/paired/standard/voxel/RegionalCycleGAN.xlsx', sheet_name='Sheet1') 
    pPiB2_paired = pd.read_csv('/home/yche14/PET_cycleGAN/data_PET/paired/standard/voxel/OASIS_vox_selected.csv')
    weight_paired = pd.concat([pPiB1_paired, pPiB2_paired], axis=0).dropna(axis=1)
    
    col_name = 'ctx-precuneus'
    val_paired = [1 for _ in range(len(weight_paired))]
    val_unpaired = [1 for _ in range(len(weight_unpaied))]
    weight_paired[col_name] = val_paired
    weight_unpaied[col_name] = val_unpaired
    
    def get(weight):
        ctx_precuneus_w = weight['ctx-precuneus'].values.reshape(-1, 1)
        ctx_rostralmiddlefrontal_w = weight['ctx-rostralmiddlefrontal'].values.reshape(-1, 1)
        ctx_superiorfrontal_w = weight['ctx-superiorfrontal'].values.reshape(-1, 1)
        ctx_middletemporal_w = weight['ctx-middletemporal'].values.reshape(-1, 1)
        ctx_superiortemporal_w = weight['ctx-superiortemporal'].values.reshape(-1, 1)
        ctx_lateralorbitofrontal_w = weight['ctx-lateralorbitofrontal'].values.reshape(-1, 1)
        ctx_medialorbitofrontal_w = weight['ctx-medialorbitofrontal'].values.reshape(-1, 1)
        
        return np.hstack([ctx_precuneus_w, ctx_rostralmiddlefrontal_w, ctx_superiorfrontal_w, ctx_middletemporal_w, ctx_superiortemporal_w, ctx_lateralorbitofrontal_w, ctx_medialorbitofrontal_w])
    
    return get(weight_paired), get(weight_unpaied)
        
    
    
def read_data(normalize=False):
    
    # non sep. SUVR dataset
    uPiB1_path = '/home/yche14/PET_cycleGAN/data_PET/unpaired/standard/PIB/AIBL_PIB_PUP.xlsx'
    uPiB2_path = '/home/yche14/PET_cycleGAN/data_PET/unpaired/standard/PIB/OASIS_PIB_PUP_removed.csv'
    uPiB3_path = '/home/yche14/PET_cycleGAN/data_PET/unpaired/standard/PIB/WRAP_SUVR.csv'
    uPiB4_path = '/home/yche14/PET_cycleGAN/data_PET/unpaired/standard/PIB/ADRC_SUVR.csv'
    uPiB5_path  = '/home/yche14/PET_cycleGAN/data_PET/unpaired/standard/PIB/CLPIB_SUVR.csv'
    uFBP_path = '/home/yche14/PET_cycleGAN/data_PET/unpaired/standard/FBP/ALL-AV45-PUP-BAI-SUVR-11162023.xlsx'
    pPiB1_path = '/home/yche14/PET_cycleGAN/data_PET/paired/standard/PIB/paired_PiB_SUVR.csv'
    pPiB2_path = '/home/yche14/PET_cycleGAN/data_PET/paired/standard/PIB/OASIS_PIB_PUP_selected.csv'
    pFBP1_path = '/home/yche14/PET_cycleGAN/data_PET/paired/standard/FBP/paired_FBP_SUVR.csv'
    pFBP2_path = '/home/yche14/PET_cycleGAN/data_PET/paired/standard/FBP/OASIS_FBP_selected.csv'
    
    # CL are same for both sep. and non-sep. datasets
    uPiB1_CL = pd.read_excel(uPiB1_path, sheet_name='Sheet1') 
    uPiB2_CL = pd.read_csv('/home/yche14/PET_cycleGAN/data_PET/unpaired/standard/PIB/OASIS_PIB_PUP_CL_removed.csv')
    uPiB3_CL = pd.read_excel('/home/yche14/PET_cycleGAN/data_PET/unpaired/standard/PIB/WRAP_PIB.xlsx', sheet_name='Sheet1')
    uPiB4_CL = pd.read_excel('/home/yche14/PET_cycleGAN/data_PET/unpaired/standard/PIB/WADRC_PIB.xlsx', sheet_name='Sheet1')
    uPiB5_CL = pd.read_excel('/home/yche14/PET_cycleGAN/data_PET/unpaired/standard/PIB/CLPIB_CL.xlsx', sheet_name='Sheet1')
    uFBP_CL = pd.read_excel(uFBP_path, sheet_name='Demo')
    
    p_CL = pd.read_excel('/home/yche14/PET_cycleGAN/data_PET/paired/Centioid_Summary.xlsx', sheet_name='Sheet1')
    p_O_PIB_CL = pd.read_csv('/home/yche14/PET_cycleGAN/data_PET/paired/standard/PIB/OASIS_PIB_PUP_CL_selected.csv')
    p_O_FBP_CL = pd.read_csv('/home/yche14/PET_cycleGAN/data_PET/paired/standard/FBP/OASIS_FBP_CL_selected.csv')
    
    # QC
    adni_path = '/home/yche14/PET_cycleGAN/data_PET/unpaired/standard/FBP_QC/ADNI_FBP_QC_2024.xlsx'
    aibl_path = '/home/yche14/PET_cycleGAN/data_PET/unpaired/standard/PIB_QC/AIBL_PIB_QC.xlsx'
    cl_path = '/home/yche14/PET_cycleGAN/data_PET/unpaired/standard/PIB_QC/CLPIB_CL_QC.xlsx'
    oasis_path = '/home/yche14/PET_cycleGAN/data_PET/unpaired/standard/PIB_QC/OASIS_PIB_PUP_QC.xlsx'
    wadrc_path = '/home/yche14/PET_cycleGAN/data_PET/unpaired/standard/PIB_QC/WADRC_SUVR_QC.xlsx'
    wrap_path = '/home/yche14/PET_cycleGAN/data_PET/unpaired/standard/PIB_QC/WRAP_SUVR_QC.xlsx'

    adni_QC = pd.read_excel(adni_path, sheet_name='Sheet1')[['PUP ID', 'VQCError 2']]
    adni_QC = adni_QC[adni_QC['VQCError 2'].notnull() & adni_QC['PUP ID'].notnull()]
    aibl_QC = pd.read_excel(aibl_path, sheet_name='Sheet1')[['PUPID', 'VQC_Error 2']]
    cl_QC = pd.read_excel(cl_path, sheet_name='Sheet1')[['ID', 'VQC_Error']]
    oasis_QC = pd.read_excel(oasis_path, sheet_name='Summary')[['PIB_ID', 'VQC_Error']]
    wadrc_QC = pd.read_excel(wadrc_path, sheet_name='WADRC_SUVR')[['ID', 'VQC_Error']]
    wrap_QC = pd.read_excel(wrap_path, sheet_name='Demo')[['ID', 'VQC_Error']]
    
    # # sep. SUVR dataset
    # uPiB1_sep_path = '/data/amciilab/processedDataset/OASIS/Oasis-PUP/results/PiB_IDS_SUVRLR.csv'
    # uPiB2_sep_path = '/data/amciilab/processedDataset/AIBL/AIBL-PUP/results/PiB_IDS_SUVRLR.csv'
    # uFBP_sep_path = '/data/amciilab/processedDataset/ADNI/ADNI-PUP/results/FBP/list_ADNI_FBP_SUVRLR.csv'
    # pFBP_sep_path = '/data/amciilab/processedDataset/Centiloid/FBP_PIB/PUP_FBP/runExtract/list_id_SUVRLR.csv'
    # pPiB_sep_path = '/data/amciilab/processedDataset/Centiloid/FBP_PIB/PUP_PIB/runExtract/list_id_SUVRLR.csv'
    # # non sep. RSF dataset
    # uPiB1_RSF_path = '/data/amciilab/processedDataset/AIBL/AIBL-PUP/results/PiB_IDS_SUVR_RSF.csv'
    # uPiB2_RSF_path = '/data/amciilab/processedDataset/OASIS/Oasis-PUP/results/PiB_IDS_SUVR_RSF.csv'
    # uPiB3_RSF_path = '/data/amciilab/ysu/Wisconsin/WRAP/PUP_PIB_WRAP/SUVR_results/ID_list_SUVR_RSF.csv'
    # uPiB4_RSF_path = '/data/amciilab/ysu/Wisconsin/ADRC/PUP_PIB_WADRC/SUVR_results/ID_list_SUVR_RSF.csv'
    # uPiB5_RSF_path = './data_PET/CLPIB_SUVR_RSF.csv'
    # pPiB_RSF_path = '/data/amciilab/processedDataset/Centiloid/FBP_PIB/PUP_PIB/runExtract/list_id_SUVR_RSF.csv'
    # pFBP_RSF_path = '/data/amciilab/processedDataset/Centiloid/FBP_PIB/PUP_FBP/runExtract/list_id_SUVR_RSF.csv'
    
    
    uPiB1 = pd.read_excel(uPiB1_path, sheet_name='AIBL_PIB_PUP')
    uPiB1 = uPiB1.merge(aibl_QC, left_on='ID', right_on='PUPID')
    uPiB1 = uPiB1[uPiB1['VQC_Error 2']==0]
    uPiB1_CL = uPiB1_CL[uPiB1_CL['SID'].isin(uPiB1['ID'])]
    
    assert uPiB1['ID'].equals(uPiB1_CL['SID'])
    
    uPiB2 = pd.read_csv(uPiB2_path)
    uPiB2 = uPiB2.merge(oasis_QC, left_on='ID', right_on='PIB_ID')
    uPiB2 = uPiB2[uPiB2['VQC_Error']==0]
    uPiB2_CL = uPiB2_CL[uPiB2_CL['PIB_ID'].isin(uPiB2['ID'])]
    
    assert uPiB2['ID'].equals(uPiB2_CL['PIB_ID'])
    
    
    uPiB3 = pd.read_csv(uPiB3_path)
    uPiB3 = uPiB3.merge(wrap_QC, left_on='ID', right_on='ID')
    uPiB3 = uPiB3[uPiB3['VQC_Error']==0]
    uPiB3_CL = uPiB3_CL[uPiB3_CL['ID'].isin(uPiB3['ID'])]
    
    assert uPiB3['ID'].equals(uPiB3_CL['ID'])
    
    uPiB4 = pd.read_csv(uPiB4_path)
    uPiB4 = uPiB4.merge(wadrc_QC, left_on='ID', right_on='ID')
    uPiB4 = uPiB4[uPiB4['VQC_Error']==0]
    uPiB4_CL = uPiB4_CL[uPiB4_CL['ID'].isin(uPiB4['ID'])]
    
    assert uPiB4['ID'].equals(uPiB4_CL['ID'])
    
    uPiB5 = pd.read_csv(uPiB5_path)
    uPiB5 = uPiB5.merge(cl_QC, left_on='ID', right_on='ID')
    uPiB5 = uPiB5[uPiB5['VQC_Error']==0]
    uPiB5_CL = uPiB5_CL[uPiB5_CL['ID'].isin(uPiB5['ID'])]
    
    assert uPiB5['ID'].equals(uPiB5_CL['ID'])
    
    uFBP = pd.read_excel(uFBP_path, sheet_name='ALL_AV45_PUP_BAI_SUVR')
    uFBP = uFBP.merge(adni_QC, left_on='PUP ID', right_on='PUP ID')
    uFBP = uFBP[uFBP['VQCError 2']==0]
    uFBP_CL = uFBP_CL[uFBP_CL['PUP ID'].isin(uFBP['PUP ID'])]
    
    assert uFBP['PUP ID'].equals(uFBP_CL['PUP ID'])
    
    pPiB1 = pd.read_csv(pPiB1_path)
    pPiB2 = pd.read_csv(pPiB2_path)
    pFBP1 = pd.read_csv(pFBP1_path)
    pFBP2 = pd.read_csv(pFBP2_path)
    
    pWeight, uWeight = get_vox_weight()

    # read CL
    uPiB_CL = pd.concat([uPiB1_CL['CL'], uPiB2_CL['CL'], uPiB3_CL['CL'], uPiB4_CL['CL'], uPiB5_CL['CL']], axis=0)
    uFBP_CL = uFBP_CL['CL']
    pPiB_CL = pd.concat([p_CL['PIB_CL'],p_O_PIB_CL['CL']], axis=0)
    pFBP_CL = pd.concat([p_CL['FBP_CL'],p_O_FBP_CL['FBP_CL']], axis=0)
        
    # remove ID column and other columns that are not needed
    uPiB1 = uPiB1.iloc[:, 1:90]
    uPiB2 = uPiB2.iloc[:, 1:90]
    uPiB3 = uPiB3.iloc[:, 1:90]
    uPiB4 = uPiB4.iloc[:, 1:90]
    uPiB5 = uPiB5.iloc[:, 1:90]
    
    uFBP = uFBP.iloc[:, 1:90]
    pPiB1 = pPiB1.iloc[:, 1:90]
    pPiB2 = pPiB2.iloc[:, 1:90]
    pFBP1 = pFBP1.iloc[:, 1:90]
    pFBP2 = pFBP2.iloc[:, 1:90]
   
    uPiB = pd.concat([uPiB1, uPiB2, uPiB3, uPiB4, uPiB5], axis=0)
    pPiB = pd.concat([pPiB1, pPiB2], axis=0)
    pFBP = pd.concat([pFBP1, pFBP2], axis=0)
    
    # # adding CL to unpaired dataset
    # uPiB = pd.concat([uPiB, uPiB_CL/uPiB_CL.max()], axis=1)
    # uFBP = pd.concat([uFBP, uFBP_CL/uFBP_CL.max()], axis=1)
    # pPiB = pd.concat([pPiB, p_CL['PIB_CL']/p_CL['PIB_CL'].max()], axis=1)
    # pFBP = pd.concat([pFBP, p_CL['FBP_CL']/p_CL['FBP_CL'].max()], axis=1)
    
        
    # Drop columns with only one unique value
    names = []
    for i, name in enumerate(uFBP.columns):
        if len(uFBP[name].unique()) == 1:
            print(i, name) 
            names.append(name)
      
    
    uPiB = uPiB.drop(columns=names)
    uFBP = uFBP.drop(columns=names)
    pPiB = pPiB.drop(columns=names)
    pFBP = pFBP.drop(columns=names)
    
    print('uPiB shape:', uPiB.shape)
    print('uFBP shape:', uFBP.shape)
    print('pPiB shape:', pPiB.shape)
    print('pFBP shape:', pFBP.shape)
    
    # df to numpy
    uPiB = to_np_float(uPiB)
    uFBP = to_np_float(uFBP)
    pPiB = to_np_float(pPiB)
    pFBP = to_np_float(pFBP)
    uPiB_CL = to_np_float(uPiB_CL)
    uFBP_CL = to_np_float(uFBP_CL)
    pPiB_CL = to_np_float(pPiB_CL)
    pFBP_CL = to_np_float(pFBP_CL)
    
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
        
    #  if baseline:
    #     # baseline selection for uPiB2
    #     sorted_uPiB2 = uPiB2.sort_values(by='ID')
    #     sorted_uPiB2['ID_prefix'] = uPiB2['ID'].str.extract(r'^(OAS\d+_PIB)')
    #     sorted_uPiB2 = sorted_uPiB2.drop_duplicates(subset='ID_prefix', keep='first')
    #     uPiB2 = sorted_uPiB2.drop(columns=['ID_prefix'])
    #     # baseline selection for uPiB1
    #     uPiB1 = uPiB1[uPiB1['ID'].str.endswith('bl')]
    #     # baseline selection for uFBP
    #     sorted_uFBP = uFBP.sort_values(by=['RID', 'Actual Date'])
    #     uFBP = sorted_uFBP.drop_duplicates(subset='RID', keep='first')
    #     uFBP_CL = uFBP_CL[uFBP_CL['PUP ID'].isin(uFBP['PUP ID'])]['CL']
    #     uPiB1_CL_baseline = uPiB1_CL[uPiB1_CL['SID'].isin(uPiB1['ID'])]
    #     uPiB2_CL_baseline = uPiB2_CL[uPiB2_CL['PIB_ID'].isin(uPiB2['ID'])]
    #     uPiB_CL = pd.concat([uPiB1_CL_baseline['CL'], uPiB2_CL_baseline['CL']])
        
    return uPiB, uFBP, pPiB, pFBP, uPiB_CL, uFBP_CL, pPiB_CL, pFBP_CL, uPiB_scaler, uFBP_scaler, pWeight, uWeight
    
    
class PairedDataset(Dataset):
    def __init__(self, pFBP, pPiB):
        self.pFBP = pFBP
        self.pPiB = pPiB
    def __len__(self):
        return len(self.pFBP)
    def __getitem__(self, idx):
        return self.pFBP[idx], self.pPiB[idx]
    
class UnpairedDataset(Dataset):
    def __init__(self, uFBP, uPiB, uPiB_CL=None, uFBP_CL=None, uWeight=None, resample='matching'):
        self.uFBP = uFBP
        self.uPiB = uPiB
        self.uWeight = uWeight
        self.resample = resample
        
        if self.resample:
            assert uPiB_CL is not None and uFBP_CL is not None
            # Calculate histograms
            bins = np.histogram_bin_edges(np.concatenate([uPiB_CL, uFBP_CL]), bins=20)
            hist1, bin_edges1 = np.histogram(uPiB_CL, bins=bins)
            hist2, _ = np.histogram(uFBP_CL, bins=bins)
            
            assert self.resample in ['matching', 'resample_to_n', 'resample_tail', 'resample_CL_threshold']
            
            if self.resample == 'matching':
                idx_PiB, idx_FBP = distribution_matching(uPiB_CL, uFBP_CL, hist1, hist2, bin_edges1)
                self.resample_FBP = self.uFBP
                self.resample_PiB = self.uPiB
                
                if len(idx_PiB) > 0:
                    resample_PiB = self.uPiB[idx_PiB]
                    resample_uWeight = self.uWeight[idx_PiB]
                    self.resample_PiB = np.concatenate([self.uPiB, resample_PiB], axis=0)
                    self.resample_uWeight = np.concatenate([self.uWeight, resample_uWeight], axis=0)
                if len(idx_FBP) > 0:
                    resample_FBP = self.uFBP[idx_FBP]
                    self.resample_FBP = np.concatenate([self.uFBP, resample_FBP], axis=0)
                    
            elif self.resample == 'resample_to_n':
                idx_PiB, idx_FBP = resample_to_n(uPiB_CL, uFBP_CL, hist1, hist2, bin_edges1)
                self.resample_FBP = self.uFBP[idx_FBP]
                self.resample_PiB = self.uPiB[idx_PiB]
                self.resample_uWeight = self.uWeight[idx_PiB]
                
            elif self.resample == 'resample_tail':
                idx_PiB, idx_FBP = resample_tail(uPiB_CL, uFBP_CL, hist1, hist2, bin_edges1)
                self.resample_FBP = self.uFBP[idx_FBP]
                self.resample_PiB = self.uPiB[idx_PiB]
                self.resample_uWeight = self.uWeight[idx_PiB]
                
            elif self.resample == 'resample_CL_threshold':
                idx_PiB, idx_FBP = resample_CL_threshold(uPiB_CL, uFBP_CL, 30, greater=True)
                self.resample_FBP = self.uFBP[idx_FBP]
                self.resample_PiB = self.uPiB[idx_PiB]
                self.resample_uWeight = self.uWeight[idx_PiB]
            else:
                raise ValueError('Invalid resample method')
        else:
            self.resample_FBP = self.uFBP
            self.resample_PiB = self.uPiB
            self.resample_uWeight = self.uWeight
            
        self.len = max(len(self.resample_FBP), len(self.resample_PiB))    
        self.len1 = len(self.resample_FBP)
        self.len2 = len(self.resample_PiB)
    
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        return self.resample_FBP[idx%self.len1], self.resample_PiB[idx%self.len2], self.resample_uWeight[idx%self.len2]
    
def get_data_loaders(uPiB, uFBP, pPiB, pFBP, uPiB_CL, uFBP_CL, pWeight, uWeight,
                     batch_size_u, resample='matching', shuffle=True):
    
    uPiB = torch.from_numpy(uPiB).float()
    uFBP = torch.from_numpy(uFBP).float()
    uWeight = torch.from_numpy(uWeight).float()
    
    pPiB1 = torch.from_numpy(pPiB[:46]).float()
    pFBP1 = torch.from_numpy(pFBP[:46]).float()
    pPiB2 = torch.from_numpy(pPiB[46:]).float()
    pFBP2 = torch.from_numpy(pFBP[46:]).float()
    pWeight1 = torch.from_numpy(pWeight[:46]).float()
    pWeight2 = torch.from_numpy(pWeight[46:]).float()
    
    unpaired_dataset = UnpairedDataset(uFBP, uPiB, uPiB_CL, uFBP_CL, uWeight, resample=resample)
    unpaired_loader = DataLoader(unpaired_dataset, batch_size=batch_size_u, shuffle=shuffle)
    
    paired_dataset_O = (pFBP2, pPiB2, pWeight2) # val data from paired dataset O
    paired_dataset_C = (pFBP1, pPiB1, pWeight1) # test data from paired dataset C
    
    return paired_dataset_C, paired_dataset_O, unpaired_loader

if __name__ == "__main__":
    
    

    uPiB, uFBP, pPiB, pFBP, uPiB_CL, uFBP_CL, pPiB_CL, pFBP_CL, uPiB_scaler, uFBP_scaler, pWeight, uWeight = read_data(normalize=False)
    print(uPiB.shape, uFBP.shape, pPiB.shape, pFBP.shape)
    print(uPiB_CL.shape, uFBP_CL.shape)
    print(pWeight.shape, uWeight.shape)
    