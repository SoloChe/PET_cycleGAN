import pandas as pd
import numpy as np
import torch

#%% MCSUVR V1 not efficient enough; based on sample ID
def get_pID_df():
    pPiB_path = '/data/amciilab/processedDataset/Centiloid/FBP_PIB/PUP_PIB/runExtract/list_id_SUVR.csv'
    pPiB_raw = pd.read_csv(pPiB_path)
    pPiB = pPiB_raw.iloc[:, 1:90]
    
    names = []
    for i, name in enumerate(pPiB.columns):
        if len(pPiB[name].unique()) == 1:
            names.append(name)
    print(f'dropped columns: {names}')
    pPiB_df = pPiB.drop(columns=names)
    ID = pPiB_raw['ID']
    return pPiB_raw, pPiB_df, ID
    
def tensor_to_df(fake_PiB, ID, pPiB_df):
    if isinstance(fake_PiB, torch.Tensor):
        fake_PiB = fake_PiB.cpu().detach().numpy().astype(np.float32)
    fake_PiB_df = pPiB_df.copy()
    # add ID
    fake_PiB_df['ID'] = ID
    fake_PiB_df['ID'] = fake_PiB_df['ID'].astype(str)
    # put ID the first column
    cols = fake_PiB_df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    fake_PiB_df = fake_PiB_df[cols]
    fake_PiB_df.iloc[:,1:] = fake_PiB
    return fake_PiB_df
    
def load_weights(separate=False):
    weight_path = './data_PET/RegionalCycleGAN.xlsx'
    weight_raw = pd.read_excel(weight_path, sheet_name='Sheet1') 

    col_name = 'ctx-precuneus'
    val = [1 for _ in range(46)] 
    weight_raw[col_name] = val
    
    regions = {'PREC':['ctx-precuneus'], 'PREF':['ctx-rostralmiddlefrontal', 'ctx-superiorfrontal'], 'TEMP':['ctx-middletemporal', 'ctx-superiortemporal'], 'GR':['ctx-lateralorbitofrontal', 'ctx-medialorbitofrontal']}
    
    if not separate:
        region_index = {'ctx-precuneus':39, 
                    'ctx-rostralmiddlefrontal':41, 
                    'ctx-superiorfrontal':42, 
                    'ctx-middletemporal':29, 
                    'ctx-superiortemporal':44, 
                    'ctx-lateralorbitofrontal':26, 
                    'ctx-medialorbitofrontal':28}
    else:
        region_index = {'right':{'ctx-precuneus': 73,
                        'ctx-rostralmiddlefrontal': 77,
                        'ctx-superiorfrontal': 79,
                        'ctx-middletemporal': 53,
                        'ctx-superiortemporal': 83,
                        'ctx-lateralorbitofrontal': 47,
                        'ctx-medialorbitofrontal': 51},
                        
                        'left':{'ctx-precuneus': 72,
                        'ctx-rostralmiddlefrontal': 76,
                        'ctx-superiorfrontal': 78,
                        'ctx-middletemporal': 52,
                        'ctx-superiortemporal': 82,
                        'ctx-lateralorbitofrontal': 46,
                        'ctx-medialorbitofrontal': 50}}
    
    return weight_raw, regions, region_index

def cal_MCSUVR(pPiB_raw, fake_PiB_df):
    weight_raw, regions = load_weights()
    region_value_real = {'PREC':[], 'PREF':[], 'TEMP':[], 'GR':[]}
    region_value_fake = {'PREC':[], 'PREF':[], 'TEMP':[], 'GR':[]}

    for ID in weight_raw['ID']:
        id = ID.strip().split('/')[7]
        
        scan_real = pPiB_raw[pPiB_raw['ID'] == id]
        scan_fake = fake_PiB_df[fake_PiB_df['ID'] == id]        
        scan_weight = weight_raw[weight_raw['ID'] == ID]
        
        for key in regions.keys():
            rv_scan_real = 0
            rv_scan_fake = 0
            w = 0
                
            for region in regions[key]:
                w += scan_weight[region].values[0]
                rv_scan_real += scan_real[region].values[0] * scan_weight[region].values[0]
                rv_scan_fake += scan_fake[region].values[0] * scan_weight[region].values[0]
                    
            region_value_real[key].append(rv_scan_real/(w+1e-8))
            region_value_fake[key].append(rv_scan_fake/(w+1e-8))
    
    REAL_MCSUVR = (np.array(region_value_real['PREC']) + np.array(region_value_real['PREF']) + np.array(region_value_real['TEMP']) + np.array(region_value_real['GR'])) / 4
    FAKE_MCSUVR = (np.array(region_value_fake['PREC']) + np.array(region_value_fake['PREF']) + np.array(region_value_fake['TEMP']) + np.array(region_value_fake['GR'])) / 4
    return REAL_MCSUVR, FAKE_MCSUVR

#%% MCSUVR V2 with torch
def cal_MCSUVR_torch(data, region_to_index, weight, separate=False):
    
    ctx_precuneus_w = torch.from_numpy(weight['ctx-precuneus'].values)
    ctx_rostralmiddlefrontal_w = torch.from_numpy(weight['ctx-rostralmiddlefrontal'].values)
    ctx_superiorfrontal_w = torch.from_numpy(weight['ctx-superiorfrontal'].values)
    ctx_middletemporal_w = torch.from_numpy(weight['ctx-middletemporal'].values)
    ctx_superiortemporal_w = torch.from_numpy(weight['ctx-superiortemporal'].values)
    ctx_lateralorbitofrontal_w = torch.from_numpy(weight['ctx-lateralorbitofrontal'].values)
    ctx_medialorbitofrontal_w = torch.from_numpy(weight['ctx-medialorbitofrontal'].values)
    
    if not separate:
        # PREC 'ctx-precuneus'
        idx_ctx_precuneus = region_to_index['ctx-precuneus']
        # PREF 'ctx-rostralmiddlefrontal', 'ctx-superiorfrontal'
        idx_ctx_rostralmiddlefrontal = region_to_index['ctx-rostralmiddlefrontal']
        idx_ctx_superiorfrontal = region_to_index['ctx-superiorfrontal']
        # TEMP 'ctx-middletemporal', 'ctx-superiortemporal'
        idx_ctx_middletemporal = region_to_index['ctx-middletemporal']
        idx_ctx_superiortemporal = region_to_index['ctx-superiortemporal']
        # GR 'ctx-lateralorbitofrontal', 'ctx-medialorbitofrontal'
        idx_ctx_lateralorbitofrontal = region_to_index['ctx-lateralorbitofrontal']
        idx_ctx_medialorbitofrontal = region_to_index['ctx-medialorbitofrontal']
    
    
        PREC = (ctx_precuneus_w * data[:, idx_ctx_precuneus]) / (ctx_precuneus_w + 1e-8)
        PREF = (ctx_rostralmiddlefrontal_w * data[:, idx_ctx_rostralmiddlefrontal]  + ctx_superiorfrontal_w * data[:, idx_ctx_superiorfrontal]) / (ctx_superiorfrontal_w + ctx_rostralmiddlefrontal_w + 1e-8)
        TEMP = (ctx_middletemporal_w * data[:, idx_ctx_middletemporal] + ctx_superiortemporal_w * data[:, idx_ctx_superiortemporal]) / (ctx_middletemporal_w + ctx_superiortemporal_w + 1e-8)
        GR = (ctx_lateralorbitofrontal_w * data[:, idx_ctx_lateralorbitofrontal] + ctx_medialorbitofrontal_w * data[:, idx_ctx_medialorbitofrontal]) / (ctx_lateralorbitofrontal_w + ctx_medialorbitofrontal_w + 1e-8)
        MCSUVR = (PREC + PREF + TEMP + GR) / 4
    else:
        region_to_index_left = region_to_index['left']
        region_to_index_right = region_to_index['right']
        
        idx_ctx_lateralorbitofrontal_left = region_to_index_left['ctx-lateralorbitofrontal']
        idx_ctx_medialorbitofrontal_left = region_to_index_left['ctx-medialorbitofrontal']
        idx_ctx_middletemporal_left = region_to_index_left['ctx-middletemporal']
        idx_ctx_precuneus_left = region_to_index_left['ctx-precuneus']
        idx_ctx_rostralmiddlefrontal_left = region_to_index_left['ctx-rostralmiddlefrontal']
        idx_ctx_superiorfrontal_left = region_to_index_left['ctx-superiorfrontal']
        idx_ctx_superiortemporal_left = region_to_index_left['ctx-superiortemporal']
        
        idx_ctx_lateralorbitofrontal_right = region_to_index_right['ctx-lateralorbitofrontal']
        idx_ctx_medialorbitofrontal_right = region_to_index_right['ctx-medialorbitofrontal']
        idx_ctx_middletemporal_right = region_to_index_right['ctx-middletemporal']
        idx_ctx_precuneus_right = region_to_index_right['ctx-precuneus']
        idx_ctx_rostralmiddlefrontal_right = region_to_index_right['ctx-rostralmiddlefrontal']
        idx_ctx_superiorfrontal_right = region_to_index_right['ctx-superiorfrontal']
        idx_ctx_superiortemporal_right = region_to_index_right['ctx-superiortemporal']
        
        PREC = (ctx_precuneus_w * (data[:, idx_ctx_precuneus_right]+data[:, idx_ctx_precuneus_left])/2) / (ctx_precuneus_w + 1e-8)
        PREF = (ctx_rostralmiddlefrontal_w * (data[:, idx_ctx_rostralmiddlefrontal_right]+data[:, idx_ctx_rostralmiddlefrontal_left])/2 + ctx_superiorfrontal_w * (data[:, idx_ctx_superiorfrontal_right]+data[:, idx_ctx_superiorfrontal_left])/2) / (ctx_superiorfrontal_w + ctx_rostralmiddlefrontal_w + 1e-8)
        TEMP = (ctx_middletemporal_w * (data[:, idx_ctx_middletemporal_right]+data[:, idx_ctx_middletemporal_left])/2 + ctx_superiortemporal_w * (data[:, idx_ctx_superiortemporal_right]+data[:, idx_ctx_superiortemporal_left])/2) / (ctx_middletemporal_w + ctx_superiortemporal_w + 1e-8)
        GR = (ctx_lateralorbitofrontal_w * (data[:, idx_ctx_lateralorbitofrontal_right]+data[:, idx_ctx_lateralorbitofrontal_left])/2 + ctx_medialorbitofrontal_w * (data[:, idx_ctx_medialorbitofrontal_right]+data[:, idx_ctx_medialorbitofrontal_left])/2) / (ctx_lateralorbitofrontal_w + ctx_medialorbitofrontal_w + 1e-8)
        MCSUVR = (PREC + PREF + TEMP + GR) / 4
        
    return MCSUVR

#%% correlation between real and fake MCSUVR
def cal_correlation(X, Y):
    mean_X = np.mean(X)
    mean_Y = np.mean(Y)
    # Numerator of the Pearson correlation formula
    numerator = sum((x - mean_X) * (y - mean_Y) for x, y in zip(X, Y))
    # Denominator of the Pearson correlation formula
    denominator = np.sqrt(sum((x - mean_X) ** 2 for x in X) * sum((y - mean_Y) ** 2 for y in Y))
    # Pearson correlation coefficient
    r = numerator / denominator
    return r