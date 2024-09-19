import pandas as pd
import numpy as np

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
    

def load_weights():
    weight_path = './data_PET/RegionalCycleGAN.xlsx'
    weight_raw = pd.read_excel(weight_path, sheet_name='Sheet1') 

    col_name = 'ctx-precuneus'
    val = [1 for _ in range(46)] 
    weight_raw[col_name] = val
    
    regions = {'PREC':['ctx-precuneus'], 'PREF':['ctx-rostralmiddlefrontal', 'ctx-superiorfrontal'], 'TEMP':['ctx-middletemporal', 'ctx-superiortemporal'], 'GR':['ctx-lateralorbitofrontal', 'ctx-medialorbitofrontal']}
    return weight_raw, regions

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