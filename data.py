
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, Dataset

from pathlib import Path
import pandas as pd

def to_np_float(x):
    return x.to_numpy().astype(float)
    
def read_data(numpy=True):
    data_path = Path("./data")
    paired_data_path = data_path / "paired"
    unpaired_data_path = data_path / "unpaired"

    paired_blood = pd.read_csv(paired_data_path / "blood.complete.csv")
    paired_brain = pd.read_csv(paired_data_path / "brain.complete.csv")

    unpaired_blood = pd.read_csv(unpaired_data_path / "beta.harmonized.blood.csv")
    unpaired_brain = pd.read_csv(unpaired_data_path / "beta.harmonized.brain.csv")
    
    paired_brain_T = paired_brain.transpose()
    paired_brain_T.columns = paired_brain_T.iloc[0]
    paired_brain_T = paired_brain_T.drop(paired_brain_T.index[0])
    
    paired_blood_T = paired_blood.transpose()
    paired_blood_T.columns = paired_blood_T.iloc[0]
    paired_blood_T = paired_blood_T.drop(paired_blood_T.index[0])
    
    unpaired_blood_T = unpaired_blood.transpose()
    unpaired_blood_T.columns = unpaired_blood_T.iloc[0]
    unpaired_blood_T = unpaired_blood_T.drop(unpaired_blood_T.index[0])
    
    unpaired_brain_T = unpaired_brain.transpose()
    unpaired_brain_T.columns = unpaired_brain_T.iloc[0]
    unpaired_brain_T = unpaired_brain_T.drop(unpaired_brain_T.index[0])
    
    # find the shared columns in the paired and unpaired data and keep only those columns
    shared_columns = paired_brain_T.columns.intersection(unpaired_brain_T.columns)
    # select only the shared columns
    paired_brain_T = paired_brain_T[shared_columns]
    unpaired_brain_T = unpaired_brain_T[shared_columns]
    paired_blood_T = paired_blood_T[shared_columns]
    unpaired_blood_T = unpaired_blood_T[shared_columns]
     
    # shape check
    assert paired_brain_T.shape == paired_blood_T.shape == (347, 47)
    assert unpaired_brain_T.shape == (2545, 47)
    assert unpaired_blood_T.shape == (9462, 47)
    if numpy:
        return to_np_float(paired_brain_T), to_np_float(paired_blood_T), to_np_float(unpaired_brain_T), to_np_float(unpaired_blood_T)
    else:
        return paired_brain_T, paired_blood_T, unpaired_brain_T, unpaired_blood_T

class PairedDataset(Dataset):
    def __init__(self, paired_brain, paired_blood):
        self.paired_brain = paired_brain
        self.paired_blood = paired_blood
        
    def __len__(self):
        return len(self.paired_brain)
    
    def __getitem__(self, idx):
        return self.paired_brain[idx], self.paired_blood[idx]
    
class UnpairedDataset(Dataset):
    def __init__(self, unpaired_brain, unpaired_blood):
        self.unpaired_brain = unpaired_brain
        self.unpaired_blood = unpaired_blood
        self.len_unpaired_brain = len(unpaired_brain)
        self.len_unpaired_blood = len(unpaired_blood)
        self.max_len = max(self.len_unpaired_brain, self.len_unpaired_blood) 
    def __len__(self):
        return self.max_len
    
    def __getitem__(self, idx):
        return self.unpaired_brain[idx%self.len_unpaired_brain], self.unpaired_blood[idx%self.len_unpaired_blood]



def get_data_loaders(batch_size_pa, batch_size_un):
    paired_brain, paired_blood, unpaired_brain, unpaired_blood = read_data()
    
    paired_brain = torch.from_numpy(paired_brain).float()
    paired_blood = torch.from_numpy(paired_blood).float()
    unpaired_brain = torch.from_numpy(unpaired_brain).float()
    unpaired_blood = torch.from_numpy(unpaired_blood).float() 
    
    # create paired dataset
    paired_dataset = PairedDataset(paired_brain, paired_blood)
    paired_loader = DataLoader(paired_dataset, batch_size=batch_size_pa, shuffle=False)
    
    # create unpaired dataset
    unpaired_dataset = UnpairedDataset(unpaired_brain, unpaired_blood)
    unpaired_loader = DataLoader(unpaired_dataset, batch_size=batch_size_un, shuffle=True)
    
    return paired_loader, unpaired_loader

def get_unpaired_blood():
    _, _, _, unpaired_blood = read_data()
    unpaired_blood = torch.from_numpy(unpaired_blood).float()
    return unpaired_blood

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
        
    unpaired_blood = get_unpaired_blood()
    print(unpaired_blood.shape)
    