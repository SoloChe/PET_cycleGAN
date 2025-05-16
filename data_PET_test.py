import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from resample import to_np_float
from sklearn.model_selection import train_test_split

def cal_MCSUVR_torch(data: torch.Tensor) -> torch.Tensor:
    """
    Get the voxel weights for the PET data.
    Average column 40, 41, 43, 46
    """
    if not isinstance(data, torch.Tensor):
        data = torch.from_numpy(data)
    # Load the PET data
    columns = [39, 40, 42, 45]
    # Average the columns
    MCSUVR = torch.mean(data[:, columns], dim=1)
    return MCSUVR

def read_data(file_path: str) -> np.ndarray:
    """
    Read the data from the file.
    """
    # Load the data
    data = pd.read_csv(file_path, header=None)
    # Convert to numpy array
    data = to_np_float(data)
    return data

class PairedDataset(Dataset):
    def __init__(self, pFBP, pPiB):
        self.pFBP = pFBP
        self.pPiB = pPiB
    def __len__(self):
        return len(self.pFBP)
    def __getitem__(self, idx):
        return self.pFBP[idx], self.pPiB[idx]

def get_data_loaders(batch_size):
    data_FBP = read_data("data_PET/test/MSUVRFBP.csv")
    data_PIB = read_data("data_PET/test/MSUVRPIB.csv")
    FBP_train, FBP_test, PIB_train, PIB_test = train_test_split(data_FBP, data_PIB, test_size=0.2)
    # shuffle the training data
    np.random.shuffle(FBP_train)
    np.random.shuffle(PIB_train)
    # Convert to torch tensors
    FBP_train = torch.from_numpy(FBP_train).float()
    PIB_train = torch.from_numpy(PIB_train).float()
    FBP_test = torch.from_numpy(FBP_test).float()
    PIB_test = torch.from_numpy(PIB_test).float()
    
    val_num = int(len(FBP_test) // 2)
    FBP_val = FBP_test[:val_num]
    PIB_val = PIB_test[:val_num]
    FBP_test = FBP_test[val_num:]
    PIB_test = PIB_test[val_num:]
    
    train_dataset = PairedDataset(FBP_train, PIB_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = (FBP_test, PIB_test)
    val_dataset = (FBP_val, PIB_val)
    return test_dataset, val_dataset, train_loader


if __name__ == "__main__":
    # Load the data
    data_FBP = read_data("data_PET/test/MSUVRFBP.csv")
    data_PIB = read_data("data_PET/test/MSUVRPIB.csv")
    # Get the MCSUVR
    MCSUVR = cal_MCSUVR_torch(data_FBP)
    print(MCSUVR.shape)
    
    test_dataset, val_dataset, train_loader = get_data_loaders(batch_size=32)
    # Print the shape of the training data
    for i, (FBP, PIB) in enumerate(train_loader):
        print(FBP.shape, PIB.shape)
        if i == 1:
            break
    # Print the shape of the test data
    FBP_test, PIB_test = test_dataset
    print(FBP_test.shape, PIB_test.shape)