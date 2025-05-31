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

def get_data_loaders(batch_size, train_size=1000):
    data_FBP = read_data("data_PET/test3/MSUVRFBP_V3.csv")
    data_PIB = read_data("data_PET/test3/MSUVRPIB_V3.csv")
    CL = pd.read_csv("data_PET/test2/Sim2NI.csv")["CL"].to_numpy()
    
    FBP_train, FBP_test, PIB_train, PIB_test, CL_train, CL_test = train_test_split(data_FBP, data_PIB, CL, test_size=0.05)
   
    # import matplotlib.pyplot as plt
    # # Visualize the distribution of CL
    # plt.hist(CL_train, bins=50, alpha=0.5, label='CL Train Distribution')
    # plt.hist(CL_train[skewed_data_index], bins=50, alpha=0.5, label='CL Skewed Distribution')
    # plt.savefig("CL_distribution.png")
    
    
    # Method 1: Randomly select a percentage of the training data
    # train_size = len(FBP_train) * train_percent
    # random pick train_size samples from the training data
    random_indices_FBP = np.random.choice(len(FBP_train), size=int(train_size), replace=False)
    random_indices_PIB = np.random.choice(len(PIB_train), size=int(train_size), replace=False)
    
    #check if the two random indices are the same
    assert not np.array_equal(random_indices_FBP, random_indices_PIB), "Random indices for FBP and PIB should not be the same."
    
    FBP_train = FBP_train[random_indices_FBP]
    PIB_train = PIB_train[random_indices_PIB]
    
    # Method 2: Use the first N samples from the training data
    # train_size = len(FBP_train) // 2
    # FBP_train = FBP_train[:int(train_size)]
    # PIB_train = PIB_train[int(train_size):]
    
    # Method 3: Skew the training data based on CL values   
    # transformed = abs(CL_train)
    # transformed[transformed > 20] = transformed[transformed > 20] + 200
    # weights = np.exp(-transformed * 0.01)  # Exponential decay for skewness
    # weights /= weights.sum()
    
    # train_size = int(len(FBP_train) * train_percent)
    # skewed_data_index = np.random.choice(len(CL_train), size=train_size, p=weights, replace=False)
    
    # import matplotlib.pyplot as plt
    # # Visualize the distribution of CL
    # plt.hist(CL_train, bins=50, alpha=0.5, label='CL Train Distribution')
    # plt.hist(CL_train[skewed_data_index], bins=50, alpha=0.5, label='CL Skewed Distribution')
    # plt.legend()
    # plt.savefig(f"./mis/CL_distribution_{train_size}.png")
    
    # FBP_train = FBP_train[skewed_data_index]
    # PIB_train = PIB_train[skewed_data_index]
    # # shuffle the training data
    # np.random.shuffle(FBP_train)
    # np.random.shuffle(PIB_train)
    # assert not np.array_equal(FBP_train, PIB_train), "FBP and PIB training data should not be the same."
    
    
    # Convert to torch tensors
    FBP_train = torch.from_numpy(FBP_train).float()
    PIB_train = torch.from_numpy(PIB_train).float()
    FBP_test = torch.from_numpy(FBP_test).float()
    PIB_test = torch.from_numpy(PIB_test).float()
    
    print(f"FBP_train shape: {FBP_train.shape}, PIB_train shape: {PIB_train.shape}")
    
    val_num = int(len(FBP_test) // 2)
    FBP_val = FBP_test[:val_num]
    PIB_val = PIB_test[:val_num]
    FBP_test = FBP_test[val_num:]
    PIB_test = PIB_test[val_num:]
    
    train_dataset = PairedDataset(FBP_train, PIB_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = (FBP_test, PIB_test)
    val_dataset = (FBP_val, PIB_val)
    return test_dataset, val_dataset, train_loader, (FBP_train, PIB_train)


if __name__ == "__main__":
    # Load the data
    data_FBP = read_data("data_PET/test3/MSUVRFBP_V3.csv")
    data_PIB = read_data("data_PET/test3/MSUVRPIB_V3.csv")
    # Get the MCSUVR
    MCSUVR = cal_MCSUVR_torch(data_FBP)
    print(MCSUVR.shape)
    
    test_dataset, val_dataset, train_loader, train_dataset = get_data_loaders(batch_size=32, train_size=100)
    # Print the shape of the training data
    for i, (FBP, PIB) in enumerate(train_loader):
        print(FBP.shape, PIB.shape)
        if i == 1:
            break
    # Print the shape of the test data
    FBP_test, PIB_test = test_dataset
    print(FBP_test.shape, PIB_test.shape)