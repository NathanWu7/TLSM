import torch
from torch.utils.data import Dataset
import os
import numpy as np
from math import exp
import h5py
from tqdm import tqdm

def reparameterize(mu, logsigma):
    std = torch.exp(0.5*logsigma)
    eps = torch.randn_like(std)
    return mu + eps*std


# for representation learning
class TactilePairDataset(Dataset):
    def __init__(self,file_dir, file_source, file_goal, transform):
        self.train_data = []
        self.transform = transform
        self.files = [file_source,file_goal]
        print(self.files)
        self.length = 0
        for file in self.files:
            h5f_data = h5py.File(os.path.join(os.path.dirname(__file__), file_dir,file,'data_vae_pair.h5'), "r")
            self.length = len(h5f_data)
            np_data = np.zeros((len(h5f_data), 128, 128), dtype=np.uint8)
            data = []
            for i in tqdm(range(self.length)):
                dset = h5f_data["data_"+str(i)]
                np_data[i] = dset[:]
            np_data = np_data  / 255   
            print(np_data.shape)
            self.train_data.append(np_data)  
            
    def __len__(self):
        return self.length
        
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return torch.stack([self.transform(d[idx]) for d in self.train_data])  

class TactileDataset(Dataset):
    def __init__(self,file_dir, tag, transform):
        self.train_data = []
        self.transform = transform
        self.files = [f for f in os.listdir(file_dir) if tag in f]
        print(self.files)
        self.length = 0
        for file in self.files:
            h5f_data = h5py.File(os.path.join(os.path.dirname(__file__), file_dir,file,'data_vae_all.h5'), "r")
            self.length = len(h5f_data)
            np_data = np.zeros((len(h5f_data), 128, 128), dtype=np.uint8)
            data = []
            for i in tqdm(range(self.length)):
                dset = h5f_data["data_"+str(i)]
                np_data[i] = dset[:]
            np_data = np_data  / 255   
            print(np_data.shape)
            self.train_data = np_data 
            
    def __len__(self):
        return self.length
        
    def __getitem__(self,idx):

        return torch.tensor(self.train_data[idx])  