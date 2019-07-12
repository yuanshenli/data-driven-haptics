import os
from torch.utils.data import Dataset, TensorDataset
import numpy as np
from tqdm import tqdm

class HDDataset(Dataset):
    def __init__(self, path='', group=None):
        self.path = path
        self.group = group
        data = []
        mypath = os.path.join(path, group)
        x = []
        y = []
        for idx, file in enumerate(tqdm(sorted(os.listdir(mypath)))):
            filename = os.path.join(mypath, file)
            # print(filename)
            this_x = []
            with open(filename) as f:
                for idx, line in enumerate(f):
                    line = line.split(",")
                    d0 = float(line[0]) # pos
                    d1 = float(line[1]) # force
                    d2 = float(line[2]) # acc
                    if idx != 299: 
                        this_x.append([d0, d1, d2])  
                    else:
                        y.append(d1)
            x.append(this_x)
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        
    def __getitem__(self, index):
        return (self.x[index,:,0], self.x[index,:,1], self.x[index,:,2], self.y[index])

    def __len__(self):
        return len(self.y)

