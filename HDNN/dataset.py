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
        for idx, file in enumerate(tqdm(os.listdir(mypath))):
            filename = os.path.join(mypath, file)
            # print(filename)
            this_x = []
            # this_y = []
            with open(filename) as f:
                for idx, line in enumerate(f):
                    line = line.split(",")
                    d0 = float(line[0])
                    d1 = float(line[1])
                    d2 = float(line[2])
                    if idx != 299: 
                        this_x.append([d0, d1, d2])
                    else:
                        y.append(d2)
            x.append(this_x)
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        
    def __getitem__(self, index):
        return (self.x[index,:,0], self.x[index,:,1], self.x[index,:,2], self.y[index])

    def __len__(self):
        return len(self.x)

# # Wait, is this a CPU tensor now? Why? Where is .to(device)?
# x_train_tensor = torch.from_numpy(x_train).float()
# y_train_tensor = torch.from_numpy(y_train).float()

# train_data = CustomDataset(x_train_tensor, y_train_tensor)
# print(train_data[0])

# train_data = TensorDataset(x_train_tensor, y_train_tensor)
# print(train_data[0])
