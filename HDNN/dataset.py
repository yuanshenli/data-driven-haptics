import os
from torch.utils.data import Dataset, TensorDataset
import numpy as np

class HDDataset(Dataset):
    def __init__(self, path, group=None):
        self.path = path
        self.group = group
        data = []
        mypath = os.join(group, path)
        for idx, file in enumerate(os.listdir(mypath)):
            data[idx] = np.loadtxt(os.join(mypath, file), delimiter=',')


        self.x = data[len(data.shape[0])-1:,:]
        self.y = data[len(data.shape[0]):,2]
        
    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)

# # Wait, is this a CPU tensor now? Why? Where is .to(device)?
# x_train_tensor = torch.from_numpy(x_train).float()
# y_train_tensor = torch.from_numpy(y_train).float()

# train_data = CustomDataset(x_train_tensor, y_train_tensor)
# print(train_data[0])

# train_data = TensorDataset(x_train_tensor, y_train_tensor)
# print(train_data[0])
