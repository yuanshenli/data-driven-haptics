import os
import sys
from datetime import datetime
from tqdm import tqdm

from hd_model import HDModel
from dataset import HDDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# import torch.nn.utils
# import torch.nn.functional as F


logdir = 'runs/HDModel-190711-210401'

resume_epochs = 2
n_epochs = 10000
validation_interval = 50

lr=0.0006
batch_size = 64

loss_sum = 0
loss_iter = 0

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_path = os.path.join(logdir, f'model-{resume_epochs}.pt')
model = torch.load(model_path)
criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# optimizer.load_state_dict(torch.load(os.path.join(logdir, 'last-optimizer-state.pt')))

print('---- loading training data ------')
train_set = HDDataset(path='data_chirp', group='train')
training_generator = DataLoader(train_set, batch_size, shuffle=False)


print('---- loading validation data ----')
validation_set = HDDataset(path='data_chirp', group='validation')
validation_generator = DataLoader(validation_set, len(validation_set), shuffle=False)



# loss on training data
model.eval()
for local_x, local_a, local_f, local_y in training_generator:
    print('---- training batch-------------------')
    # optimizer.zero_grad() # Clears existing gradients from previous epoch
    local_x = local_x.to(device)
    local_a = local_a.to(device)
    local_f = local_f.to(device)
    local_y = local_y.to(device)
    output = model(local_x, local_a, local_f)
    for idx in range(len(local_y)):
        print("{:.1f}, {:.1f}".format(output.view(-1)[idx], local_y[idx]))
    # print(local_y)
    loss = criterion(output.view(-1), local_y)
    loss_sum += loss.item()
    loss_iter += 1

   

# loss on validation data
validation_output_file_name = logdir + '/validation_out.log'
with open(validation_output_file_name, 'w') as validation_output_file:
    for validation_x, validation_a, validation_f, validation_y in validation_generator:
        print('---- validation batch-------------------') 
        validation_x = validation_x.to(device)
        validation_a = validation_a.to(device)
        validation_f = validation_f.to(device)
        validation_y = validation_y.to(device)
        validation_output = model(validation_x, validation_a, validation_f)
        for idx in range(len(validation_y)):
            print("{:.1f}, {:.1f}".format(validation_output.view(-1)[idx], validation_y[idx]))
            validation_output_file.write('%.1f, %.1f\n' % (validation_output.view(-1)[idx], validation_y[idx]))
        
        val_loss = criterion(validation_output.view(-1), validation_y)
           
validation_output_file.close()


# print("Validation loss: {:.4f}".format(val_loss.item()))
print("Training loss: {:.4f}; Validation loss: {:.4f}".format(loss_sum/loss_iter, val_loss.item()))





            



