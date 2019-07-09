import os
from datetime import datetime
from hd_model import HDModel
from dataset import HDDataset
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# import torch.nn.utils
# import torch.nn.functional as F


logdir = 'runs/HDModel-' + datetime.now().strftime('%y%m%d-%H%M%S')
if not os.path.exists(logdir):
	os.makedirs(logdir)
n_epochs = 2000
checkpoint_interval = 10
lr=0.01

batch_size = 64
input_size = 299

best_loss = -1



use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

model = HDModel(input_size=input_size)
model.to(device)


# Define Loss, Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

print('-------------------- loading training data ----------------------')
dataset = HDDataset(path='data', group='train')
training_generator = DataLoader(dataset, batch_size, shuffle=True)

print('-------------------- training ----------------------')
# Training Run
for epoch in tqdm(range(n_epochs)):
	optimizer.zero_grad() # Clears existing gradients from previous epoch
	for local_x, local_a, local_f, local_y in training_generator:
		local_x = local_x.to(device)
		local_a = local_a.to(device)
		local_f = local_f.to(device)
		local_y = local_y.to(device)
		output = model(local_x, local_a, local_f)
		loss = criterion(output.view(-1), local_y)
		loss.backward() # Does backpropagation and calculates gradients
		optimizer.step() # Updates the weights accordingly
	
	if epoch % checkpoint_interval == 0 and epoch != 0:
		print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
		print("Loss: {:.4f}".format(loss.item()))
		torch.save(model, os.path.join(logdir, f'model-{epoch}.pt'))
		torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))
	
	# if loss.item() < best_loss or best_loss == -1:
	# 	best_loss = loss.item()
	# 	torch.save(model, os.path.join(logdir, f'model-{epoch}.pt'))
 #        torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))

	# if i % checkpoint_interval == 0:
			



