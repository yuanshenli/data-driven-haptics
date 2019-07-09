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
n_epochs = 10000
save_interval = 100
validation_interval = 50
lr=0.01

batch_size = 64
input_size = 299

best_loss = -1

epoch_tracker = []
training_loss_tracker = []
validation_loss_tracker = []

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

model = HDModel(input_size=input_size)
# if torch.cuda.device_count() > 1:
# 	model = torch.nn.DataParallel(model, device_ids=[0, 2])
model.to(device)


# Define Loss, Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

print('---- loading training data ------')
train_set = HDDataset(path='data', group='train')
training_generator = DataLoader(train_set, batch_size, shuffle=True)


print('---- loading validation data ----')
validation_set = HDDataset(path='data', group='validation')
validation_generator = DataLoader(validation_set, len(validation_set), shuffle=True)

print('---- training -------------------')
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
	
	if epoch % save_interval == 0 and epoch != 0:
		torch.save(model, os.path.join(logdir, f'model-{epoch}.pt'))
		torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))

	if epoch % validation_interval == 0 and epoch != 0:
		model.eval()

		for validation_x, validation_a, validation_f, validation_y in validation_generator:
			validation_x = local_x.to(device)
			validation_a = local_a.to(device)
			validation_f = local_f.to(device)
			validation_y = local_y.to(device)
			output = model(validation_x, validation_a, validation_f)
			val_loss = criterion(output.view(-1), validation_y)
		print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
		print("Training loss: {:.4f}; Validation loss: {:.4f}".format(loss.item(), val_loss.item()))
		training_loss_tracker.append(loss.item())
		validation_loss_tracker.append(val_loss.item())
		epoch_tracker.append(epoch)
		

		model.train()
	

	if loss.item() < best_loss or best_loss == -1:
		best_loss = loss.item()
		torch.save(model, os.path.join(logdir, f'best_model.pt'))
		torch.save(optimizer.state_dict(), os.path.join(logdir, 'best-model-optimizer-state.pt'))
print(training_loss_tracker)
print(validation_loss_tracker)
print(epoch_tracker)

	# if i % checkpoint_interval == 0:
			



