import os
import sys
from datetime import datetime
from tqdm import tqdm
import numpy as np

from hd_model import HDModel
from dataset import HDDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# import torch.nn.utils
# import torch.nn.functional as F


logdir = 'runs/HDModel-' + datetime.now().strftime('%y%m%d-%H%M%S')
if not os.path.exists(logdir):
	os.makedirs(logdir)

resume_epochs = None
n_epochs = 10000
validation_interval = 1

lr=0.0001
batch_size = 32
input_size = 3
seq_length = 299

patience, num_trial = 0, 0
max_patience, max_trial = 5, 5

loss_sum = 0
loss_iter = 0

hist_epoch = []
hist_training_loss = []
hist_validation_loss = []

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
train_set = HDDataset(path='data_chirp', group='train')
training_generator = DataLoader(train_set, batch_size, shuffle=True)

print('---- loading validation data ----')
validation_set = HDDataset(path='data_chirp', group='validation')
validation_generator = DataLoader(validation_set, len(validation_set), shuffle=True)

# batch = list(validation_generator)[0]
# print(batch[0].size())
# w, y = batch[0]
# print(w.shape)

print('---- training -------------------')
for epoch in tqdm(range(n_epochs)):
	# Train
	model.train()
	loss_sum = 0
	loss_iter = 0
	for local_x, local_a, local_f, local_y in training_generator:
		optimizer.zero_grad() # Clears existing gradients from previous epoch
		local_x = local_x.to(device)
		local_a = local_a.to(device)
		local_f = local_f.to(device)
		local_y = local_y.to(device)
		# print(local_x.size())
		output = model(local_x, local_a, local_f)
		loss = criterion(output.view(-1), local_y)
		loss.backward() # Does backpropagation and calculates gradients
		loss_sum += loss.item()
		loss_iter += 1
		optimizer.step() # Updates the weights accordingly

	# Evaludate
	val_loss_sum = 0
	val_loss_iter = 0
	if epoch % validation_interval == 0 and epoch != 0:
		# model.eval()
		with torch.no_grad():
			for validation_x, validation_a, validation_f, validation_y in training_generator:
				validation_x = validation_x.to(device)
				validation_a = validation_a.to(device)
				validation_f = validation_f.to(device)
				validation_y = validation_y.to(device)
				validation_output = model(validation_x, validation_a, validation_f)

				val_loss = criterion(validation_output.view(-1), validation_y)

				val_loss_sum += val_loss.item()
				val_loss_iter += 1
				if val_loss_iter > len(validation_set):
					break
			print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
			# print("Training loss: {:.4f}; Validation loss: {:.4f}".format(loss_sum/loss_iter, val_loss_sum/val_loss_iter))
			print(f"Training loss: {loss_sum/loss_iter}; Validation loss: {val_loss_sum/val_loss_iter}")
			# print("Training loss: {:.4f}; Validation loss: {:.4f}".format(loss.item(), val_loss.item()))

		is_better = len(hist_validation_loss) == 0 or val_loss.item() < min(hist_validation_loss)

		hist_training_loss.append(loss_sum/loss_iter)
		hist_validation_loss.append(val_loss.item())
		hist_epoch.append(epoch)

		# Save and Early Stop
		if is_better:
			patience = 0
			torch.save(model, os.path.join(logdir, f'model-{epoch}.pt'))
			torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))
		else:
			patience += 1
			print('hit patience %d' % patience, file=sys.stderr)
			if patience == max_patience:
				num_trial += 1
				print('hit #%d trial' % num_trial, file=sys.stderr)
				if num_trial == max_trial:
					print('early stop!', file=sys.stderr)
					hist_file_name = logdir + '/loss_hist.log'
					hist_file = open(hist_file_name, "w")
					for i in range(len(hist_epoch)):
						print(str(hist_epoch[i])+', '+str(hist_training_loss[i])+', '+str(hist_validation_loss[i]))
						hist_file.write("%d, %.2f, %.2f\n" % (hist_epoch[i], hist_training_loss[i], hist_validation_loss[i]))
					hist_file.close()
					exit(0)
				# reset patience
				patience = 0
	
	# if loss.item() < best_loss or best_loss == -1:
	# 	best_loss = loss.item()
	# 	torch.save(model, os.path.join(logdir, f'best_model.pt'))
	# 	torch.save(optimizer.state_dict(), os.path.join(logdir, 'best-model-optimizer-state.pt'))

hist_file_name = logdir + '/loss_hist.log'
hist_file = open(hist_file_name, "w")
for i in range(len(hist_epoch)):
	hist_file.write("%d, %.2f, %.2f\n" % (hist_epoch[i], hist_training_loss[i], hist_validation_loss[i]))
hist_file.close()

# print(hist_training_loss)
# print(hist_validation_loss)
# print(hist_epoch)



