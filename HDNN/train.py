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
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter


logdir = 'runs/HDModel-' + datetime.now().strftime('%y%m%d-%H%M%S')
# logdir = 'runs/HDModel-190721-171338'

if not os.path.exists(logdir):
	os.makedirs(logdir)

print(f'tensorboard --logdir={logdir}')
writer = SummaryWriter(log_dir=logdir)

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

resume_epochs = None
n_epochs = 10000
validation_interval = 1

lr=0.0001
learning_rate_decay_steps = 1000
learning_rate_decay_rate = 0.98

batch_size = 32
input_size = 3
seq_length = 10001
cont_length = 10000

patience, num_trial = 0, 0
max_patience, max_trial = 50, 50

hist_epoch = []
hist_training_loss = []
hist_validation_loss = []

# init model
if resume_epochs is None:
	model = HDModel(seq_len=seq_length, input_size=input_size)
	# if torch.cuda.device_count() > 1:
	# 	model = torch.nn.DataParallel(model, device_ids=[0, 2])
	model.to(device)
	
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	resume_epochs = 0
else: 
	model_path = os.path.join(logdir, f'model-{resume_epochs}.pt')
	model = torch.load(model_path)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	optimizer.load_state_dict(torch.load(os.path.join(logdir, 'last-optimizer-state.pt')))


print(model)
# criterion = nn.MSELoss()
class_weight = torch.FloatTensor([0.99, 1.0])
class_weight = class_weight.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weight)
scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)


# Load data
print('---- loading training data ------')
train_set = HDDataset(path='data_chirp', 
					  data_name='7_20_16_35.txt', 
					  group='train',
					  sequence_length=seq_length,
					  continuous_length=cont_length)
training_generator = DataLoader(train_set, batch_size, shuffle=True)

print('---- loading validation data ----')
validation_set = HDDataset(path='data_chirp', 
					  data_name='7_20_16_35.txt', 
					  group='validation',
					  sequence_length=seq_length,
					  continuous_length=cont_length)
validation_generator = DataLoader(validation_set, batch_size, shuffle=True)

print(f"Training size: {len(train_set)}; Validation size: {len(validation_set)}")


print('---- training -------------------')
for epoch in tqdm(range(resume_epochs + 1, n_epochs + 1)):
	# Train
	model.train()
	loss_sum = 0
	loss_iter = 0
	for local_x, local_a, local_f, local_y in training_generator:
		scheduler.step()

		optimizer.zero_grad() # Clears existing gradients from previous epoch
		local_x = local_x.to(device)
		local_a = local_a.to(device)
		local_f = local_f.to(device)
		local_y = local_y.to(device)

		# print(local_x.size())
		output = model(local_x, local_a, local_f)
		# print(local_y.size())
		# print(output.size())

		loss = criterion(output, local_y)
		loss.backward() # Does backpropagation and calculates gradients
		loss_sum += loss.item()
		loss_iter += 1
		# print(loss_iter)
		optimizer.step() # Updates the weights accordingly
		if loss_iter > 1000:
			break


	# Evaludate
	val_loss_sum = 0
	val_loss_iter = 0
	if epoch % validation_interval == 0 and epoch != 0:
		model.eval()
		with torch.no_grad():
			for validation_x, validation_a, validation_f, validation_y in validation_generator:
				validation_x = validation_x.to(device)
				validation_a = validation_a.to(device)
				validation_f = validation_f.to(device)
				validation_y = validation_y.to(device)
				validation_output = model(validation_x, validation_a, validation_f)

				# val_loss = criterion(validation_output.view(-1), validation_y)
				val_loss = criterion(validation_output, validation_y)

				val_loss_sum += val_loss.item()
				val_loss_iter += 1
			
				if val_loss_iter > 1000:
					break

			ave_train_loss = loss_sum/loss_iter
			ave_val_loss = val_loss_sum/val_loss_iter
			tqdm.write('Epoch: {}/{}......'.format(epoch, n_epochs), end=' ')
			tqdm.write("Training loss: %.4f; Validation loss: %.4f" % (ave_train_loss, ave_val_loss))
			# print("Training loss: {:.4f}; Validation loss: {:.4f}".format(loss.item(), val_loss.item()))

		is_better = len(hist_validation_loss) == 0 or ave_val_loss < min(hist_validation_loss)

		writer.add_scalar('loss/train_loss', ave_train_loss, epoch)
		writer.add_scalar('loss/val_loss', ave_val_loss, epoch)


		hist_training_loss.append(ave_train_loss)
		hist_validation_loss.append(ave_val_loss)
		hist_epoch.append(epoch)

		# Save and Early Stop
		if is_better:
			patience = 0
			torch.save(model, os.path.join(logdir, f'model-{epoch}.pt'))
			torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))
		else:
			patience += 1
			tqdm.write('hit patience %d' % patience, file=sys.stderr)
			if patience == max_patience:
				num_trial += 1
				tqdm.write('hit #%d trial' % num_trial, file=sys.stderr)
				if num_trial == max_trial:
					tqdm.write('early stop!', file=sys.stderr)
					hist_file_name = logdir + '/loss_hist.log'
					hist_file = open(hist_file_name, "w")
					for i in range(len(hist_epoch)):
						tqdm.write(str(hist_epoch[i])+', '+str(hist_training_loss[i])+', '+str(hist_validation_loss[i]))
						hist_file.write("%d, %.2f, %.2f\n" % (hist_epoch[i], hist_training_loss[i], hist_validation_loss[i]))
					hist_file.close()
					writer.close()
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
writer.close()
hist_file.close()

# print(hist_training_loss)
# print(hist_validation_loss)
# print(hist_epoch)



