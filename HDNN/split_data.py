import os
import numpy as np

def split_file(lines_per_file, starting_line, save_dir):
	smallfile = None
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	with open('data_chirp/7_11_16_43.txt') as bigfile:
		for lineno, line in enumerate(bigfile):
			if lineno % lines_per_file == starting_line:
				if smallfile:
					smallfile.close()
				small_filename = save_dir + '/small_file_{}.txt'.format(lineno)
				smallfile = open(small_filename, "w")
			if smallfile:
				smallfile.write(line)
		if smallfile:
			smallfile.close()
	if bigfile:
		bigfile.close()
	
	# pad the last file to the same length
	with open(small_filename, 'r+') as smallfile:
		for lineno, line in enumerate(smallfile):
			pass
		if lineno < lines_per_file - 1:
			for ii in range(lineno + 1, lines_per_file):
				smallfile.write('0.0, 0.0, 0.0\n')
	if smallfile:
		smallfile.close()

if __name__== "__main__":
	lines_per_file = 300
	train_percent = 0.8
	validate_percent = 0.1
	test_percent = 0.1
	starting_line = 0
	for starting_line in range(int(lines_per_file * train_percent)):
		split_file(lines_per_file, starting_line, 'data_chirp/train')

	for starting_line in range(int(lines_per_file * train_percent), int(lines_per_file * (train_percent + validate_percent))):
		split_file(lines_per_file, starting_line, 'data_chirp/validation')

	for starting_line in range(int(lines_per_file * (train_percent + validate_percent)), lines_per_file):
		split_file(lines_per_file, starting_line, 'data_chirp/test')







