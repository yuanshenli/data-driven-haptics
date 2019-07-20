import os
from torch.utils.data import Dataset, TensorDataset
import numpy as np
from tqdm import tqdm

class HDDataset(Dataset):
    def __init__(self, path='', data_name='', group=None, sequence_length=300, continuous_length=300):
        self.path = path
        self.data_name = data_name
        self.myfile_name = os.path.join(self.path, self.data_name)
        self.group = group
        self.sequence_length = sequence_length
        self.continuous_length = continuous_length

        x = []
        idx_set = []
        train_percent = 0.8
        validate_percent = 0.1
        test_percent = 0.1

        # count the total lines
        with open(self.myfile_name) as f:
            for total_lines, l in enumerate(f):
                pass
        total_lines += 1
        if f:
            f.close()

        last_idx = total_lines - self.sequence_length # last possible index can be accessed without getting out of range
        
        # a chunk is a piece of continuous data that is going to be dividied into different sets
        one_chunk = sequence_length * 3 + continuous_length - 3
        num_chunks = total_lines // one_chunk   # number of complete chunks of train, test and validation divisions
        
        # number of data for each set in each chunk
        train_per_chunk = int(continuous_length * train_percent)
        validation_per_chunk = int(continuous_length * validate_percent)
        test_per_chunk = int(continuous_length * test_percent)

        train_validation_offset = sequence_length + train_per_chunk - 1
        validation_test_offset = sequence_length + validation_per_chunk - 1

        # last chunk is probably incomplete
        last_chunk_cutoff_train = one_chunk*num_chunks + train_per_chunk - 1   
        last_chunk_cutoff_validation = one_chunk*num_chunks + train_validation_offset + validation_per_chunk - 1   
        last_chunk_cutoff_test = one_chunk*num_chunks + train_validation_offset + validation_test_offset + test_per_chunk - 1
        
        # save all data in an array
        with open(self.myfile_name) as f:
            for idx, line in enumerate(f):
                line = line.split(",")
                d0 = float(line[0]) # pos
                d1 = float(line[1]) # force
                d2 = float(line[2]) # acc
                x.append([d0, d1, d2]) 
        if f:
            f.close()

        if group == 'train':
            for jj in range(num_chunks):
                for ii in range(int(continuous_length * train_percent)):
                    idx_set.append(ii + one_chunk*jj)
            # last chunk, could be empty
            left = list(range(one_chunk*num_chunks, 
                              min(last_idx, last_chunk_cutoff_train), 1))

        elif group == 'validation':
            for jj in range(num_chunks):
                for ii in range(int(continuous_length * validate_percent)):
                    idx_set.append(ii + one_chunk*jj + train_validation_offset)
            # last chunk, could be empty
            left = list(range(one_chunk*num_chunks+train_validation_offset, 
                              min(last_idx, last_chunk_cutoff_validation), 1))

        elif group == 'test':
            for jj in range(num_chunks):
                for ii in range(int(continuous_length * test_percent)):
                    idx_set.append(ii + one_chunk*jj + train_validation_offset + validation_test_offset)
            # last chunk, could be empty
            left = list(range(one_chunk*num_chunks + train_validation_offset + validation_test_offset, 
                              min(last_idx, last_chunk_cutoff_test), 1))
        
        idx_set = idx_set + left

        self.x = np.asarray(x, dtype=np.float32)
        self.idx_set = np.asarray(idx_set, dtype=np.int)  

        # print(idx_set)  


    def __getitem__(self, index):
        this_idx = self.idx_set[index]
        # print(this_idx)
        return (self.x[this_idx:this_idx+self.sequence_length-2, 0],
                self.x[this_idx:this_idx+self.sequence_length-2, 1],
                self.x[this_idx:this_idx+self.sequence_length-2, 2], 
                self.x[this_idx+self.sequence_length-1, 1])

    def __len__(self):
        return len(self.idx_set)

    @classmethod
    def line_num(self):
        with open(self.myfile_name) as f:
            for i, l in enumerate(f):
                pass
        return i + 1

