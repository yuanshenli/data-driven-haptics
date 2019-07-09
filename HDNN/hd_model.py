import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F


class HDModel(nn.Module):
    def __init__(self, input_size, hidden_size = 100):
        super(HDModel, self).__init__()

        # Defining some parameters
        self.hidden_size = hidden_size

        self.fc_x = nn.Linear(input_size, hidden_size)
        self.relu_x = nn.ReLU()

        self.fc_a = nn.Linear(input_size, hidden_size)
        self.relu_a = nn.ReLU()

        self.fc_f = nn.Linear(input_size, hidden_size)
        self.relu_f = nn.ReLU()

        self.fc = nn.Linear(hidden_size*3, 1)
    
    def forward(self, x, a, f):
        
        out_x = self.fc_x(x)
        out_x = self.relu_x(out_x)

        out_a = self.fc_a(a)
        out_a = self.relu_a(out_a)

        out_f = self.fc_f(f)
        out_f = self.relu_f(out_f)

        out_combined = torch.cat((out_x, out_a, out_f), dim=1)

        out = self.fc(out_combined)
        
        return out
