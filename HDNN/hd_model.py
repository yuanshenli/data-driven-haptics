import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F


class HDModel(nn.Module):
    def __init__(self, input_size = 3, hidden_size = 64):
        super(HDModel, self).__init__()

        # Defining some parameters
        self.hidden_size = hidden_size

        self.rnn_x = nn.LSTM(input_size, hidden_size)
        self.rnn_a = nn.LSTM(input_size, hidden_size)
        self.rnn_f = nn.LSTM(input_size, hidden_size)

        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(299, 1)

    
    def forward(self, x, a, f):
        

        x_unsqueezed = torch.unsqueeze(x, 2)     # (batch, seq, input)   (32, 299, 1)
        a_unsqueezed = torch.unsqueeze(a, 2)
        f_unsqueezed = torch.unsqueeze(f, 2)

        out = torch.cat((x_unsqueezed, a_unsqueezed, f_unsqueezed), 2)   # (batch, seq, input)   (32, 299, 3)

        out, _ = self.rnn(out)         # (batch, seq, hidden)   (32, 299, 3)
        out = self.relu(out[:,-1])          # (batch, seq, hidden)   (32, 299, 1)
        out = self.fc(out)

        # out1_relu = self.relu(out1)
        # out = self.fc2(torch.squeeze(out1_relu)) # (batch, 1)   (32, 1)

        # print(stacked.size())
        # print(hidden_out.size())
        # print(out_relu.size())
        # print(out1.size())
        # print(out1_relu.size())
        # print(out.size())
        
        return out
