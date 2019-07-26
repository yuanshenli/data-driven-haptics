import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F


class HDModel(nn.Module):
    def __init__(self, seq_len=299, input_size=3, hidden_size=256):
        super(HDModel, self).__init__()

        # Defining some parameters
        self.seq_len = seq_len
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.cnn1 = nn.Conv1d(input_size, hidden_size // 32, kernel_size=11, stride=1, padding=5)
        self.bn1 = nn.BatchNorm1d(hidden_size // 32)
        self.relu1 = nn.ReLU()
        # self.mp1 = nn.MaxPool1d(kernel_size=2)

        self.cnn2 = nn.Conv1d(hidden_size // 32, hidden_size // 16, kernel_size=7, stride=1, padding=3)
        self.bn2 = nn.BatchNorm1d(hidden_size // 16)
        self.relu2 = nn.ReLU()
        self.mp2 = nn.MaxPool1d(kernel_size=2)

        self.cnn3 = nn.Conv1d(hidden_size // 16, hidden_size // 8, kernel_size=7, stride=1, padding=3)
        self.bn3 = nn.BatchNorm1d(hidden_size // 8)
        self.relu3 = nn.ReLU()

        self.cnn4 = nn.Conv1d(hidden_size // 8, hidden_size // 4, kernel_size=7, stride=1, padding=3)
        self.bn4 = nn.BatchNorm1d(hidden_size // 4)
        self.relu4 = nn.ReLU()
        self.mp4 = nn.MaxPool1d(kernel_size=2)

        self.cnn5 = nn.Conv1d(hidden_size // 4, hidden_size // 2, kernel_size=5, stride=1, padding=2)
        self.bn5 = nn.BatchNorm1d(hidden_size // 2)
        self.relu5 = nn.ReLU()

        self.cnn6 = nn.Conv1d(hidden_size // 2, hidden_size, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm1d(hidden_size)
        self.relu6 = nn.ReLU()

        self.cnn7 = nn.Conv1d(hidden_size, 1, kernel_size=3, padding=1)
        # self.bn7 = nn.BatchNorm1d(hidden_size//16)
        self.relu7 = nn.ReLU()

        self.fc1 = nn.Linear((self.seq_len - 1) // 4 * hidden_size, 256)
        self.relu8 = nn.ReLU()

        self.fc2 = nn.Linear(256, 1)
        self.relu9 = nn.ReLU()

        # self.softmax = nn.LogSoftmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x_unsqueezed = torch.unsqueeze(x, 2)  # (batch, seq, input)   (32, 299, 1)
        # a_unsqueezed = torch.unsqueeze(a, 2)
        # f_unsqueezed = torch.unsqueeze(f, 2)

        # x = torch.cat((x_unsqueezed, a_unsqueezed, f_unsqueezed), 2)  # (batch, seq, input)   (32, 299, 3)

        # x = x.permute(0, 2, 1)
        x = x
        # print('layer cnn1')
        x = self.cnn1(x)
        # print(x.size())
        # x = self.bn1(x)
        # print(x.size())
        x = self.relu1(x)
        # print(x.size())

        # print('layer cnn2')
        x = self.cnn2(x)
        # print(x.size())
        # x = self.bn2(x)
        # print(x.size())
        x = self.relu2(x)
        # print(x.size())
        x = self.mp2(x)
        # print(x.size())

        # print('layer cnn3')
        x = self.cnn3(x)
        # print(x.size())
        # x = self.bn3(x)
        # print(x.size())
        x = self.relu3(x)
        # print(x.size())

        # print('layer cnn4')
        x = self.cnn4(x)
        # print(x.size())
        # x = self.bn4(x)
        # print(x.size())
        x = self.relu4(x)
        # print(x.size())
        x = self.mp4(x)
        # print(x.size())

        # print('layer cnn5')
        x = self.cnn5(x)
        # print(x.size())
        # x = self.bn5(x)
        # print(x.size())
        x = self.relu5(x)
        # print(x.size())

        # print('layer cnn6')
        x = self.cnn6(x)
        # print(x.size())
        # x = self.bn6(x)
        # print(x.size())
        x = self.relu6(x)
        # print(x.size())

        # # print('layer cnn7')
        # x = self.cnn7(x)
        # # print(x.size())
        # x = self.relu7(x)
        # # print(x.size())

        # print('layer fc')
        x = x.view(x.shape[0], -1)
        # print(x.size())

        x = self.fc1(x)
        # print(x.size())
        x = self.relu8(x)
        x = self.fc2(x)
        # x = self.relu9(x)

        # x = self.sigmoid(x)
        # print(x.size())

        return x

# class HDModel_LSTM(nn.Module):
#     def __init__(self, input_size = 3, hidden_size = 64):
#         super(HDModel, self).__init__()

#         # Defining some parameters
#         self.hidden_size = hidden_size

#         self.rnn_x = nn.LSTM(input_size, hidden_size)
#         self.rnn_a = nn.LSTM(input_size, hidden_size)
#         self.rnn_f = nn.LSTM(input_size, hidden_size)

#         self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)

#         self.fc = nn.Linear(hidden_size, 1)
#         self.relu = nn.ReLU()

#         self.fc2 = nn.Linear(299, 1)


#     def forward(self, x, a, f):


#         x_unsqueezed = torch.unsqueeze(x, 2)     # (batch, seq, input)   (32, 299, 1)
#         a_unsqueezed = torch.unsqueeze(a, 2)
#         f_unsqueezed = torch.unsqueeze(f, 2)

#         out = torch.cat((x_unsqueezed, a_unsqueezed, f_unsqueezed), 2)   # (batch, seq, input)   (32, 299, 3)

#         out, _ = self.rnn(out)         # (batch, seq, hidden)   (32, 299, 3)
#         out = self.relu(out[:,-1])          # (batch, seq, hidden)   (32, 299, 1)
#         out = self.fc(out)

#         # out1_relu = self.relu(out1)
#         # out = self.fc2(torch.squeeze(out1_relu)) # (batch, 1)   (32, 1)

#         # # print(stacked.size())
#         # # print(hidden_out.size())
#         # # print(out_relu.size())
#         # # print(out1.size())
#         # # print(out1_relu.size())
#         # # print(out.size())

#         return out
