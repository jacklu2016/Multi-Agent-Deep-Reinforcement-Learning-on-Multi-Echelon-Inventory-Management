import torch
from torch import nn

class LSTMLayer(nn.Module):
    def __init__(self, inputs_dim, outputs_dim, recurrent_N, use_orthogonal):
        super(LSTMLayer, self).__init__()
        # 定义LSTM
        self.rnn = nn.LSTM(inputs_dim, outputs_dim, recurrent_N)
        # 定义回归层网络，输入的特征维度等于LSTM的输出，输出维度为1
        self.reg = nn.Sequential(
            nn.Linear(outputs_dim, 1)
        )

    def forward(self, x, hxs, masks):
        x, (ht,ct) = self.rnn(x)
        seq_len, batch_size, hidden_size= x.shape
        x = x.view(-1, hidden_size)
        x = self.reg(x)
        x = x.view(seq_len, batch_size, -1)
        return x
