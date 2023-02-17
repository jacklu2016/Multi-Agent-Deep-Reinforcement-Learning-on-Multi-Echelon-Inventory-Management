import torch
from torch import nn


class LSTMLayer(nn.Module):
    def __init__(self, inputs_dim, outputs_dim, recurrent_N, use_orthogonal):
        super(LSTMLayer, self).__init__()
        self._recurrent_N = recurrent_N
        self._use_orthogonal = use_orthogonal
        # 定义LSTM
        self.rnn = nn.LSTM(inputs_dim, outputs_dim, recurrent_N)
        # 定义回归层网络，输入的特征维度等于LSTM的输出，输出维度为1
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        self.norm = nn.LayerNorm(outputs_dim)

    def forward(self, x, hxs, masks):
        x = self.norm(x)
        return x, hxs
