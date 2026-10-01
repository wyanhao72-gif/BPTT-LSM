"""
ETT时间序列预测网络模块
本模块实现了用于ETT数据集时间序列预测的混合网络结构。
结合了线性层和储层计算，支持历史步长信息的利用。
"""

import numpy as np
import torch
import torch.nn as nn


class net_mix:
    def __init__(self, input_layer, rc, output_layer, history_step):
        self.input_layer = input_layer
        self.rc = rc
        self.output_layer = output_layer
        self.history_step = history_step

    def forward(self, x):
        # x = x.permute(1, 0, 2)
        l, b, c = x.shape
        x = x.reshape(l, c)
        x = self.input_layer(x)
        self.rc(x)
        x = self.rc[0].rc_state_s_h[-x.shape[0]:]  # 取网络的状态（s, v） (channel, batch, length)
        if self.history_step != 1:
            new_rc_out_set = torch.tensor([]).to(x)
            for x_len_para in range(x.shape[0]):
                new_rc_out = torch.tensor([]).to(x)
                for history_step_para in range(self.history_step):
                    head_index = x_len_para - history_step_para
                    if head_index < 0:
                        new_rc_out = torch.cat([new_rc_out, torch.zeros(size=(1, 1, x.shape[-1])).to(x)], dim=2)
                    else:
                        new_rc_out = torch.cat([new_rc_out, x[head_index: head_index + 1]], dim=2)
                new_rc_out_set = torch.cat([new_rc_out_set, new_rc_out], dim=0)
        else:
            new_rc_out_set = x.clone()

        x = self.output_layer(new_rc_out_set)
        return x


# 定义一个简单的循环神经网络模型
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden = None

    def forward(self, x):
        seq_length, batch_size, _ = x.shape
        # 初始化隐藏状态
        if self.hidden is None:
            self.hidden = torch.zeros(1, batch_size, self.hidden_size).to(x)
        # 使用BPTT进行截断反向传播
        output, self.hidden = self.rnn(x, self.hidden)
        output = self.fc(output)
        return output

    def init(self):
        self.hidden = None


class LstmRNN(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  # utilize the LSTM model in torch.nn
        self.fc = nn.Linear(hidden_size, output_size)
        self.h_t = None
        self.c_t = None

    def forward(self, _x):
        seq_length, batch_size, _ = _x.shape
        if self.h_t is None:
            self.h_t = torch.zeros(1, batch_size, self.hidden_size).to(_x)
            self.c_t = torch.zeros(1, batch_size, self.hidden_size).to(_x)
        output, (self.h_t, self.c_t) = self.lstm(_x,
                                                 (self.h_t, self.c_t))  # _x is input, size (seq_len, batch, input_size)
        output = self.fc(output)  # output输入的尺寸应该为(seq_len, batch, hidden_size)
        return output

    def init(self):
        self.h_t = None
        self.c_t = None


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers)  # utilize the LSTM model in torch.nn
        self.fc = nn.Linear(hidden_size, output_size)
        self.h_t = None

    def forward(self, _x):
        seq_length, batch_size, _ = _x.shape
        if self.h_t is None:
            self.h_t = torch.zeros(1, batch_size, self.hidden_size).to(_x)

        output, self.h_t = self.gru(_x, self.h_t)  # _x is input, size (seq_len, batch, input_size)
        output = self.fc(output)  # output输入的尺寸应该为(seq_len, batch, hidden_size)
        return output

    def init(self):
        self.h_t = None


