# https://github.com/iShohei220/corl-gqn/blob/taniguchi/ssmgqn/conv_lstm.py

import torch
from torch import nn

def init_weights(distribution):
    print("---- init weights ----")
    for m in distribution.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.normal_(m.bias)
        elif isinstance(
            m, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d)):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LSTM, nn.LSTMCell, nn.GRU, nn.GRUCell)):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    nn.init.orthogonal_(param.data)
                else:
                    nn.init.normal_(param.data)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        else:
            print("  ", type(m))
            continue
        print("ok", type(m))

class Conv2dLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv2dLSTMCell, self).__init__()

        kwargs = dict(kernel_size=kernel_size, stride=stride, padding=padding)
        
        in_channels += out_channels
        
        self.forget = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.input  = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.output = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.state  = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, input, states):
        (cell, hidden) = states
        input = torch.cat((hidden, input), dim=1)
        
        forget_gate = torch.sigmoid(self.forget(input))
        input_gate  = torch.sigmoid(self.input(input))
        output_gate = torch.sigmoid(self.output(input))
        state_gate  = torch.tanh(self.state(input))

        # Update internal cell state
        cell = forget_gate * cell + input_gate * state_gate
        hidden = output_gate * torch.tanh(cell)

        return cell, hidden


class Conv2dLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv2dLSTM, self).__init__()
        self.core = Conv2dLSTMCell(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, input, states):
        output = []
        (cell, hidden) = states
        cell, hidden = cell[0], hidden[0]
        _T = input.size(0)
        
        for t in range(_T):
            cell, hidden = self.core(input[t], (cell, hidden))
            output.append(hidden)
        output = torch.stack(output)
        
        return output, (cell, hidden)