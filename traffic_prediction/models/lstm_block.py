"""
Streaming LSTM block.
Captures short-term temporal patterns.
"""

import torch.nn as nn

class LSTMBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(LSTMBlock, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x, hidden=None):
        # x: (batch, seq_len, input_dim)
        out, hidden = self.lstm(x, hidden)
        return out, hidden
