"""
Transformer-XL block (simplified).
Captures long-term dependencies using recurrence + relative encoding.
We project input features into an embedding dimension that is divisible by num_heads.
"""

import torch
import torch.nn as nn

class TransformerXLBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim=128, num_heads=4, num_layers=2):
        super(TransformerXLBlock, self).__init__()
        # Project input_dim (e.g., 215 features) into a lower-dimensional embedding
        self.proj = nn.Linear(input_dim, embed_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, memory=None):
        """
        x: (batch, seq_len, input_dim)
        """
        x_proj = self.proj(x)          # (batch, seq_len, embed_dim)
        out = self.transformer(x_proj) # (batch, seq_len, embed_dim)
        return out
