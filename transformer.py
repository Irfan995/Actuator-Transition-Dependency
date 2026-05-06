import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim,
        state_embed_dim,
        time_embed_dim,
        atd_embed_dim,
        num_heads,
        num_layers,
        output_dim,
        max_len=50,
        hidden_dim=64,
        dropout_prob=0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.state_processor = nn.Sequential(nn.Linear(1, state_embed_dim), nn.ReLU())
        self.time_processor = nn.Sequential(nn.Linear(1, time_embed_dim), nn.ReLU())
        self.atd_processor = nn.Sequential(nn.Linear(1, atd_embed_dim), nn.ReLU())

        combined_dim = embed_dim + state_embed_dim + time_embed_dim + atd_embed_dim
        self.pos_encoder = PositionalEncoding(combined_dim, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=combined_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout_prob,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attention_pool = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.fc_actuator = nn.Linear(combined_dim, output_dim)
        self.fc_state = nn.Linear(combined_dim, 1)

    def forward(self, x_ids, x_states, x_deltas, x_atd):
        embedded_ids = self.embedding(x_ids)
        embedded_states = self.state_processor(x_states.unsqueeze(2))
        embedded_deltas = self.time_processor(torch.log1p(x_deltas).unsqueeze(2))
        embedded_atd = self.atd_processor(x_atd.unsqueeze(2))

        x = torch.cat(
            [embedded_ids, embedded_states, embedded_deltas, embedded_atd],
            dim=2,
        )
        x = self.pos_encoder(x)
        x = self.transformer(x)

        scores = self.attention_pool(x)
        weights = F.softmax(scores, dim=1)
        pooled = self.dropout((x * weights).sum(dim=1))

        actuator_out = self.fc_actuator(pooled)
        state_out = torch.sigmoid(self.fc_state(pooled)).squeeze(1)
        return actuator_out, state_out
