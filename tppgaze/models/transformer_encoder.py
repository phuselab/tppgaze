import math
import torch
import torch.nn as nn

# https://github.com/SimiaoZuo/Transformer-Hawkes-Process/blob/master/transformer/Models.py

def get_non_pad_mask(seq: torch.Tensor):
    """Get the non-padding positions."""
    return seq.ne(0).type(torch.float)

class Encoder(nn.Module):
    """A encoder model with self attention mechanism."""

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        n_layers: int,
        n_head: int,
        dropout: float,
    ):
        super().__init__()

        self.d_model = d_model

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)])

        self.marks_linear = nn.Linear(2, d_model)

        self.layer_stack = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_head,
                    dim_feedforward=d_inner,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def temporal_enc(self, time: torch.Tensor, non_pad_mask: torch.Tensor):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result

    def forward(
        self, times: torch.Tensor, marks: torch.Tensor, non_pad_mask: torch.Tensor
    ):
        """Encode event sequences via masked self-attention."""
        tem_enc = self.temporal_enc(times, non_pad_mask)
        enc_output = self.marks_linear(marks)

        for enc_layer in self.layer_stack:
            enc_output += tem_enc
            enc_output = enc_layer(enc_output)
        return enc_output

class RNN_layers(nn.Module):
    """
    Optional recurrent layers. This is inspired by the fact that adding
    recurrent layers on top of the Transformer helps language modeling.
    """

    def __init__(self, d_model: int, d_rnn: int):
        super().__init__()

        self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True)
        self.projection = nn.Linear(d_rnn, d_model)

    def forward(self, data: torch.Tensor):
        out = self.rnn(data)[0]

        out = self.projection(out)
        return out

class Transformer(nn.Module):
    """A sequence to sequence model with attention mechanism."""

    def __init__(
        self,
        d_model=256,
        d_rnn=128,
        d_inner=1024,
        n_layers=4,
        n_head=4,
        dropout=0.1,
    ):
        super().__init__()

        self.encoder = Encoder(
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            dropout=dropout,
        )

        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))

        # OPTIONAL recurrent layer, this sometimes helps
        self.rnn = RNN_layers(d_model, d_rnn)

    def forward(self, times, marks):
        non_pad_mask = get_non_pad_mask(times)
        enc_output = self.encoder(times, marks, non_pad_mask)
        enc_output = self.rnn(enc_output)
        return enc_output

class TransformerEncoderContext(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        mean_log_inter_time: float,
        std_log_inter_time: float,
        dropout=0.4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.context_init = nn.Parameter(torch.zeros(self.hidden_size))
        self.encoder = Transformer(
            d_model=hidden_size, d_rnn=hidden_size, dropout=dropout,
        )
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time

    def get_features(self, inter_times, marks):
        features = torch.log(inter_times + 1e-8).unsqueeze(
            -1
        )  # (batch_size, seq_len, 1)
        features = (features - self.mean_log_inter_time) / self.std_log_inter_time
        # marks
        mark_emb = marks  # (batch_size, seq_len, mark_embedding_size)
        return torch.cat([features, mark_emb], dim=-1)

    def get_context_init(self) -> torch.Tensor:
        return self.context_init

    def forward(self, features, remove_last):
        context: torch.Tensor = self.encoder(features[:, :, 0], features[:, :, 1:])
        batch_size, seq_len, context_size = context.data.shape
        context_init = self.context_init[None, None, :].expand(
            batch_size, 1, -1
        )  # (batch_size, 1, context_size)

        # Shift the context by vectors by 1: context embedding after event i is used to predict event i + 1
        if remove_last:
            context = context[:, :-1, :]
        context = torch.cat([context_init, context], dim=1)
        return context