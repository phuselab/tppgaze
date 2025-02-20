import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        input_size: int,
        mean_log_inter_time: float,
        std_log_inter_time: float,
        type="GRU",
        bidirectional=False,
        num_layers=1,
        dropout=0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.input_size = input_size
        if self.bidirectional:
            self.context_multiply = 2
        else:
            self.context_multiply = 1
        self.context_init = nn.Parameter(
            torch.zeros(self.hidden_size * self.context_multiply)
        )  # initial state of the RNN
        self.rnn: nn.RNNBase = getattr(nn, type)(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=self.bidirectional,
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
        context: torch.Tensor = self.rnn(features)[0] # it contains for each timestep t the hidden state of the RNN at that timestep
        batch_size, seq_len, context_size = context.data.shape
        context_init = self.context_init[None, None, :].expand(
            batch_size, 1, -1
        )  # (batch_size, 1, context_size)

        # Shift the context by vectors by 1: context embedding after event i is used to predict event i + 1
        if remove_last:
            context = context[:, :-1, :]
        context = torch.cat([context_init, context], dim=1)

        return context
