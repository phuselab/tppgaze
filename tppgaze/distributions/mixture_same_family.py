import torch
import torch.distributions as D
import torch.nn as nn
from tppgaze.utils import clamp_preserve_gradients
from torch.distributions import MixtureSameFamily as TorchMixtureSameFamily

class MixtureSameFamily(TorchMixtureSameFamily):
    def log_cdf(self, x):
        x = self._pad(x)
        log_cdf_x = self.component_distribution.log_cdf(x)
        mix_logits = self.mixture_distribution.logits
        return torch.logsumexp(log_cdf_x + mix_logits, dim=-1)

    def log_survival_function(self, x):
        x = self._pad(x)
        log_sf_x = self.component_distribution.log_survival_function(x)
        mix_logits = self.mixture_distribution.logits
        return torch.logsumexp(log_sf_x + mix_logits, dim=-1)


class MixtureSameFamilyModule(nn.Module):
    def __init__(self, in_features: int, marks_dist_components: int):
        super().__init__()

        self.marks_dist_components = marks_dist_components

        self.linear = torch.nn.Linear(
            in_features=in_features, out_features=5 * marks_dist_components
        )

    def forward(self, context, temperature=None):
        # Marks
        mark_params: torch.Tensor = self.linear(
            context
        )  # (batch_size, seq_len, num_marks)

        locs = torch.tanh(mark_params[..., : 2 * self.marks_dist_components])
        log_scales = mark_params[
            ..., 2 * self.marks_dist_components : (4 * self.marks_dist_components)
        ]
        log_weights = mark_params[..., (4 * self.marks_dist_components) :]
        seq_len = context.shape[1]
        locs = locs.reshape(-1, seq_len, self.marks_dist_components, 2)
        log_scales = log_scales.reshape(-1, seq_len, self.marks_dist_components, 2)
        log_scales = clamp_preserve_gradients(log_scales, -5.0, 3.0)
        log_weights = torch.log_softmax(log_weights, dim=-1)
        mixture_dist = D.Categorical(logits=log_weights)
        scales = temperature * log_scales if temperature is not None else log_scales
        component_dist = D.Independent(
            base_distribution=D.Normal(loc=locs, scale=torch.exp(scales)),
            reinterpreted_batch_ndims=1,
        )
        return D.MixtureSameFamily(mixture_dist, component_dist)
