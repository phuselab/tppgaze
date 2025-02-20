from typing import Optional, List
import torch
import torch.nn as nn
import torch.distributions as D
from tppgaze.utils import clamp_preserve_gradients
from tppgaze.distributions.normal import Normal
from tppgaze.distributions.mixture_same_family import MixtureSameFamily
from tppgaze.distributions.transformed_distribution import TransformedDistribution

class LogNormalMixture(nn.Module):
    def __init__(
        self,
        in_features: int,
        durations_dist_components: int,
        mean_log_inter_time: float,
        std_log_inter_time: float,
    ):
        super().__init__()

        self.in_features = in_features
        self.durations_dist_components = durations_dist_components
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time

        self.linear = nn.Linear(
            in_features,
            3 * self.durations_dist_components,
        )

    def forward(
        self, context: torch.Tensor, temperature: Optional[float] = None
    ) -> D.Distribution:
        raw_params = self.linear(
            context
        )  # (batch_size, seq_len, 3 * durations_dist_components)

        # Slice the tensor to get the parameters of the mixture
        locs = raw_params[..., : self.durations_dist_components]
        log_scales = raw_params[
            ..., self.durations_dist_components : (2 * self.durations_dist_components)
        ]
        log_weights = raw_params[..., (2 * self.durations_dist_components) :]
        log_scales = clamp_preserve_gradients(log_scales, -5.0, 3.0)
        log_weights = torch.log_softmax(log_weights, dim=-1)

        return self.LogNormalMixtureDistribution(
            locs=locs,
            log_scales=log_scales,
            log_weights=log_weights,
            mean_log_inter_time=self.mean_log_inter_time,
            std_log_inter_time=self.std_log_inter_time,
        )

    class LogNormalMixtureDistribution(TransformedDistribution):
        """
        Mixture of log-normal distributions.

        We model it in the following way (see Appendix D.2 in the paper):

        x ~ GaussianMixtureModel(locs, log_scales, log_weights)
        y = std_log_inter_time * x + mean_log_inter_time
        z = exp(y)

        Args:
            locs: Location parameters of the component distributions,
                shape (batch_size, seq_len, durations_dist_components)
            log_scales: Logarithms of scale parameters of the component distributions,
                shape (batch_size, seq_len, durations_dist_components)
            log_weights: Logarithms of mixing probabilities for the component distributions,
                shape (batch_size, seq_len, durations_dist_components)
            mean_log_inter_time: Average log-inter-event-time, see tppgaze.data.dataset.get_inter_time_statistics
            std_log_inter_time: Std of log-inter-event-times, see tppgaze.data.dataset.get_inter_time_statistics
        """

        def __init__(
            self,
            locs: torch.Tensor,
            log_scales: torch.Tensor,
            log_weights: torch.Tensor,
            mean_log_inter_time: float = 0.0,
            std_log_inter_time: float = 1.0,
        ):
            mixture_dist = D.Categorical(logits=log_weights)
            component_dist = Normal(loc=locs, scale=log_scales.exp())
            GMM = MixtureSameFamily(mixture_dist, component_dist)
            if mean_log_inter_time == 0.0 and std_log_inter_time == 1.0:
                transforms: List[D.Transform] = []
            else:
                transforms = [
                    D.AffineTransform(loc=mean_log_inter_time, scale=std_log_inter_time)
                ]
            self.mean_log_inter_time = mean_log_inter_time
            self.std_log_inter_time = std_log_inter_time
            transforms.append(D.ExpTransform())
            super().__init__(GMM, transforms)

        @property
        def mean(self) -> torch.Tensor:
            """
            Compute the expected value of the distribution.

            See https://github.com/shchur/ifl-tpp/issues/3#issuecomment-623720667

            Returns:
                mean: Expected value, shape (batch_size, seq_len)
            """
            a = self.std_log_inter_time
            b = self.mean_log_inter_time
            loc = self.base_dist._component_distribution.loc  # type: ignore
            variance = self.base_dist._component_distribution.variance  # type: ignore
            log_weights = self.base_dist._mixture_distribution.logits  # type: ignore
            return (log_weights + a * loc + b + 0.5 * a**2 * variance).logsumexp(-1).exp()