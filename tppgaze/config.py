from omegaconf import DictConfig
import torch.nn.functional as F

from tppgaze.models.cat_merge import CatMerge
from tppgaze.models.dg3ish import DenseNetDG3ish
from tppgaze.models.rnn import RNN
from tppgaze.models.transformer_encoder import TransformerEncoderContext

from tppgaze.distributions.log_normal_mixture import LogNormalMixture
from tppgaze.distributions.mixture_same_family import MixtureSameFamilyModule

def get_activation(activation: str):
    if activation == "relu":
        return F.relu
    elif activation == "tanh":
        return F.tanh
    elif activation == "gelu":
        return F.gelu
    elif activation == "identity":
        return lambda x: x


def get_context(cfg: DictConfig):
    if cfg.context.type == "transformer":
        return TransformerEncoderContext(
            hidden_size=cfg.context.size,
            mean_log_inter_time=cfg.mean_log_inter_times,
            std_log_inter_time=cfg.std_log_inter_times,
            dropout=cfg.dropout,
        )
    elif cfg.context.type == "rnn":
        return RNN(
            input_size=3,
            hidden_size=cfg.context.size,
            mean_log_inter_time=cfg.mean_log_inter_times,
            std_log_inter_time=cfg.std_log_inter_times,
            type='GRU',
            bidirectional=False,
            dropout=cfg.dropout,
        )
    else:
        raise ValueError(f"Context type {cfg.context.type} not supported")


def get_metadata(cfg: DictConfig):
    if cfg.metadata.type == "densenet":
        return DenseNetDG3ish(
            channels1=8,
            channels2=16,
            channels3=1,
            output_dim=cfg.metadata.size,
        )
    else:
        raise ValueError(f"Metadata type {cfg.metadata.type} not supported")


def get_merge(cfg: DictConfig):
    if cfg.merge == "cat":
        return CatMerge()
    else:
        raise ValueError(f"Merge type {cfg.merge} not supported")


def get_distributions_input_size(cfg: DictConfig) -> int:
    if cfg.merge == "cat":
        return cfg.metadata.size + cfg.context.size


def get_inter_times_dist(cfg: DictConfig):
    return LogNormalMixture(
        get_distributions_input_size(cfg),
        cfg.durations_dist_components,
        cfg.mean_log_inter_times,
        cfg.std_log_inter_times,
    )

def get_marks_dist(cfg: DictConfig):
    return MixtureSameFamilyModule(
        get_distributions_input_size(cfg), cfg.marks_dist_components
    )
