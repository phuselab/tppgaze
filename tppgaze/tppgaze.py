from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.distributions.distribution import Distribution
from omegaconf import OmegaConf
from tppgaze.config import get_context, get_metadata, get_merge, get_inter_times_dist, get_marks_dist
from tppgaze.utils import preprocess_image, rescale, diff
import numpy as np

class TPPGaze(nn.Module):
    def __init__(self, cfg_path: str, checkpoint_path, device) -> None:
        super().__init__()
        
        self.cfg = OmegaConf.load(cfg_path)
        self.set_seed(self.cfg.seed)
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.context = get_context(self.cfg)  # type: ignore
        self.metadata = get_metadata(self.cfg)
        self.merge = get_merge(self.cfg)
        self.inter_time_dist = get_inter_times_dist(self.cfg)  # type: ignore
        self.marks_dist = get_marks_dist(self.cfg)
        self.dropout = nn.Dropout(self.cfg.dropout)
        
        self.temperature = self.cfg.temperature

    def load_model(self):
        self.to(self.device)
        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        self.load_state_dict(state_dict)
        self.eval()

    def set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)

    def generate_predictions(self, image_path, sample_duration=3.0, n_simulations=5):
        image, H, W = preprocess_image(image_path)
        
        inter_times_batch, marks_batch, mask = self.sample(
            t_end=sample_duration,
            img=image.to(self.device),
            batch_size=n_simulations,
            temperature=self.temperature,
        )
        T = (inter_times_batch * 1000).cpu().detach().numpy()
        X = np.clip(rescale(-1.0, 1.0, 0, W - 1, marks_batch[:, :, 0]).cpu().detach().numpy(), 0, W - 1)
        Y = np.clip(rescale(-1.0, 1.0, 0, H - 1, marks_batch[:, :, 1]).cpu().detach().numpy(), 0, H - 1)
        scanpaths = [np.vstack([X[i, :mask[i].sum().int().item()], Y[i, :mask[i].sum().int().item()], T[i, :mask[i].sum().int().item()]]).T for i in range(n_simulations)]
        return scanpaths

    def get_features(
        self, inter_times: torch.Tensor, marks: torch.Tensor
    ) -> torch.Tensor:
        features = self.context.get_features(inter_times, marks)
        return features

    def get_context(
        self, features: torch.Tensor, remove_last: bool = True
    ) -> torch.Tensor:
        return self.context.forward(features, remove_last)

    def get_metadata(self, images: torch.Tensor) -> torch.Tensor:
        return self.metadata.forward(images)

    def get_inter_time_dist(self, context: torch.Tensor) -> Distribution:
        return self.inter_time_dist.forward(context, None)

    def get_marks_dist(
        self, context: torch.Tensor, temperature: Optional[float] = None
    ) -> Distribution:
        return self.marks_dist.forward(context, temperature)

    def sample(
        self,
        t_end: float,
        img: torch.Tensor,
        context_init: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        batch_size: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context_init = context_init.view(self.context.get_context_init().shape[-1]) if context_init is not None else self.context.get_context_init()
        
        history = context_init[None, None, :].expand(batch_size, 1, -1)
        imgs = img.repeat(batch_size, 1, 1, 1)
        
        metadata = self.get_metadata(imgs)
        next_context = self.merge.forward([history, metadata])
        
        inter_times = torch.empty(batch_size, 0, device=self.device)
        
        marks = torch.empty(batch_size, 0, 2, device=self.device, dtype=torch.float)
        
        generated = False

        while not generated:
            next_inter_times = self.get_inter_time_dist(next_context).sample()

            inter_times = torch.cat([inter_times, next_inter_times], dim=1)
            
            curr_mark = self.get_marks_dist(next_context, self.temperature).sample()

            marks = torch.cat([marks, curr_mark], dim=1)
            
            with torch.no_grad():
                generated = inter_times.sum(-1).min() >= t_end
            
            features = self.get_features(
                inter_times, marks
            )  # (batch_size, seq_len, num_features)

            history = self.get_context(
                features, remove_last=False
            )  # (batch_size, seq_len, context_size)

            metadata = self.get_metadata(imgs)

            context = self.merge.forward([history, metadata])

            next_context = context[:, [-1], :]  # (batch_size, 1, context_size)

        arrival_times = inter_times.cumsum(-1)  # (batch_size, seq_len)
        inter_times = diff(arrival_times.clamp(max=t_end), dim=-1)
        mask = (arrival_times <= t_end).float()  # (batch_size, seq_len)
        mark_msk = torch.stack([mask, mask], dim=2)
        marks = marks * mark_msk  # (batch_size, seq_len)

        return inter_times, marks, mask
