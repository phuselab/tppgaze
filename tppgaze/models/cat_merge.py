import torch
from typing import List
import torch.nn as nn

class CatMerge(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        [context, metadata] = features
        metadata = metadata.repeat(1, context.shape[1], 1)
        return torch.cat([context, metadata], dim=2)
