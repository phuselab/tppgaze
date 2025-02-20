from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

from tppgaze.models.coordconv import AddCoords
from tppgaze.models.densenet import DenseNet

class DG3ishBase(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        channels1: int,
        channels2: int,
        channels3: int,
        output_dim: int,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.gelu,
    ):
        super(DG3ishBase, self).__init__()
        self.backbone = backbone

        self.channels1 = channels1
        self.channels2 = channels2
        self.channels3 = channels3
        self.output_dim = output_dim

        self.coordconv = AddCoords(rank=2, with_r=True)

        self.conv1 = nn.Conv2d(2051, self.channels1, 1, 1, bias=False)
        self.image_norm1 = nn.BatchNorm2d(self.channels1)
        self.conv_activation1 = nn.Softplus()

        self.conv2 = nn.Conv2d(self.channels1, self.channels2, 1, 1, bias=False)
        self.image_norm2 = nn.BatchNorm2d(self.channels2)
        self.conv_activation2 = nn.Softplus()

        self.conv3 = nn.Conv2d(self.channels2, self.channels3, 1, 1, bias=False)
        self.image_norm3 = nn.BatchNorm2d(self.channels3)
        self.conv_activation3 = nn.Softplus()

        self.flatten = nn.Flatten(2)

        self.fc1 = nn.Linear(256, self.output_dim)
        self.activation = activation
    
    def get_features(self, batch) -> torch.Tensor:
        return batch.stimuli
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device

        x = self.backbone(images)

        x = self.coordconv(x, device)
        x = self.conv_activation1(self.image_norm1(self.conv1(x)))
        x = self.conv_activation2(self.image_norm2(self.conv2(x)))
        x = self.conv_activation3(self.image_norm3(self.conv3(x)))
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.activation(x)
        return x

class DenseNetDG3ish(DG3ishBase):
    def __init__(
        self,
        channels1: int = 8,
        channels2: int = 16,
        channels3: int = 1,
        output_dim: int = 256,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.gelu,
    ):
        super().__init__(
            DenseNet(), channels1, channels2, channels3, output_dim, activation
        )