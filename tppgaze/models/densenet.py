import torch
import torch.nn as nn

class RGBDenseNet201(nn.Sequential):
    def __init__(self):
        super(RGBDenseNet201, self).__init__()
        self.densenet = torch.hub.load(
            "pytorch/vision:v0.10.0",
            "densenet201",
            weights="DenseNet201_Weights.DEFAULT",
        )
        super(RGBDenseNet201, self).__init__(self.densenet)

class FeatureExtractor(nn.Module):
    def __init__(self, features: nn.Sequential, targets):
        super().__init__()
        self.features = features
        self.targets = targets
        self.outputs = {}
        for target in targets:
            layer: nn.Module = dict([*self.features.named_modules()])[target]
            layer.register_forward_hook(self.save_outputs_hook(target))

    def save_outputs_hook(self, layer_id: str):
        def fn(_, __, output):
            self.outputs[layer_id] = output.clone()
        return fn

    def forward(self, x):
        self.outputs.clear()
        _ = self.features(x)
        return [self.outputs[target] for target in self.targets]

class DenseNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = FeatureExtractor(
            RGBDenseNet201(),
            [
                "0.features.denseblock4.denselayer32.norm1",
                "0.features.denseblock4.denselayer32.conv1",
                "0.features.denseblock4.denselayer31.conv2",
            ],
        )
        for param in self.features.parameters():
            param.requires_grad = False
        self.features.eval()
    
    def get_features(self, batch) -> torch.Tensor:
        return batch.stimuli
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features.forward(images)
        x = torch.cat(x, dim=1)
        return x