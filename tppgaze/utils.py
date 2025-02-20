import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from typing import Optional, List

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from PIL import Image

def pad_sequence(
    sequences: List[torch.Tensor],
    padding_value: float = 0,
    max_len: Optional[int] = None,
):
    r"""Pad a list of variable length Tensors with ``padding_value``"""
    dtype = sequences[0].dtype
    device = sequences[0].device
    seq_shape = sequences[0].shape
    trailing_dims = seq_shape[1:]
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    out_dims = (len(sequences), max_len) + trailing_dims

    out_tensor = torch.empty(*out_dims, dtype=dtype, device=device).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        out_tensor[i, :length, ...] = tensor

    return out_tensor


def diff(x, dim: int = -1):
    """Inverse of x.cumsum(dim=dim).
    Compute differences between subsequent elements of the tensor.
    Args:
        x: Input tensor of arbitrary shape.
        dim: Dimension over which to compute the difference, {-2, -1}.
    Returns:
        diff: Tensor of the the same shape as x.
    """
    if dim == -1:
        return x - F.pad(x, (1, 0))[..., :-1]
    elif dim == -2:
        return x - F.pad(x, (0, 0, 1, 0))[..., :-1, :]
    else:
        raise ValueError("dim must be equal to -1 or -2")


def clamp_preserve_gradients(x: torch.Tensor, min: float, max: float) -> torch.Tensor:
    """Clamp the tensor while preserving gradients in the clamped region."""
    return x + (x.clamp(min, max) - x).detach()


def rescale(current_min, current_max, min, max, value):
    return ((value - current_min) / (current_max - current_min)) * (max - min) + min


def aggregate(values: List[torch.Tensor], lengths: List[torch.Tensor]):
    """Calculate masked average of values.

    Sequences may have different lengths, so it's necessary to exclude
    the masked values in the padded sequence when computing the average.

    Arguments:
        values (List[tensor]): List of batches where each batch contains
            padded values, shape (batch size, sequence length)
        lengths (List[tensor]): List of batches where each batch contains
            lengths of sequences in a batch, shape (batch size)

    Returns:
        mean (float): Average value in values taking padding into account
    """

    total = 0.0
    for batch, length in zip(values, lengths):
        length = length.long()
        mask = torch.arange(batch.shape[1])[None, :] < length[:, None]
        mask = mask.float()

        batch[torch.isnan(batch)] = 0  # set NaNs to 0
        batch *= mask

        total += batch.sum()

    total_length = sum([x.sum() for x in lengths])

    return total / total_length


_preprocess = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

def get_preprocess():
    return _preprocess

def preprocess_image(path: str):
    image = Image.open(path)
    image = image.convert("RGB")
    w, h = image.size
    image_tensor: torch.Tensor = get_preprocess()(image)  # type: ignore
    return image_tensor, h, w

def visualize_scanpaths(image_path, scanpaths):
    for scanpath in scanpaths:
        fig, axs = plt.subplots()
        axs.grid(False)
        
        image = Image.open(image_path)
        axs.imshow(image, interpolation="none")

        x = scanpath[:, 0]
        y = scanpath[:, 1]
        t = scanpath[:, 2]
        draw_scanpath(axs, x, y, t)
        plt.axis("off")
        plt.show()

def draw_scanpath(
    ax: Axes,
    fix_x: List[float],
    fix_y: List[float],
    fix_d: List[float],
    alpha=1,
    invert_y=False,
    ydim=None,
):    
    # COLOURS
    # all colours are from the Tango colourmap, see:
    # http://tango.freedesktop.org/Tango_Icon_Theme_Guidelines#Color_Palette
    COLORS = {
        "orange": ["#fcaf3e", "#f57900", "#ce5c00"],
        "skyblue": ["#729fcf", "#3465a4", "#204a87"],
        "scarletred": ["#ef2929", "#cc0000", "#a40000"],
        "aluminium": ["#eeeeec", "#d3d7cf", "#babdb6", "#888a85", "#555753", "#2e3436"],
    }

    if invert_y:
        if ydim is None:
            raise RuntimeError("ydim must be provided")
        fix_y = ydim - 1 - fix_y

    for i in range(1, len(fix_x)):
        ax.arrow(
            fix_x[i - 1],
            fix_y[i - 1],
            fix_x[i] - fix_x[i - 1],
            fix_y[i] - fix_y[i - 1],
            alpha=alpha,
            fc=COLORS["orange"][0],
            ec=COLORS["orange"][0],
            fill=True,
            shape="full",
            width=3,
            head_width=0,
            head_starts_at_zero=False,
            overhang=0,
        )

    if len(fix_d) == 0:
        fix_d = [200] * len(fix_x) # default duration for visualization
    
    for i in range(len(fix_x)):
        color = COLORS["aluminium"][0]
        if i == 0:
            color = COLORS["skyblue"][0]
        elif i == len(fix_x) - 1:
            color = COLORS["scarletred"][0]

        ax.plot(
            fix_x[i],
            fix_y[i],
            marker="o",
            ms=fix_d[i] / 10,
            mfc=color,
            mec="black",
            alpha=0.7,
        )

    for i in range(len(fix_x)):
        ax.text(
            fix_x[i] - 4,
            fix_y[i] + 1,
            str(i + 1),
            color="black",
            ha="left",
            va="center",
            multialignment="center",
            alpha=alpha,
            fontsize=14,
        )