"""
model.py – MobileNetV3-Small classifier for event-camera frames.

The standard MobileNetV3-Small expects 3-channel RGB input.  We modify:
  1. The first Conv2d to accept `in_channels` (default 2 for two_channel repr).
  2. The final classifier Linear to output `num_classes` (default 24).

The rest of the network (all inverted-residual blocks, SE modules, hard-swish
activations) is unchanged, so pretrained weights from ImageNet can optionally
be used for the depthwise body while only the first conv and head are
re-initialised (see `pretrained` argument).

Usage
-----
    from model import build_model
    net = build_model(in_channels=2, num_classes=24, pretrained=False)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm
from torchvision.models import MobileNet_V3_Small_Weights


def build_model(
    in_channels: int = 2,
    num_classes: int = 24,
    pretrained: bool = False,
) -> nn.Module:
    """
    Build a MobileNetV3-Small modified for event-camera classification.

    Parameters
    ----------
    in_channels : number of input channels (2 for two_channel, 1 for signed,
                  N for voxel_grid).
    num_classes : number of output classes (24 for 24-letter ASL).
    pretrained  : if True, load ImageNet weights for the MobileNetV3-Small
                  backbone and only re-initialise the first conv and head.
                  Useful when in_channels == 3; for other channel counts
                  only the body benefits.

    Returns
    -------
    nn.Module  (MobileNetV3-Small variant)
    """
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    net = tvm.mobilenet_v3_small(weights=weights)

    # --- Patch first convolution -------------------------------------------
    # Original: Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
    orig_conv = net.features[0][0]
    if in_channels != 3:
        new_conv = nn.Conv2d(
            in_channels,
            orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=False,
        )
        # Kaiming init (same as torchvision default for this layer)
        nn.init.kaiming_normal_(new_conv.weight, mode="fan_out",
                                nonlinearity="relu")
        net.features[0][0] = new_conv

    # --- Patch classifier head ---------------------------------------------
    # Original head: Linear(576, 1000)
    # We keep the 576->1024 hidden layer and replace the output layer.
    in_features = net.classifier[-1].in_features
    net.classifier[-1] = nn.Linear(in_features, num_classes)
    nn.init.normal_(net.classifier[-1].weight, std=0.01)
    nn.init.zeros_(net.classifier[-1].bias)

    return net


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
