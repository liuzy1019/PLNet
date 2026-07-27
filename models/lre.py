"""Learnable residual enhancement decoder used by PLNet."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.convolutions import Convolution


def _resample(x, scale):
    downsampled = F.interpolate(
        x,
        size=[int(size * scale) for size in x.shape[2:]],
        mode="trilinear",
        align_corners=False,
    )
    return F.interpolate(
        downsampled,
        size=x.shape[2:],
        mode="trilinear",
        align_corners=False,
    )


class LearnableResidualEnhancement(nn.Module):
    """Fuse an encoder skip with multi-scale edge and residual enhancement."""

    def __init__(self, in_channels, out_channels, spatial_dims=3):
        super().__init__()
        if out_channels < 8 or out_channels % 8:
            raise ValueError(
                f"LRE out_channels must be a positive multiple of 8, got {out_channels}"
            )
        self.upc = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=1,
            kernel_size=1,
            bias=False,
            conv_only=True,
        )
        self.edge_conv = nn.Sequential(
            nn.Conv3d(
                out_channels * 2,
                out_channels * 2,
                kernel_size=3,
                padding=1,
                groups=out_channels * 2,
            ),
            nn.Conv3d(out_channels * 2, out_channels, kernel_size=1),
        )
        self.repr_mldw = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=out_channels,
                out_channels=out_channels,
                strides=1,
                kernel_size=3,
                bias=False,
                conv_only=True,
                groups=out_channels // 8,
            ),
            nn.Hardswish(inplace=True),
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=out_channels,
                out_channels=out_channels,
                strides=1,
                kernel_size=1,
                bias=False,
                conv_only=True,
            ),
        )
        self.norm = nn.InstanceNorm3d(out_channels)
        self.group_skip_scale = nn.Parameter(
            torch.ones(1, out_channels, 1, 1, 1),
        )
        self.group_res_scale = nn.Parameter(torch.ones(1))

    def forward(self, inputs):
        x, skip = inputs
        x = self.upc(x)
        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        x = x + skip * self.group_skip_scale

        feature_075 = _resample(x, 0.75)
        feature_050 = _resample(x, 0.50)
        feature_025 = _resample(x, 0.25)
        edge_descriptor = torch.cat(
            [
                torch.abs(feature_075 - feature_050),
                torch.abs(feature_050 - feature_025),
            ],
            dim=1,
        )
        x = x + self.edge_conv(edge_descriptor)
        return self.repr_mldw(self.norm(x)) + x * self.group_res_scale
