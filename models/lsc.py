"""Large spatial calibration encoder used by PLNet."""

from functools import partial

import torch
import torch.nn as nn
from timm.models.layers import DropPath


class LayerNorm3D(nn.Module):
    """Layer normalization for channel-first 3D feature maps."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        return self.weight[:, None, None, None] * x + self.bias[:, None, None, None]


class SpatialCalibrationModule(nn.Module):
    """Select local and dilated spatial features with pooled descriptors."""

    def __init__(self, channels):
        super().__init__()
        if channels % 2:
            raise ValueError(f"Spatial calibration requires even channels, got {channels}")
        self.conv0 = nn.Conv3d(
            channels,
            channels,
            kernel_size=5,
            padding=2,
            groups=channels,
        )
        self.conv_spatial = nn.Conv3d(
            channels,
            channels,
            kernel_size=7,
            padding=9,
            dilation=3,
            groups=channels,
        )
        self.conv1 = nn.Conv3d(channels, channels // 2, kernel_size=1)
        self.conv2 = nn.Conv3d(channels, channels // 2, kernel_size=1)
        self.conv_squeeze = nn.Conv3d(3, 2, kernel_size=7, padding=3)
        self.conv = nn.Conv3d(channels // 2, channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        local_features = self.conv0(x)
        local = self.conv1(local_features)
        dilated = self.conv2(self.conv_spatial(local_features))
        features = torch.cat([local, dilated], dim=1)
        descriptor = torch.cat(
            [
                torch.mean(features, dim=1, keepdim=True),
                torch.max(features, dim=1, keepdim=True).values,
                torch.std(features, dim=1, keepdim=True),
            ],
            dim=1,
        )
        masks = torch.sigmoid(self.act(self.conv_squeeze(descriptor)))
        attention = local * masks[:, 0:1] + dilated * masks[:, 1:2]
        return x * self.conv(attention)


class SpatialAttention(nn.Module):
    """Residual spatial calibration projection."""

    def __init__(self, channels):
        super().__init__()
        self.spatial_gating_unit = SpatialCalibrationModule(channels)
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x):
        return x + self.proj(self.spatial_gating_unit(x))


class LSCBlock(nn.Module):
    """Large spatial calibration followed by multi-kernel depthwise fusion."""

    def __init__(
        self,
        channels,
        drop=0.0,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
    ):
        super().__init__()
        self.attn = SpatialAttention(channels)
        self.dwc1 = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.dwc2 = nn.Conv3d(
            channels,
            channels,
            kernel_size=5,
            padding=2,
            groups=channels,
            bias=False,
        )
        self.dwc3 = nn.Conv3d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
            bias=False,
        )
        self.pwc = nn.Conv3d(channels, channels, kernel_size=1)
        self.conv1x1x1 = nn.Conv3d(channels * 3, channels, kernel_size=1, bias=False)
        self.layer_scale1 = nn.Parameter(
            layer_scale_init_value * torch.ones(channels),
        )
        # Kept for compatibility with the original released checkpoint.
        self.layer_scale2 = nn.Parameter(
            layer_scale_init_value * torch.ones(channels),
        )
        self.drop = nn.Dropout(drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.ln = LayerNorm3D(channels)
        self.act = nn.GELU()

    def _branch(self, convolution, x):
        return self.pwc(self.act(convolution(x)))

    def forward(self, x):
        shortcut = x
        calibrated = self.attn(x)
        scale = self.layer_scale1[:, None, None, None]
        x = shortcut + self.drop_path(scale * calibrated)
        shortcut = x
        normalized = self.ln(x)
        multi_scale = torch.cat(
            [
                self._branch(self.dwc1, normalized),
                self._branch(self.dwc2, normalized),
                self._branch(self.dwc3, normalized),
            ],
            dim=1,
        )
        return shortcut + self.drop_path(self.conv1x1x1(multi_scale))


class LSCEncoder(nn.Module):
    """Three-stage LSC encoder with original checkpoint-compatible names."""

    def __init__(
        self,
        in_channels=32,
        dims=(64, 128, 256),
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        out_indices=(0, 1, 2),
    ):
        super().__init__()
        if len(dims) != 3:
            raise ValueError(f"LSCEncoder requires three stages, got {len(dims)}")
        self.num_layers = len(dims)
        self.out_indices = tuple(out_indices)

        self.ds_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(
                        in_channels,
                        dims[0],
                        kernel_size=5,
                        stride=2,
                        padding=2,
                    ),
                    LayerNorm3D(dims[0]),
                ),
                nn.Sequential(
                    LayerNorm3D(dims[0]),
                    nn.Conv3d(dims[0], dims[1], kernel_size=2, stride=2),
                ),
                nn.Sequential(
                    LayerNorm3D(dims[1]),
                    nn.Conv3d(dims[1], dims[2], kernel_size=2, stride=2),
                ),
            ],
        )
        drop_path_rates = torch.linspace(0, drop_path_rate, len(dims)).tolist()
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    LSCBlock(
                        channels=channels,
                        drop_path=drop_path_rates[index],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                )
                for index, channels in enumerate(dims)
            ],
        )

        norm_layer = partial(LayerNorm3D, eps=1e-6)
        for index, channels in enumerate(dims):
            self.add_module(f"Norm{index}", norm_layer(channels))

    def forward(self, x):
        outputs = []
        for index in range(self.num_layers):
            x = self.ds_layers[index](x)
            x = self.stages[index](x)
            if index in self.out_indices:
                outputs.append(getattr(self, f"Norm{index}")(x))
        return tuple(outputs)
