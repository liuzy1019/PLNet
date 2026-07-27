from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_3tuple, trunc_normal_

from .lre import LearnableResidualEnhancement
from .lsc import LSCEncoder


class IRB(nn.Module):
    """
    MLP Layer
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        ksize=3,
        act_layer=nn.Hardswish,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(in_features, hidden_features, 1, 1, 0)
        self.act = act_layer()
        self.conv = nn.Conv3d(
            hidden_features,
            hidden_features,
            kernel_size=ksize,
            padding=ksize // 2,
            stride=1,
            groups=hidden_features,
        )
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1, 1, 0)
    def forward(self, x, D, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, D, H, W)
        x = self.fc1(x)
        x = self.act(x)
        x = self.conv(x)
        x = self.act(x)
        x = self.fc2(x)
        return x.reshape(B, C, -1).permute(0, 2, 1)


class PoolingAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=2,
        qkv_bias=False,
        pool_ratios=(1, 2, 3, 6),
    ):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias))
        self.kv = nn.Sequential(nn.Linear(dim, dim * 2, bias=qkv_bias))

        self.proj = nn.Linear(dim, dim)
        self.pool_ratios = pool_ratios
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, D, H, W, d_convs):
        if len(d_convs) != len(self.pool_ratios):
            raise ValueError(
                f"Expected {len(self.pool_ratios)} pooling convolutions, got {len(d_convs)}"
            )
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        pools = []
        x_ = x.permute(0, 2, 1).reshape(B, C, D, H, W)

        for pool_ratio, depthwise_conv in zip(self.pool_ratios, d_convs):
            pool = F.adaptive_avg_pool3d(
                x_, (round(D / pool_ratio), round(H / pool_ratio), round(W / pool_ratio))
            )
            pool = pool + depthwise_conv(pool)
            pools.append(pool.view(B, C, -1))
        pools = torch.cat(pools, dim=2)
        pools = self.norm(pools.permute(0, 2, 1))

        kv = (
            self.kv(pools)
            .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x.transpose(1, 2).contiguous().reshape(B, N, C)

        x = self.proj(x)
        return x


class Block(nn.Module):
    """
    P2T Block
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        pool_ratios=(12, 16, 20, 24),
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PoolingAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            pool_ratios=pool_ratios,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = IRB(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=nn.Hardswish,
            ksize=3,
        )

    def forward(self, x, D, H, W, d_convs=None):
        x = x + self.drop_path(self.attn(self.norm1(x), D, H, W, d_convs=d_convs))
        x = x + self.drop_path(self.mlp(self.norm2(x), D, H, W))
        return x


class PatchEmbed3D(nn.Module):
    """
    Volume to Patch Embedding
    """

    def __init__(
        self,
        vol_size=(32, 32, 32),
        patch_size=(2, 2, 2),
        kernel_size=3,
        in_chans=1,
        embed_dim=768,
        overlap=True,
    ):
        super().__init__()
        vol_size = to_3tuple(vol_size)
        patch_size = to_3tuple(patch_size)

        self.patch_size = patch_size
        if any(size % patch for size, patch in zip(vol_size, patch_size)):
            raise ValueError(
                f"vol_size {vol_size} must be divisible by patch_size {patch_size}"
            )
        if not overlap:
            self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        else:
            self.proj = nn.Conv3d(
                in_chans,
                embed_dim,
                kernel_size=kernel_size,
                stride=patch_size,
                padding=kernel_size // 2,
            )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, _, D, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        D, H, W = D // self.patch_size[0], H // self.patch_size[1], W // self.patch_size[2]

        return x, (D, H, W)


class PyramidPoolingTransformer3D(nn.Module):
    def __init__(
        self,
        vol_size=(80, 160, 160),
        patch_size=2,
        in_chans=1,
        embed_dims=(64, 128, 256),
        num_heads=(1, 2, 4),
        mlp_ratios=(8, 8, 4),
        qkv_bias=True,
        drop_path_rate=0.1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=(2, 2, 9),
    ):

        super().__init__()

        pool_ratios = [[12, 16, 20, 24], [6, 8, 10, 12], [3, 4, 5, 6]]
        self.patch_embed1 = PatchEmbed3D(
            vol_size=vol_size,
            patch_size=patch_size,
            kernel_size=7,
            in_chans=in_chans,
            embed_dim=embed_dims[0],
            overlap=True,
        )

        self.patch_embed2 = PatchEmbed3D(
            vol_size=tuple(np.array(vol_size) // 2),
            patch_size=patch_size,
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1],
            overlap=True,
        )

        self.patch_embed3 = PatchEmbed3D(
            vol_size=tuple(np.array(vol_size) // 4),
            patch_size=patch_size,
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2],
            overlap=True,
        )

        self.d_convs1 = nn.ModuleList(
            [
                nn.Conv3d(
                    embed_dims[0],
                    embed_dims[0],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=embed_dims[0],
                )
                for _ in pool_ratios[0]
            ]
        )
        self.d_convs2 = nn.ModuleList(
            [
                nn.Conv3d(
                    embed_dims[1],
                    embed_dims[1],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=embed_dims[1],
                )
                for _ in pool_ratios[1]
            ]
        )
        self.d_convs3 = nn.ModuleList(
            [
                nn.Conv3d(
                    embed_dims[2],
                    embed_dims[2],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=embed_dims[2],
                )
                for _ in pool_ratios[2]
            ]
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        self.block1 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[0],
                    num_heads=num_heads[0],
                    mlp_ratio=mlp_ratios[0],
                    qkv_bias=qkv_bias,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    pool_ratios=pool_ratios[0],
                )
                for i in range(depths[0])
            ]
        )
        cur += depths[0]

        self.block2 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[1],
                    num_heads=num_heads[1],
                    mlp_ratio=mlp_ratios[1],
                    qkv_bias=qkv_bias,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    pool_ratios=pool_ratios[1],
                )
                for i in range(depths[1])
            ]
        )
        cur += depths[1]

        self.block3 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[2],
                    num_heads=num_heads[2],
                    mlp_ratio=mlp_ratios[2],
                    qkv_bias=qkv_bias,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    pool_ratios=pool_ratios[2],
                )
                for i in range(depths[2])
            ]
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        outs = []
        B = x.shape[0]

        x, (D, H, W) = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, D, H, W, self.d_convs1)
        x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3)
        outs.append(x)

        x, (D, H, W) = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, D, H, W, self.d_convs2)
        x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3)
        outs.append(x)

        x, (D, H, W) = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, D, H, W, self.d_convs3)
        x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3)
        outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)

        return x

class Conv3d(nn.Conv3d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=(1, 1, 1),
        padding=(0, 0, 0),
        dilation=(1, 1, 1),
        groups=1,
        bias=False,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )

    def forward(self, x):
        weight = self.weight
        weight_mean = (
            weight.mean(dim=1, keepdim=True)
            .mean(dim=2, keepdim=True)
            .mean(dim=3, keepdim=True)
            .mean(dim=4, keepdim=True)
        )
        weight = weight - weight_mean
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(
            -1, 1, 1, 1, 1
        )
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3x3(
    in_planes,
    out_planes,
    kernel_size=(3, 3, 3),
    stride=(1, 1, 1),
    padding=(1, 1, 1),
    dilation=(1, 1, 1),
    bias=False,
    weight_std=False,
):
    """Create a 3D convolution, optionally with weight standardization."""
    if weight_std:
        return Conv3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
    else:
        return nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )


class NoBottleneck(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        stride=(1, 1, 1),
        dilation=(1, 1, 1),
        downsample=None,
        fist_dilation=1,
        multi_grid=1,
        weight_std=False,
    ):

        super().__init__()
        self.weight_std = weight_std

        self.gn1 = nn.GroupNorm(8, in_planes)
        self.conv1 = conv3x3x3(
            in_planes,
            out_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=dilation * multi_grid,
            dilation=dilation * multi_grid,
            bias=False,
            weight_std=self.weight_std,
        )

        self.hardswish = nn.Hardswish(inplace=True)

        self.downsample = downsample
    def forward(self, x):
        skip = x
        seg = self.gn1(x)
        seg = self.hardswish(seg)
        seg = self.conv1(seg)

        if self.downsample is not None:
            skip = self.downsample(x)

        seg = seg + skip
        return seg


class PLNetModel(nn.Module):
    """
    PLNet with P2T and LSC encoders, LRE decoder, and deep supervision.
    """

    def __init__(
        self,
        shape,
        block,
        layers,
        num_classes=8,
        weight_std=False,
    ):
        super().__init__()
        if len(shape) != 3 or any(size < 24 or size % 8 for size in shape):
            raise ValueError(
                f"PLNet input shape must contain three multiples of 8, each at least 24: {shape}"
            )
        self.shape = tuple(shape)
        self.weight_std = weight_std

        # Stem
        self.stem = nn.Sequential(
            conv3x3x3(1, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), weight_std=self.weight_std),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )

        # P2T encoder
        self.p2t3d = PyramidPoolingTransformer3D(
            vol_size=self.shape,
            patch_size=2,
            in_chans=1,
            embed_dims=[64, 128, 256],
            num_heads=[1, 2, 4],
            mlp_ratios=[8, 8, 4],
            qkv_bias=True,
            drop_path_rate=0.0,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 6],
        )

        # LSC encoder
        self.lsk = LSCEncoder(
            in_channels=32,
            dims=[64, 128, 256],
            drop_path_rate=0.0,
            layer_scale_init_value=1e-6,
        )

        # Fusion
        self.layer0 = self._make_layer(block, 32, 32, layers[0], stride=(1, 1, 1))
        self.layer1 = self._make_layer(block, 64, 64, layers[1], stride=(1, 1, 1))
        self.layer2 = self._make_layer(block, 128, 128, layers[2], stride=(1, 1, 1))
        self.layer3 = self._make_layer(
            block, 256, 256, layers[3], stride=(1, 1, 1), dilation=(2, 2, 2)
        )

        # LRE decoder
        up_block = partial(LearnableResidualEnhancement, spatial_dims=3)
        self.up3 = up_block(in_channels=256, out_channels=128)
        self.up2 = up_block(in_channels=128, out_channels=64)
        self.up1 = up_block(in_channels=64, out_channels=32)

        # Segmentation heads
        expr = 2
        self.seg_cls = nn.Sequential(
            nn.GroupNorm(32, 32),
            nn.Conv3d(32, 32 * expr, 1),
            nn.Hardswish(inplace=True),
            nn.Conv3d(32 * expr, num_classes, 1),
        )

        self.d1_cls = nn.Sequential(nn.Conv3d(32, num_classes, kernel_size=1))
        self.d2_cls = nn.Sequential(nn.Conv3d(64, num_classes, kernel_size=1))
        self.d3_cls = nn.Sequential(nn.Conv3d(128, num_classes, kernel_size=1))

    def _make_layer(
        self, block, inplanes, outplanes, blocks, stride=(1, 1, 1), dilation=(1, 1, 1), multi_grid=1
    ):
        downsample = None
        if stride[0] != 1 or stride[1] != 1 or stride[2] != 1 or inplanes != outplanes:
            downsample = nn.Sequential(
                nn.GroupNorm(8, inplanes),
                nn.Hardswish(inplace=True),
                conv3x3x3(
                    inplanes,
                    outplanes,
                    kernel_size=(1, 1, 1),
                    stride=stride,
                    padding=(0, 0, 0),
                    weight_std=self.weight_std,
                ),
            )

        layers = []

        def generate_multi_grid(index, grids):
            return grids[index % len(grids)] if isinstance(grids, tuple) else 1

        layers.append(
            block(
                inplanes,
                outplanes,
                stride,
                dilation=dilation,
                downsample=downsample,
                multi_grid=generate_multi_grid(0, multi_grid),
                weight_std=self.weight_std,
            )
        )

        for i in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    outplanes,
                    dilation=dilation,
                    multi_grid=generate_multi_grid(i, multi_grid),
                    weight_std=self.weight_std,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, inputs):
        x = inputs
        x = self.stem(x)
        x = self.layer0(x)
        skip0 = x

        # Encoder
        out_p2t = self.p2t3d(inputs)
        out_lsk = self.lsk(skip0)

        x = out_lsk[0] + out_p2t[0]
        x = self.layer1(x)
        skip1 = x

        x = out_lsk[1] + out_p2t[1]
        x = self.layer2(x)
        skip2 = x

        x = out_lsk[2] + out_p2t[2]
        x = self.layer3(x)
        del out_p2t, out_lsk

        # Decoder
        d3 = self.up3([x, skip2])
        del skip2

        d2 = self.up2([d3, skip1])
        del skip1

        d1 = self.up1([d2, skip0])
        del skip0

        # Segmentation and deep-supervision predictions
        seg = self.seg_cls(d1)
        d1 = self.d1_cls(d1)
        d2 = self.d2_cls(d2)
        d3 = self.d3_cls(d3)

        d1 = F.interpolate(
            d1,
            size=self.shape,
            mode="trilinear",
            align_corners=True,
        )
        d2 = F.interpolate(
            d2,
            size=self.shape,
            mode="trilinear",
            align_corners=True,
        )
        d3 = F.interpolate(
            d3,
            size=self.shape,
            mode="trilinear",
            align_corners=True,
        )

        return [seg, d1, d2, d3]


def PLNet(shape, num_classes=8, weight_std=True):
    model = PLNetModel(shape, NoBottleneck, [1, 2, 2, 2], num_classes, weight_std)
    return model
