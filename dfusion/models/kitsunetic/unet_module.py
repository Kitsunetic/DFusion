"""
from https://github.com/openai/guided-diffusion

MIT License

Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import math
from abc import abstractmethod

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from dfusion.models.kitsunetic.attention import QKVAttention, QKVAttentionLegacy, SpatialTransformer
from dfusion.models.kitsunetic.modules import *
from dfusion.models.kitsunetic.utils import *
from dfusion.utils.indexing import unsqueeze_as


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb=None):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb=None, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class TransposedUpsample(nn.Module):
    "Learned 2x upsampling without padding"

    def __init__(self, channels, out_channels=None, ks=5):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.up = nn.ConvTranspose2d(self.channels, self.out_channels, kernel_size=ks, stride=2)

    def forward(self, x):
        return self.up(x)


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=2, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=2, stride=2)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels=None,
        dropout=0.0,
        out_channels=None,
        num_groups=32,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels, num_groups),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = None
        if emb_channels != None:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                linear(
                    emb_channels,
                    2 * self.out_channels if use_scale_shift_norm else self.out_channels,
                ),
            )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels, num_groups),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x, emb)
        else:
            return self._forward(x, emb)

    def _forward(self, x, emb=None):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        if self.emb_layers is not None:
            emb_out = self.emb_layers(emb)  # .type(h.dtype)
            emb_out = unsqueeze_as(emb_out, x)

            if self.use_scale_shift_norm:
                out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
                scale, shift = th.chunk(emb_out, 2, dim=1)
                h = out_norm(h) * (1 + scale) + shift
                h = out_rest(h)

            else:
                h = h + emb_out
                h = self.out_layers(h)

        else:
            h = self.out_layers(h)

        return self.skip_connection(x) + h


class StackedResBlocks(TimestepEmbedSequential):
    def __init__(self, num_blocks, *args, **kwargs):
        m = [ResBlock(*args, **kwargs)]
        kwargs.update({"up": False, "down": False, "channels": m[0].out_channels})
        for _ in range(num_blocks):
            m.append(ResBlock(*args, **kwargs))

        super().__init__(*m)


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        attention_type="qkv_legacy",  # qkv, qkv_legacy, xformers, flash, flash16
        num_groups=32,
    ):
        super().__init__()
        self.channels = channels
        self.attention_type = attention_type
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels, num_groups)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if attention_type == "qkv":
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        elif attention_type == "qkv_legacy":
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)
        elif attention_type == "xformers":
            from xformers.components.attention import ScaledDotProduct

            self.attention = ScaledDotProduct()
        elif attention_type in ("memory_efficient_attention", "flash", "flash16"):
            from xformers.ops import memory_efficient_attention

            self.attention = memory_efficient_attention
        else:
            raise NotImplementedError(attention_type)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        # return checkpoint(
        #     self._forward, (x,), self.parameters(), False
        # )  # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        # return pt_checkpoint(self._forward, x)  # pytorch
        if self.training and self.use_checkpoint:
            return checkpoint(self._forward, x)
        else:
            return self._forward(x)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))

        if self.attention_type in ("qkv", "qkv_legacy"):
            h = self.attention(qkv)
        else:
            # q, k, v = rearrange(qkv, "b (x h c) l -> x b l h c", x=3, h=self.num_heads)
            q, k, v = rearrange(qkv, "b (x h c) l -> x (b h) l c", x=3, h=self.num_heads).contiguous()
            # q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
            if self.attention_type == "flash16" and q.device.type != "cpu":
                dtype = q.dtype
                q, k, v = q.half(), k.half(), v.half()
            h = self.attention(q, k, v)
            if self.attention_type == "flash16" and q.device.type != "cpu":
                h = h.to(dtype)
            # h = rearrange(h, "b l h c -> b (h c) l").contiguous()
            h = rearrange(h, "(b h) l c -> b (h c) l", h=self.num_heads).contiguous()

        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)
