"""
from https://github.com/CompVis/latent-diffusion

MIT License

Copyright (c) 2022 Machine Vision and Learning Group, LMU Munich

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
from dfusion.models.kitsunetic.unet_module import *


class AdaGRN(nn.Module):
    def __init__(self, dims, dim, hidden_dim, out_dim, emb_dim, dropout) -> None:
        super().__init__()

        self.emb_layers = None
        if emb_dim != None:
            self.emb_layers = nn.Sequential(nn.GELU(), linear(emb_dim, 2 * hidden_dim))
        else:
            self.weight = nn.Parameter(th.zeros(1, hidden_dim))
            self.bias = nn.Parameter(th.zeros(1, hidden_dim))

        self.conv1 = conv_nd(dims, dim, hidden_dim, 1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.conv2 = conv_nd(dims, hidden_dim, out_dim, 1)

    def forward(self, x: Tensor, emb: Tensor):
        x = self.conv1(x)  # (N, H, ...)
        x = self.act(x)
        x = self.drop(x)

        x_g = x.flatten(2).norm(dim=-1)  # (N, H)
        x_n = x_g / (th.mean(x_g, dim=-1, keepdim=True) + 1e-6)  # (N, H)
        x_n = unsqueeze_as(x_n, x)  # (N, H, ...)
        if self.emb_layers is not None:
            emb = self.emb_layers(emb)
            emb = unsqueeze_as(emb, x)
            weight, bias = th.chunk(emb, 2, dim=1)  # (N, H, ...)
        else:
            weight = unsqueeze_as(self.weight, x)  # (1, H, ...)
            bias = unsqueeze_as(self.bias, x)
        x = x + th.addcmul(bias, weight, x * x_n)

        x = self.conv2(x)
        x = self.drop(x)
        return x


class ConvNextBlock(TimestepBlock):
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
        emb_channels,
        dropout,
        out_channels=None,
        num_groups=32,
        dims=2,
        up=False,
        down=False,
        kernel_size=7,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        hidden_dim = self.out_channels * 2

        self.in_layers = nn.Sequential(
            conv_nd(dims, channels, channels, kernel_size, padding=kernel_size // 2, groups=channels),
            normalization(channels, num_groups),
            # nn.SiLU(),
            conv_nd(dims, channels, hidden_dim, 1),
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
            self.emb_layers = nn.Sequential(nn.SiLU(), linear(emb_channels, 2 * hidden_dim))
        self.out_layers = nn.Sequential(
            normalization(hidden_dim, num_groups),
            nn.GELU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, hidden_dim, self.out_channels, 1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
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

            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)

        else:
            h = self.out_layers(h)

        return self.skip_connection(x) + h


class UNext2(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        dims=2,
        in_channels=3,
        model_channels=64,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=[2],
        dropout=0,
        channel_mult=[1, 2, 4, 8],
        kernel_sizes=[7, 5, 3, 1],
        num_groups=32,
        conv_resample=True,
        num_classes=None,
        use_checkpoint=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        legacy=True,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, "Fool!! You forgot to include the dimension of your cross-attention conditioning..."

        if context_dim is not None:
            assert (
                use_spatial_transformer
            ), "Fool!! You forgot to use the spatial transformer for your cross-attention conditioning..."
            from omegaconf.listconfig import ListConfig

            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, "Either num_heads or num_head_channels has to be set"

        if num_head_channels == -1:
            assert num_heads != -1, "Either num_heads or num_head_channels has to be set"

        self.model_channels = model_channels
        self.num_classes = num_classes

        # time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # class embedding
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # layers option
        res_kwargs = dict(
            emb_channels=time_embed_dim,
            dropout=dropout,
            num_groups=num_groups,
            dims=dims,
        )
        attn_kwargs = dict(
            use_checkpoint=use_checkpoint,
            use_new_attention_order=use_new_attention_order,
            num_groups=num_groups,
        )

        # layers start:
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))])
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ConvNextBlock(ch, out_channels=mult * model_channels, kernel_size=kernel_sizes[level], **res_kwargs)]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    # layers.append(
                    #     AttentionBlock(ch, num_heads=num_heads, num_head_channels=dim_head, **attn_kwargs)
                    #     if not use_spatial_transformer
                    #     else SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim)
                    # )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ConvNextBlock(ch, out_channels=out_ch, down=True, kernel_size=kernel_sizes[level], **res_kwargs)
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ConvNextBlock(ch, kernel_size=kernel_sizes[-1], **res_kwargs),
            # AttentionBlock(ch, num_heads=num_heads, num_head_channels=dim_head, **attn_kwargs)
            # if not use_spatial_transformer
            # else SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim),
            ConvNextBlock(ch, kernel_size=kernel_sizes[-1], **res_kwargs),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ConvNextBlock(ch + ich, out_channels=model_channels * mult, kernel_size=kernel_sizes[level], **res_kwargs)
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    # layers.append(
                    #     AttentionBlock(ch, num_heads=num_heads, num_head_channels=dim_head, **attn_kwargs)
                    #     if not use_spatial_transformer
                    #     else SpatialTransformer(
                    #         ch,
                    #         num_heads,
                    #         dim_head,
                    #         depth=transformer_depth,
                    #         context_dim=context_dim,
                    #         num_groups=num_groups,
                    #     )
                    # )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ConvNextBlock(ch, out_channels=out_ch, up=True, kernel_size=kernel_sizes[level], **res_kwargs)
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch, num_groups),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
            # conv_nd(dims, model_channels, out_channels, 3, padding=1),
        )

    def forward(self, x, t, context=None, y=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.num_classes is not None:
            assert y is not None, "must specify y if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(t, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)

        h = self.out(h)
        return h


if __name__ == "__main__":
    model = UNext2(dims=2, in_channels=3, out_channels=6, num_heads=8)
    x = th.rand(2, 3, 64, 64)
    t = th.randint(0, 1000, (2,))
    out = model(x, t)
    print(out.shape)
