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
from functools import partial

from dfusion.models.kitsunetic.unet_module import *


class UNet(nn.Module):
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
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
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
        num_groups=32,
        conv_resample=True,
        num_classes=None,
        use_checkpoint=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        attention_type="qkv_legacy",  # qkv, qkv_legacy, xformers
        no_attn=False,
        no_time_emb=False,
        is_auto_encoder=False,
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
        self.no_time_emb = no_time_emb
        self.is_auto_encoder = is_auto_encoder

        # time embedding
        if not no_time_emb or self.num_classes is not None:
            time_embed_dim = model_channels * 4
        else:
            time_embed_dim = None

        if not no_time_emb:
            self.time_embed = nn.Sequential(
                linear(model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )

        # class embedding
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # layers option
        res_fn = partial(
            ResBlock,
            emb_channels=time_embed_dim,
            dropout=dropout,
            num_groups=num_groups,
            dims=dims,
            use_checkpoint=use_checkpoint,
            use_scale_shift_norm=use_scale_shift_norm,
        )
        if not use_spatial_transformer:
            attn_fn = lambda ch, num_heads, dim_head: AttentionBlock(
                channels=ch,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_checkpoint=use_checkpoint,
                attention_type=attention_type,
                num_groups=num_groups,
            )
        else:
            attn_fn = lambda ch, num_heads, dim_head: SpatialTransformer(
                in_channels=ch,
                n_heads=num_heads,
                d_head=dim_head,
                depth=transformer_depth,
                context_dim=context_dim,
                attention_type=attention_type,
            )
        if no_attn:
            attn_fn = lambda *args, **kwargs: nn.Identity()

        # layers start
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))])
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [res_fn(ch, out_channels=mult * model_channels)]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if attention_type == "qkv_legacy":
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(attn_fn(ch, num_heads, dim_head))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        res_fn(ch, out_channels=out_ch, down=True)
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
        if attention_type == "qkv_legacy":
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            res_fn(ch),
            attn_fn(ch, num_heads, dim_head),
            res_fn(ch),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = ch + input_block_chans.pop() if not is_auto_encoder else ch
                layers = [res_fn(ich, out_channels=model_channels * mult)]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if attention_type == "qkv_legacy":
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(attn_fn(ch, num_heads, dim_head))
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        res_fn(ch, out_channels=out_ch, up=True)
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        if not is_auto_encoder:
            assert len(input_block_chans) == 0

        self.out = nn.Sequential(
            normalization(ch, num_groups),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
            # conv_nd(dims, model_channels, out_channels, 3, padding=1),
        )

    def get_embedding(self, t=None, y=None):
        emb = 0

        if not self.no_time_emb:
            assert t is not None, "must specify `t` if the model has time embedding"
            t_emb = timestep_embedding(t, self.model_channels, repeat_only=False)
            emb += self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y is not None, "must specify `y` if the model is class-conditional"
            assert y.shape == (x.shape[0],)
            emb += self.label_emb(y)

        return emb

    def encode(self, x, emb=None, context=None):
        if not self.no_time_emb or self.num_classes is not None:
            assert emb is not None
        
        hs = []
        h = x

        for module in self.input_blocks:
            h = module(h, emb, context)
            if not self.is_auto_encoder:
                hs.append(h)

        return h, hs

    def decode(self, h, hs=None, emb=None, context=None):
        if not self.is_auto_encoder:
            assert hs is not None
        if not self.no_time_emb or self.num_classes is not None:
            assert emb is not None
            
        h = self.middle_block(h, emb, context)

        for module in self.output_blocks:
            if not self.is_auto_encoder:
                h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)

        h = self.out(h)
        return h

    def forward(self, x, t=None, context=None, y=None):
        emb = self.get_embedding(t=t, y=y)
        h, hs = self.encode(x, emb, context=context)
        h = self.decode(h, hs, emb, context=context)
        return h


if __name__ == "__main__":
    model = UNet(dims=1, in_channels=1, out_channels=1, num_head_channels=32, attention_type="xformers")
    x = th.rand(2, 1, 1024)
    t = th.randint(0, 1000, (2,))
    out = model(x, t)
    print(out.shape)
