from timm.models.layers import DropPath, trunc_normal_

from dfusion.models.kitsunetic import convnextv2
from dfusion.models.kitsunetic.unet_module import *


class Downsample(nn.Sequential):
    def __init__(self, dim, dim_out, scale=2):
        super().__init__(
            convnextv2.LayerNorm(dim, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dim, dim_out, kernel_size=scale, stride=scale),
        )


class Upsample(nn.Module):
    def __init__(self, dim, dim_out, scale=2) -> None:
        super().__init__()
        self.scale = scale
        self.module = nn.Sequential(
            # convnextv2.LayerNorm(dim, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dim, dim_out, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode="nearest")
        x = self.module(x)
        return x


class UpsampleTransposed(nn.Sequential):  # deprecated?
    def __init__(self, dim, dim_out, scale=2):
        super().__init__(
            convnextv2.LayerNorm(dim, eps=1e-6, data_format="channels_first"),
            nn.ConvTranspose2d(dim, dim_out, kernel_size=scale, stride=scale),
        )


class TimeBlock(convnextv2.Block):
    def __init__(self, dim, drop_path=0, dim_emb=None, dim_out=None):
        super().__init__(dim, drop_path, dim_out)
        self.dim_emb = dim_emb

        if self.dim_emb is not None:
            self.emb_layer = nn.Sequential(
                # nn.SiLU(),
                nn.GELU(),
                nn.Linear(dim_emb, self.dwconv.out_channels * 2),
            )

    def forward(self, x, emb=None):
        input = self.shortcut(x)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)

        # insert embedding using AdaLayerNorm
        if self.dim_emb is not None:
            emb = self.emb_layer(emb)  # (N, 2C)
            emb = emb.view(emb.size(0), *(1 for _ in range(x.ndim - emb.ndim)), emb.size(-1))  # (N, ..., 2C)
            scale, shift = th.chunk(emb, 2, dim=-1)
            x = x * (1 + scale) + shift

        x = self.pwconv1(x)
        x = self.act(x)
        # x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class UNextV2(nn.Module):
    def __init__(
        self,
        in_chans=3,
        out_chans=None,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        attention_resolutions=[2],
        drop_path_rate=0.0,
        use_time_emb=False,
    ) -> None:
        out_chans = default(out_chans, in_chans)

        super().__init__()

        self.dims = dims
        self.depths = depths
        self.use_time_emb = use_time_emb

        self.time_emb_dim = None
        if self.use_time_emb:
            self.time_emb_dim_start = dims[0]
            self.time_emb_dim = self.time_emb_dim_start * 4
            self.time_embed = nn.Sequential(
                nn.Linear(dims[0], self.time_emb_dim),
                # nn.SiLU(),
                nn.GELU(),
                nn.Linear(self.time_emb_dim, self.time_emb_dim),
            )

        stem_in = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], 1),
            convnextv2.LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        stem_out = nn.Sequential(
            convnextv2.LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dims[0], out_chans, 1),
        )

        self.downsample_layers = nn.ModuleList([stem_in])
        for i in range(3):
            downsample_layer = Downsample(dims[i], dims[i + 1], 2)
            self.downsample_layers.append(downsample_layer)

        self.upsample_layers = nn.ModuleList([stem_out])
        for i in range(3):
            upsample_layer = Upsample(dims[i + 1], dims[i], 2)
            self.upsample_layers.append(upsample_layer)

        self.stages1 = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        self.stages2 = nn.ModuleList()
        dp_rates = [x.item() for x in th.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        ds = 1
        for i in range(4):
            stage1, stage2 = nn.ModuleList(), nn.ModuleList()
            for j in range(depths[i]):
                stage1.append(TimeBlock(dims[i], drop_path=dp_rates[cur + j], dim_emb=self.time_emb_dim))
                stage2.append(TimeBlock(dims[i] * 2, drop_path=dp_rates[cur + j], dim_emb=self.time_emb_dim, dim_out=dims[i]))

                # if ds in attention_resolutions:
                #     stage1.append(AttentionBlock(dims[i], num_heads=8))
                #     stage2.append(AttentionBlock(dims[i], num_heads=8))
            self.stages1.append(stage1)
            self.stages2.append(stage2)

            cur += depths[i]

        self.stages_mid = nn.Sequential(
            TimeBlock(dims[-1], drop_path=dp_rates[-1], dim_emb=self.time_emb_dim),
            AttentionBlock(dims[-1], num_heads=8),
            TimeBlock(dims[-1], dp_rates[-1], dim_emb=self.time_emb_dim),
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear, nn.ConvTranspose2d, nn.ConvTranspose1d)):
            trunc_normal_(m.weight, std=0.02)
            if hasattr(m, "bias"):
                nn.init.zeros_(m.bias)

    def forward(self, x, t=None):
        emb = None
        if self.use_time_emb:
            assert t is not None
            emb = timestep_embedding(t, self.time_emb_dim_start, repeat_only=False)
            emb = self.time_embed(emb)

        hs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            for layer in self.stages1[i]:
                if isinstance(layer, TimeBlock):
                    x = layer(x, emb)
                else:
                    x = layer(x)
                hs.append(x)

        for layer in self.stages_mid:
            if isinstance(layer, TimeBlock):
                x = layer(x, emb)
            else:
                x = layer(x)

        # print(x.shape)
        # for i in range(len(hs)):
        #     print(f"hs[{i:02d}]: {hs[i].shape}")

        for i in range(3, -1, -1):
            for layer in self.stages2[i]:
                h = hs.pop()
                # print(x.shape, h.shape)
                x = th.cat([x, h], dim=1)
                if isinstance(layer, TimeBlock):
                    x = layer(x, emb)
                else:
                    x = layer(x)
            x = self.upsample_layers[i](x)

        assert len(hs) == 0
        return x


# def unextv2_atto(**kwargs):
#     model = UNextV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
#     return model


def unextv2_atto(**kwargs):
    model = UNextV2(depths=[2, 2, 2, 2], dims=[128, 256, 256, 256], **kwargs)
    return model


# def unextv2_femto(**kwargs):
#     model = UNextV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
#     return model


def unextv2_femto(**kwargs):
    model = UNextV2(depths=[3, 3, 3, 3], dims=[128, 256, 256, 256], **kwargs)
    return model


# def unextv2_pico(**kwargs):
#     model = UNextV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
#     return model


def unextv2_pico(**kwargs):
    # model = UNextV2(depths=[2, 2, 4, 4], dims=[64, 128, 256, 512], **kwargs)
    model = UNextV2(depths=[4, 4, 4, 4], dims=[128, 256, 256, 256], **kwargs)
    return model


# def unextv2_nano(**kwargs):
#     model = UNextV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
#     return model


def unextv2_nano(**kwargs):
    model = UNextV2(depths=[3, 3, 4, 4], dims=[80, 160, 320, 640], **kwargs)
    return model


def unextv2_tiny(**kwargs):
    model = UNextV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model


def unextv2_base(**kwargs):
    model = UNextV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model


def unextv2_large(**kwargs):
    model = UNextV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model


def unextv2_huge(**kwargs):
    model = UNextV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model


def __test__():
    model = unextv2_atto(use_time_emb=True)
    print(model)
    x = th.rand(2, 3, 64, 64)
    t = th.randint(0, 1000, (2,))
    out = model(x, t)
    print(out.shape)


if __name__ == "__main__":
    __test__()
