"""from latent diffusion"""

import math
from inspect import isfunction

import numpy as np
import torch
import torch as th
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn


def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU()) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial**2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, k=None, v=None, mask=None):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        if k is None or v is None:
            bs, width, length = qkv.shape
            assert width % (3 * self.n_heads) == 0
            ch = width // (3 * self.n_heads)
            q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        else:
            q, k, v = qkv, k, v
            bs, length, ch = q.size(0), q.size(2), q.size(1) // self.n_heads

        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum("bct,bcs->bts", q * scale, k * scale)  # More stable with f16 than dividing afterwards
        # weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)

        if mask is not None:
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(weight.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=self.n_heads)
            weight.masked_fill_(~mask, max_neg_value)

        weight = th.softmax(weight, dim=-1)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, k=None, v=None, mask=None):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        if k is None or v is None:
            bs, width, length = qkv.shape
            assert width % (3 * self.n_heads) == 0
            ch = width // (3 * self.n_heads)
            q, k, v = qkv.chunk(3, dim=1)
        else:
            q, k, v = qkv, k, v
            bs, length = q.size(0), q.size(2)

        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        # weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)

        if mask is not None:
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(weight.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=self.n_heads)
            weight.masked_fill_(~mask, max_neg_value)

        weight = th.softmax(weight, dim=-1)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


# class LinearAttention(nn.Module):
#     def __init__(self, dim, heads=4, dim_head=32):
#         super().__init__()
#         self.heads = heads
#         hidden_dim = dim_head * heads
#         self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
#         self.to_out = nn.Conv2d(hidden_dim, dim, 1)

#     def forward(self, x):
#         b, c, h, w = x.shape
#         qkv = self.to_qkv(x)
#         q, k, v = rearrange(qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3)
#         k = k.softmax(dim=-1)
#         context = torch.einsum("bhdn,bhen->bhde", k, v)
#         out = torch.einsum("bhde,bhdn->bhen", context, q)
#         out = rearrange(out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w)
#         return self.to_out(out)


# class SpatialSelfAttention(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.in_channels = in_channels

#         self.norm = Normalize(in_channels)
#         self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
#         self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
#         self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
#         self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

#     def forward(self, x):
#         h_ = x
#         h_ = self.norm(h_)
#         q = self.q(h_)
#         k = self.k(h_)
#         v = self.v(h_)

#         # compute attention
#         b, c, h, w = q.shape
#         q = rearrange(q, "b c h w -> b (h w) c")
#         k = rearrange(k, "b c h w -> b c (h w)")
#         w_ = torch.einsum("bij,bjk->bik", q, k)

#         w_ = w_ * (int(c) ** (-0.5))
#         w_ = torch.nn.functional.softmax(w_, dim=2)

#         # attend to values
#         v = rearrange(v, "b c h w -> b c (h w)")
#         w_ = rearrange(w_, "b i j -> b j i")
#         h_ = torch.einsum("bij,bjk->bik", v, w_)
#         h_ = rearrange(h_, "b c (h w) -> b c h w", h=h)
#         h_ = self.proj_out(h_)

#         return x + h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, attention_type="qkv_legacy"):
        super().__init__()
        self.attention_type = attention_type

        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Conv1d(query_dim, inner_dim, 1, bias=False)
        self.to_k = nn.Conv1d(context_dim, inner_dim, 1, bias=False)
        self.to_v = nn.Conv1d(context_dim, inner_dim, 1, bias=False)

        if attention_type == "qkv_legacy":
            self.attention = QKVAttentionLegacy(heads)
        elif attention_type == "qkv":
            self.attention = QKVAttention(heads)
        elif attention_type == "xformers":
            from xformers.components.attention import ScaledDotProduct

            self.attention = ScaledDotProduct()
        else:
            raise NotImplementedError(attention_type)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x, context=None, mask=None):
        x = x.transpose(1, 2).contiguous()  # (B, C, L)

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        if self.attention_type in ("qkv", "qkv_legacy"):
            out = self.attention(q, k, v)
            out = out.transpose(1, 2).contiguous()  # (B, L, C)
        else:
            # q, k, v = map(lambda t: rearrange(t, "b (h c) l -> b l h c", h=self.heads).contiguous(), (q, k, v))
            q, k, v = map(lambda t: rearrange(t, "b (h c) l -> (b h) l c", h=self.heads).contiguous(), (q, k, v))
            out = self.attention(q, k, v)
            # out = rearrange(out, "b l h c -> b l (h c)").contiguous()  # (B, L, C)
            out = rearrange(out, "(b h) l c -> b l (h c)", h=self.heads).contiguous()  # (B, L, C)

        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        attention_type="qkv_legacy",
    ):
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            attention_type=attention_type,
        )  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            attention_type=attention_type,
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        num_groups=32,
        attention_type="qkv_legacy",
    ):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels, num_groups)

        self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim,
                    attention_type=attention_type,
                )
                for d in range(depth)
            ]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))

    def forward(self, x, context=None):
        xshape = x.shape
        x_in = x

        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, "b c ... -> b (...) c")  # (B, L, C)

        for block in self.transformer_blocks:
            x = block(x, context=context)

        x = x.transpose(1, 2).view(xshape).contiguous()
        x = self.proj_out(x)
        return x + x_in
