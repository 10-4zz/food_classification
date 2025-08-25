from typing import Optional, Union

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .common_layer import (
    get_norm_layers,
    get_act_layers,
    auto_pad,
    GhostConv2d,
    ShuffleConv,
    PartialConv, DropPath,
)
from utils.math_utils import make_divisible, bound_fn


class GhostBottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ratio: Union[int, float] = 0.5,
        dw_kernel_size: int = 3,
        dilation: Optional[int] = 1,
        stride: Optional[int] = 1,
        use_se: Optional[bool] = False,
        act_fn_name: Optional[str] = "hard_swish",
        kernel_size: Optional[int] = 1,
        *args,
        **kwargs
    ) -> None:
        act_fn = nn.Hardswish()

        super().__init__()

        # Calculate the middle_channels dynamically
        middle_channels = int(round(in_channels * ratio))

        block = nn.Sequential()

        block.add_module(
            name="ghost1",
            module=GhostConv2d(
                in_channels=in_channels,
                out_channels=middle_channels,
                stride=1,
                kernel_size=kernel_size,
            ),
        )
        block.add_module(name="act_fn_2", module=act_fn)

        if stride > 1:
            conv_dw_bn =nn.Sequential(
                nn.Conv2d(
                    in_channels=middle_channels,
                    out_channels=middle_channels,
                    kernel_size=dw_kernel_size,
                    stride=stride,
                    groups=middle_channels,
                    bias=False,
                ),
                get_norm_layers(num_features=middle_channels, norm_name="batch_norm_2d"),
            )
            block.add_module(
                name="conv_dw_bn",
                module=conv_dw_bn
            )

        block.add_module(
            name="ghost2",
            module=nn.Conv2d(
                in_channels=middle_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
            ),
        )

        if (in_channels == out_channels) and (stride == 1):
            shortcut = nn.Identity()
        else:
            shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=dw_kernel_size,
                    stride=stride,  # stride
                    groups=in_channels,  # inchannels
                    bias=False,
                ),
                get_norm_layers(num_features=in_channels, norm_name="batch_norm_2d"),
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1, # 1
                    bias=False,
                ),
                get_norm_layers(num_features=out_channels, norm_name="batch_norm_2d"),
            )

        self.block = block
        self.shortcut = shortcut
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dw_kernel_size = dw_kernel_size
        self.dilation = dilation
        self.use_se = use_se
        self.stride = stride
        self.act_fn_name = act_fn_name
        self.kernel_size = kernel_size
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        y1 = self.block(x)
        y2 = self.shortcut(x)
        return y1 + y2

    def __repr__(self) -> str:
        return "{}(in_channels={}, out_channels={}, dw_kernel_size={}, stride={}, dilation={}, use_se={}, kernel_size={}, act_fn={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.dw_kernel_size,
            self.stride,
            self.dilation,
            self.use_se,
            self.kernel_size,
            self.act_fn_name,
        )


class PartialBlock(nn.Module):
    def __init__(
            self,
            in_channels: int
    ) -> None:
        super(PartialBlock, self).__init__()
        hidden_channels = int(in_channels // 3)
        remainder_channels = in_channels % 3
        self.partial_conv1 = PartialConv(
            in_channels=hidden_channels,
            divide_num=2,
        )
        self.partial_conv2 = PartialConv(
            in_channels=hidden_channels,
            divide_num=2,
        )
        self.partial_conv3 = PartialConv(
            in_channels=hidden_channels + remainder_channels,
            divide_num=2,
        )

        self.hidden_channels = hidden_channels
        self.remainder_channels = remainder_channels
    
    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x1, x2, x3 = torch.split(x, [self.hidden_channels, self.hidden_channels, self.hidden_channels + self.remainder_channels], dim=1)
        x1 = self.partial_conv1(x1)
        x2 = self.partial_conv2(x2)
        x3 = self.partial_conv3(x3)
        x = torch.cat((x1, x2, x3), 1)
        return x + residual


class InvertedResidual(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            expand_ratio: Union[int, float] = 2,
            dilation: int = 1,
            skip_connection: Optional[bool] = True,
            act_layer: str = "hard_swish",
    ) -> None:
        assert stride in [1, 2], "The stride should be 1 or 2 in the inverted residual block."
        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)
        super(InvertedResidual, self).__init__()

        block = []
        if expand_ratio != 1:
            block.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=hidden_dim,
                        kernel_size=1,
                        stride=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_dim),
                    get_act_layers(act_name=act_layer),
                )
            )

        block.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=3,
                    stride=stride,
                    dilation=dilation,
                    padding=auto_pad(3, None, dilation),
                    bias=False,
                    groups=hidden_dim,
                ),
                nn.BatchNorm2d(hidden_dim),
                get_act_layers(act_name=act_layer),
            )
        )

        block.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        )

        self.block = nn.Sequential(*block)
        self.use_res_connect = (
                stride == 1 and in_channels == out_channels and skip_connection
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


class ShuffleInvertedResidual(InvertedResidual):
    def __init__(
            self,
            h_groups: int,
            w_groups: int,
            in_channels: int,
            out_channels: int,
            stride: int,
            expand_ratio: Union[int, float],
            dilation: int = 1,
            skip_connection: Optional[bool] = True
    ) -> None:
        self.h_groups = h_groups
        self.w_groups = w_groups

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            expand_ratio=expand_ratio,
            dilation=dilation,
            skip_connection=skip_connection
        )

        tar_idx = 0 if expand_ratio == 1 else 1

        self.block[tar_idx] = ShuffleConv(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            h_groups=h_groups,
            w_groups=w_groups,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            groups=self.hidden_dim,
            use_act=True,
            use_norm=True
        )


class SingleHeadAttention(nn.Module):
    """
    This layer applies a single-head attention as described in `DeLighT <https://arxiv.org/abs/2008.00623>`_ paper

    Args:
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(N, P, C_{in})`
        attn_dropout (Optional[float]): Attention dropout. Default: 0.0
        bias (Optional[bool]): Use bias or not. Default: ``True``

    Shape:
        - Input: :math:`(N, P, C_{in})` where :math:`N` is batch size, :math:`P` is number of patches,
        and :math:`C_{in}` is input embedding dim
        - Output: same shape as the input

    """

    def __init__(
        self,
        embed_dim: int,
        attn_dropout: Optional[float] = 0.0,
        bias: Optional[bool] = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__()

        self.qkv_proj = nn.Linear(
            in_features=embed_dim, out_features=3 * embed_dim, bias=bias
        )

        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.out_proj = nn.Linear(
            in_features=embed_dim, out_features=embed_dim, bias=bias
        )

        self.softmax = nn.Softmax(dim=-1)
        self.embed_dim = embed_dim
        self.scaling = self.embed_dim**-0.5

    def __repr__(self) -> str:
        return "{}(embed_dim={}, attn_dropout={})".format(
            self.__class__.__name__, self.embed_dim, self.attn_dropout.p
        )

    def forward(
        self,
        x_q: Tensor,
        x_kv: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        *args,
        **kwargs
    ) -> Tensor:
        # [N, P, C] --> [N, P, 3C]
        if x_kv is None:
            qkv = self.qkv_proj(x_q)
            # [N, P, 3C] --> [N, P, C] x 3
            query, key, value = torch.chunk(qkv, chunks=3, dim=-1)
        else:
            query = F.linear(
                x_q,
                weight=self.qkv_proj.weight[: self.embed_dim, ...],
                bias=self.qkv_proj.bias[: self.embed_dim],
            )

            # [N, P, C] --> [N, P, 2C]
            kv = F.linear(
                x_kv,
                weight=self.qkv_proj.weight[self.embed_dim :, ...],
                bias=self.qkv_proj.bias[self.embed_dim :],
            )
            key, value = torch.chunk(kv, chunks=2, dim=-1)

        query = query * self.scaling

        # [N, P, C] --> [N, C, P]
        key = key.transpose(-2, -1)

        # QK^T
        # [N, P, C] x [N, C, P] --> [N, P, P]
        attn = torch.matmul(query, key)

        if attn_mask is not None:
            # attn_mask shape should be the same as attn
            assert list(attn_mask.shape) == list(
                attn.shape
            ), "Shape of attention mask and attn should be the same. Got: {} and {}".format(
                attn_mask.shape, attn.shape
            )
            attn = attn + attn_mask

        if key_padding_mask is not None:
            # Do not attend to padding positions
            # key padding mask size is [N, P]
            batch_size, num_src_tokens, num_tgt_tokens = attn.shape
            assert key_padding_mask.dim() == 2 and list(key_padding_mask.shape) == [
                batch_size,
                num_tgt_tokens,
            ], "Key_padding_mask should be 2-dimension with shape [{}, {}]. Got: {}".format(
                batch_size, num_tgt_tokens, key_padding_mask.shape
            )
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).to(torch.bool),
                float("-inf"),
            )

        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        # weighted sum
        # [N, P, P] x [N, P, C] --> [N, P, C]
        out = torch.matmul(attn, value)
        out = self.out_proj(out)

        return out


class MultiHeadAttention(nn.Module):
    """
    This layer applies a multi-head self- or cross-attention as described in
    `Attention is all you need <https://arxiv.org/abs/1706.03762>`_ paper

    Args:
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(N, S, C_{in})`
        num_heads (int): Number of heads in multi-head attention
        attn_dropout (Optional[float]): Attention dropout. Default: 0.0
        bias (Optional[bool]): Use bias or not. Default: ``True``

    Shape:
        - Input:
           - Query tensor (x_q) :math:`(N, S, C_{in})` where :math:`N` is batch size, :math:`S` is number of source tokens,
        and :math:`C_{in}` is input embedding dim
           - Optional Key-Value tensor (x_kv) :math:`(N, T, C_{in})` where :math:`T` is number of target tokens
        - Output: same shape as the input

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        bias: Optional[bool] = True,
        output_dim: Optional[int] = None,
        *args,
        **kwargs
    ) -> None:
        if output_dim is None:
            output_dim = embed_dim
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                "Embedding dim must be divisible by number of heads in {}. Got: embed_dim={} and num_heads={}".format(
                    self.__class__.__name__, embed_dim, num_heads
                )
            )

        self.qkv_proj = nn.Linear(
            in_features=embed_dim, out_features=3 * embed_dim, bias=bias
        )

        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.out_proj = nn.Linear(
            in_features=embed_dim, out_features=output_dim, bias=bias
        )

        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.use_separate_proj_weight = embed_dim != output_dim

    def __repr__(self):
        return "{}(head_dim={}, num_heads={}, attn_dropout={})".format(
            self.__class__.__name__, self.head_dim, self.num_heads, self.attn_dropout.p
        )

    def forward_default(
        self,
        x_q: Tensor,
        x_kv: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # [N, S, C]
        b_sz, S_len, in_channels = x_q.shape

        if x_kv is None:
            # self-attention
            # [N, S, C] --> [N, S, 3C] --> [N, S, 3, h, c] where C = hc
            qkv = self.qkv_proj(x_q).reshape(b_sz, S_len, 3, self.num_heads, -1)
            # [N, S, 3, h, c] --> [N, h, 3, S, C]
            qkv = qkv.transpose(1, 3).contiguous()

            # [N, h, 3, S, C] --> [N, h, S, C] x 3
            query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        else:
            T_len = x_kv.shape[1]

            # cross-attention
            # [N, S, C]
            query = F.linear(
                x_q,
                weight=self.qkv_proj.weight[: self.embed_dim, ...],
                bias=self.qkv_proj.bias[: self.embed_dim]
                if self.qkv_proj.bias is not None
                else None,
            )
            # [N, S, C] --> [N, S, h, c] --> [N, h, S, c]
            query = (
                query.reshape(b_sz, S_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
            )

            # [N, T, C] --> [N, T, 2C]
            kv = F.linear(
                x_kv,
                weight=self.qkv_proj.weight[self.embed_dim :, ...],
                bias=self.qkv_proj.bias[self.embed_dim :]
                if self.qkv_proj.bias is not None
                else None,
            )
            # [N, T, 2C] --> [N, T, 2, h, c]
            kv = kv.reshape(b_sz, T_len, 2, self.num_heads, self.head_dim)
            # [N, T, 2, h, c] --> [N, h, 2, T, c]
            kv = kv.transpose(1, 3).contiguous()
            key, value = kv[:, :, 0], kv[:, :, 1]

        query = query * self.scaling

        # [N h, T, c] --> [N, h, c, T]
        key = key.transpose(-1, -2)

        # QK^T
        # [N, h, S, c] x [N, h, c, T] --> [N, h, S, T]
        attn = torch.matmul(query, key)

        batch_size, num_heads, num_src_tokens, num_tgt_tokens = attn.shape
        if attn_mask is not None:
            # attn_mask shape should be the same as attn
            assert list(attn_mask.shape) == [
                batch_size,
                num_src_tokens,
                num_tgt_tokens,
            ], "Shape of attention mask should be [{}, {}, {}]. Got: {}".format(
                batch_size, num_src_tokens, num_tgt_tokens, attn_mask.shape
            )
            # [N, S, T] --> [N, 1, S, T]
            attn_mask = attn_mask.unsqueeze(1)
            attn = attn + attn_mask

        if key_padding_mask is not None:
            # Do not attend to padding positions
            # key padding mask size is [N, T]
            assert key_padding_mask.dim() == 2 and list(key_padding_mask.shape) == [
                batch_size,
                num_tgt_tokens,
            ], "Key_padding_mask should be 2-dimension with shape [{}, {}]. Got: {}".format(
                batch_size, num_tgt_tokens, key_padding_mask.shape
            )
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1)
                .unsqueeze(2)
                .to(torch.bool),  # [N, T] --> [N, 1, 1, T]
                float("-inf"),
            )

        attn_dtype = attn.dtype
        attn_as_float = self.softmax(attn.float())
        attn = attn_as_float.to(attn_dtype)
        attn = self.attn_dropout(attn)

        # weighted sum
        # [N, h, S, T] x [N, h, T, c] --> [N, h, S, c]
        out = torch.matmul(attn, value)

        # [N, h, S, c] --> [N, S, h, c] --> [N, S, C]
        out = out.transpose(1, 2).reshape(b_sz, S_len, -1)
        out = self.out_proj(out)

        return out

    def forward_pytorch(
        self,
        x_q: Tensor,
        x_kv: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        out, _ = F.multi_head_attention_forward(
            query=x_q,
            key=x_kv if x_kv is not None else x_q,
            value=x_kv if x_kv is not None else x_q,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=torch.empty([0]),
            in_proj_bias=self.qkv_proj.bias,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=self.attn_dropout.p,
            out_proj_weight=self.out_proj.weight,
            out_proj_bias=self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
            use_separate_proj_weight=True,
            q_proj_weight=self.qkv_proj.weight[: self.embed_dim, ...],
            k_proj_weight=self.qkv_proj.weight[
                self.embed_dim : 2 * self.embed_dim, ...
            ],
            v_proj_weight=self.qkv_proj.weight[2 * self.embed_dim :, ...],
        )
        return out

    def forward(
        self,
        x_q: Tensor,
        x_kv: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        *args,
        **kwargs
    ) -> Tensor:
        if kwargs.get("use_pytorch_mha", False):
            # pytorch uses sequence-first format. Make sure that input is of the form [Sequence, Batch, Hidden dim]
            return self.forward_pytorch(
                x_q=x_q,
                x_kv=x_kv,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        else:
            # our default implementation format follows batch-first format. Make sure the input is of the form
            # [Batch , Sequence, Hidden_dim]
            return self.forward_default(
                x_q=x_q,
                x_kv=x_kv,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )


class SeparableSelfAttention(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            attn_dropout: Optional[float] = 0.0,
            bias: Optional[bool] = True,
    ) -> None:
        super(SeparableSelfAttention, self).__init__()
        self.qkv_proj = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=1 + 2 * embed_dim,
            kernel_size=1,
            bias=bias
        )

        self.attn_dropout = nn.Dropout(attn_dropout)

        self.out_proj = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=1,
            bias=bias
        )

        self.embed_dim = embed_dim

    def _forward_self_attn(self, x: Tensor) -> Tensor:
        qkv = self.qkv_proj(x)

        query, key, value = torch.split(
            qkv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1
        )

        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_dropout(context_scores)

        context_vector = key * context_scores
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        out = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)

        return out

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_self_attn(x)


class FFNLinear(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            dropout: float = 0.,
            act_layer: str = "silu"
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=dim, out_features=hidden_dim),
            get_act_layers(act_name=act_layer),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_dim, out_features=dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class FFNConv(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            ffn_latent_dim: int,
            ffn_dropout: Optional[float] = 0.0,
            act_layer: str = "hard_swish",
            norm_layer: str = "layer_norm_2d",
    ) -> None:
        super(FFNConv, self).__init__()

        self.pre_norm_ffn = nn.Sequential(
            get_norm_layers(num_features=embed_dim, norm_name=norm_layer),
            nn.Conv2d(
                in_channels=embed_dim,
                out_channels=ffn_latent_dim,
                kernel_size=1,
                stride=1,
                bias=True
            ),
            get_act_layers(act_name=act_layer),
            nn.Dropout(ffn_dropout),
            nn.Conv2d(
                in_channels=ffn_latent_dim,
                out_channels=embed_dim,
                kernel_size=1,
                bias=True,
            ),
            nn.Dropout(ffn_dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Feed forward network
        x = x + self.pre_norm_ffn(x)
        return x


class TransformerEncoder(nn.Module):
    """
    This class defines the pre-norm `Transformer encoder <https://arxiv.org/abs/1706.03762>`_
    Args:
        embed_dim: :math:`C_{in}` from an expected input of size :math:`(N, P, C_{in})`.
        ffn_latent_dim: Inner dimension of the FFN.
        num_heads: Number of heads in multi-head attention. Default: 8.
        attn_dropout: Dropout rate for attention in multi-head attention. Default: 0.0
        dropout: Dropout rate. Default: 0.0.
        ffn_dropout: Dropout between FFN layers. Default: 0.0.
        transformer_norm_layer: Normalization layer. Default: layer_norm.
        drop_path: drop path dropout setting. Default: 0.0.

    Shape:
        - Input: :math:`(N, P, C_{in})` where :math:`N` is batch size, :math:`P` is number of patches,
        and :math:`C_{in}` is input embedding dim
        - Output: same shape as the input
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_latent_dim: int,
        num_heads: Optional[int] = 8,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.0,
        ffn_dropout: Optional[float] = 0.0,
        act_layer: Optional[str] = "silu",
        transformer_norm_layer: Optional[str] = "layer_norm",
        drop_path: Optional[float] = 0.0,
        *args,
        **kwargs,
    ) -> None:

        super().__init__()

        attn_unit = SingleHeadAttention(
            embed_dim=embed_dim, attn_dropout=attn_dropout, bias=True
        )
        if num_heads > 1:
            attn_unit = MultiHeadAttention(
                embed_dim,
                num_heads,
                attn_dropout=attn_dropout,
                bias=True,
            )

        self.pre_norm_mha = nn.Sequential(
            get_norm_layers(
                num_features=embed_dim, norm_name=transformer_norm_layer,
            ),
            attn_unit,
            nn.Dropout(p=dropout),
        )

        active_layer = get_act_layers(act_name=act_layer)
        self.pre_norm_ffn = nn.Sequential(
            get_norm_layers(
                num_features=embed_dim, norm_name=transformer_norm_layer,
            ),
            nn.Linear(in_features=embed_dim, out_features=ffn_latent_dim, bias=True),
            active_layer,
            nn.Dropout(p=ffn_dropout),
            nn.Linear(in_features=ffn_latent_dim, out_features=embed_dim, bias=True),
            nn.Dropout(p=dropout),
        )

        self.drop_path = nn.Identity()
        if drop_path > 0.0:
            if dropout > 0.0:
                raise ValueError(
                    "drop path dropout and dropout are mutually exclusive. "
                    "Use either of them, but not both."
                    "Got: {} and {}".format(drop_path, dropout)
                )
            self.drop_path = DropPath(drop_prob=drop_path)

        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim
        self.ffn_dropout = ffn_dropout
        self.stochastic_dropout = drop_path
        self.std_dropout = dropout
        self.attn_fn_name = attn_unit.__class__.__name__
        self.act_fn_name = active_layer.__class__.__name__
        self.norm_type = transformer_norm_layer

    def __repr__(self) -> str:
        return "{}(embed_dim={}, ffn_dim={}, dropout={}, ffn_dropout={}, stochastic_dropout={}, attn_fn={}, act_fn={}, norm_fn={})".format(
            self.__class__.__name__,
            self.embed_dim,
            self.ffn_dim,
            self.std_dropout,
            self.ffn_dropout,
            self.stochastic_dropout,
            self.attn_fn_name,
            self.act_fn_name,
            self.norm_type,
        )

    def forward(
        self,
        x: Tensor,
        x_prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        *args,
        **kwargs,
    ) -> Tensor:

        # Multi-head attention
        res = x
        x = self.pre_norm_mha[0](x)  # norm
        x = self.pre_norm_mha[1](
            x_q=x,
            x_kv=x_prev,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            *args,
            **kwargs,
        )  # mha

        x = self.drop_path(self.pre_norm_mha[2](x))  # applying stochastic depth
        x = x + res

        # Feed forward network
        x = x + self.drop_path(self.pre_norm_ffn(x))
        return x
