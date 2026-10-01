"""
统一接口模型：
1. TCN；
2. 官方 Transformer；
3. 官方 Informer；
4. 官方 Autoformer。

统一输入/输出接口与原 linear_main.py 一致：
    输入  x: [T, B, C_in]
    输出  y: [T, B, C_out]

Informer/Autoformer/Transformer 的官方模型本身使用：
    [B, T, C]
以及 x_mark、decoder input。
本文件只做形状和 decoder 输入适配，不改变官方核心模型实现。
"""

from types import SimpleNamespace

import torch
import torch.nn as nn

from models import Transformer, Informer, Autoformer


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.network = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.network(x) + self.residual(x))


class TCNForecastModel(nn.Module):
    """TCN，输入输出均为 [T, B, C]。"""

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=90,
        num_levels=3,
        kernel_size=3,
        dropout=0.05
    ):
        super().__init__()

        blocks = []
        in_channels = input_dim
        for level in range(num_levels):
            dilation = 2 ** level
            blocks.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )
            in_channels = hidden_dim

        self.tcn = nn.Sequential(*blocks)
        self.output_layer = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(f"TCN input must be [T, B, C], got {tuple(x.shape)}")

        x = x.permute(1, 2, 0)       # [B, C, T]
        x = self.tcn(x)
        x = self.output_layer(x)
        return x.permute(2, 0, 1)    # [T, B, C_out]


def build_official_config(
    input_dim,
    output_dim,
    d_model=90,
    n_heads=5,
    e_layers=1,
    d_layers=1,
    d_ff=360,
    factor=5,
    dropout=0.05,
    moving_avg=25
):
    if d_model % n_heads != 0:
        raise ValueError(
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})."
        )

    # Wrapper先将输入投影到output_dim，因此官方模型的enc/dec/c_out统一为output_dim。
    return SimpleNamespace(
        seq_len=1,              # forward时允许可变T；官方层本身不固定长度
        label_len=1,            # forward时由wrapper动态确定
        pred_len=1,             # forward时由wrapper动态修改
        enc_in=output_dim,
        dec_in=output_dim,
        c_out=output_dim,
        d_model=d_model,
        n_heads=n_heads,
        e_layers=e_layers,
        d_layers=d_layers,
        d_ff=d_ff,
        factor=factor,
        moving_avg=moving_avg,
        dropout=dropout,
        attn="prob",
        embed="timeF",
        freq="h",
        activation="gelu",
        output_attention=False,
        distil=True,
        mix=True,
        bucket_size=4,
        n_hashes=4
    )


class OfficialForecastWrapper(nn.Module):
    """
    将官方Encoder-Decoder时间序列模型适配到原工程的 net(x) 接口。

    说明：
    - 原工程每次把完整序列作为一个batch，形状为[T, 1, C]；
    - 为保持输出与目标完全同形，本适配器令pred_len等于当前输入序列长度T；
    - x_mark使用全零timeF特征。ETTh1/Lorenz/NARMA均可统一运行；
    - 不修改官方Attention、AutoCorrelation、Encoder、Decoder内部逻辑。
    """

    def __init__(
        self,
        model_name,
        input_dim,
        output_dim,
        d_model=90,
        n_heads=5,
        e_layers=1,
        d_layers=1,
        d_ff=360,
        factor=5,
        dropout=0.05,
        moving_avg=25,
        label_ratio=0.5
    ):
        super().__init__()

        self.model_name = model_name.lower()
        self.output_dim = int(output_dim)
        self.label_ratio = float(label_ratio)

        self.input_projection = (
            nn.Identity()
            if input_dim == output_dim
            else nn.Linear(input_dim, output_dim)
        )

        self.configs = build_official_config(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=d_layers,
            d_ff=d_ff,
            factor=factor,
            dropout=dropout,
            moving_avg=moving_avg
        )

        if self.model_name == "transformer":
            self.model = Transformer.Model(self.configs)
        elif self.model_name == "informer":
            self.model = Informer.Model(self.configs)
        elif self.model_name == "autoformer":
            self.model = Autoformer.Model(self.configs)
        else:
            raise ValueError(f"Unknown official model: {model_name}")

    @staticmethod
    def _time_mark(batch_size, seq_len, device, dtype):
        # embed='timeF'且freq='h'时，官方TimeFeatureEmbedding要求最后一维为4。
        return torch.zeros(
            batch_size,
            seq_len,
            4,
            device=device,
            dtype=dtype
        )

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(
                f"{self.model_name} input must be [T, B, C], got {tuple(x.shape)}"
            )

        time_steps, batch_size, _ = x.shape
        if time_steps < 2:
            raise ValueError(
                f"{self.model_name} requires at least 2 time steps, got {time_steps}."
            )

        # [T,B,C] -> [B,T,C]，并在必要时把输入通道投影到输出通道。
        x_enc = self.input_projection(x.permute(1, 0, 2))

        pred_len = time_steps
        label_len = max(1, min(time_steps, int(round(time_steps * self.label_ratio))))

        # 官方模型内部会根据self.pred_len截取输出；Autoformer还用它构造趋势项。
        self.model.pred_len = pred_len
        if hasattr(self.model, "label_len"):
            self.model.label_len = label_len
        if hasattr(self.model, "seq_len"):
            self.model.seq_len = time_steps

        x_mark_enc = self._time_mark(
            batch_size, time_steps, x.device, x.dtype
        )

        # decoder已知部分取编码器末尾label_len，未来部分用0。
        known = x_enc[:, -label_len:, :]
        future_zeros = torch.zeros(
            batch_size,
            pred_len,
            self.output_dim,
            device=x.device,
            dtype=x.dtype
        )
        x_dec = torch.cat([known, future_zeros], dim=1)

        x_mark_dec = self._time_mark(
            batch_size,
            label_len + pred_len,
            x.device,
            x.dtype
        )

        output = self.model(
            x_enc,
            x_mark_enc,
            x_dec,
            x_mark_dec
        )

        if isinstance(output, tuple):
            output = output[0]

        # [B,T,C] -> [T,B,C]
        return output.permute(1, 0, 2).contiguous()


def create_advanced_model(
    model_name,
    input_dim,
    output_dim,
    hidden_dim=90
):
    name = model_name.lower()

    if name == "tcn":
        return TCNForecastModel(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim
        )

    if name in {"transformer", "informer", "autoformer"}:
        return OfficialForecastWrapper(
            model_name=name,
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=hidden_dim,
            n_heads=5,
            e_layers=1,
            d_layers=1,
            d_ff=4 * hidden_dim,
            factor=5,
            dropout=0.05,
            moving_avg=25
        )

    raise ValueError(f"Unknown model_name: {model_name}")
