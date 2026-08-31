"""Transolver model for point-cloud flow prediction, vendored from Transolver.

Source: https://github.com/thuml/Transolver (MIT License, Copyright (c) 2024
THUML @ Tsinghua University), commit 75e0f67643806a81cd1d3f6adc88dd8c02416fe7,
file ``Car-Design-ShapeNetCar/models/Transolver.py``.

Local changes vs. upstream:
- ``timm.models.layers.trunc_normal_`` replaced by ``torch.nn.init.trunc_normal_``.
- ``forward`` takes a plain feature tensor ``[B, N, C]`` instead of a
  torch_geometric ``Data`` tuple and returns ``[B, N, out_dim]``.
- The ShapeNetCar-specific ``unified_pos``/``get_grid`` branch (hardcoded
  domain bounds and ``.cuda()`` calls) is removed; the constructor takes a
  single ``in_dim`` instead of ``fun_dim + space_dim``.
"""

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from .physics_attention import PhysicsAttentionIrregularMesh

ACTIVATION = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "softplus": nn.Softplus,
    "ELU": nn.ELU,
    "silu": nn.SiLU,
}


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act="gelu", res=True):
        super().__init__()
        if act in ACTIVATION:
            act = ACTIVATION[act]
        else:
            raise NotImplementedError(f"activation {act!r}")
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList(
            [nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)]
        )

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class TransolverBlock(nn.Module):
    """Transformer encoder block with Physics-Attention."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act="gelu",
        mlp_ratio=4,
        last_layer=False,
        out_dim=1,
        slice_num=32,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = PhysicsAttentionIrregularMesh(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


class Transolver(nn.Module):
    """Point-cloud Transolver: ``[B, N, in_dim] -> [B, N, out_dim]``."""

    def __init__(
        self,
        in_dim=7,
        out_dim=4,
        n_layers=5,
        n_hidden=256,
        dropout=0.0,
        n_head=8,
        act="gelu",
        mlp_ratio=1,
        slice_num=32,
    ):
        super().__init__()
        self.preprocess = MLP(in_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        self.n_hidden = n_hidden
        self.blocks = nn.ModuleList(
            [
                TransolverBlock(
                    num_heads=n_head,
                    hidden_dim=n_hidden,
                    dropout=dropout,
                    act=act,
                    mlp_ratio=mlp_ratio,
                    out_dim=out_dim,
                    slice_num=slice_num,
                    last_layer=(i == n_layers - 1),
                )
                for i in range(n_layers)
            ]
        )
        self.initialize_weights()
        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden, dtype=torch.float))

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected [B, N, C], got shape {tuple(x.shape)}")
        fx = self.preprocess(x)
        fx = fx + self.placeholder[None, None, :]
        for block in self.blocks:
            fx = block(fx)
        return fx
