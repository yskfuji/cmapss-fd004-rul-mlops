"""Public model registry for RULFM torch forecasters.

Provides BiLSTM, TCN, and Transformer RUL models as drop-in replacements for
the proprietary temporal model registry.  Each handler exposes the same
``build`` / ``load_from_snapshot`` / ``extras`` contract expected by
``torch_forecasters.train_univariate_torch_forecaster``.

Input contract (enforced by torch_forecasters.py):
    x : (batch, context_len, input_dim)  float32
Output contract:
    y : (batch, 1)  float32  — raw RUL in training-target space (log1p or none)
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

# ── BiLSTM ────────────────────────────────────────────────────────────────────

class _BiLSTMRUL(nn.Module):
    """Bidirectional LSTM for RUL regression.

    Args:
        input_dim:  Number of input features per time-step.
        hidden_dim: Hidden units per direction (default 64).
        num_layers: Stacked LSTM layers (default 2).
        dropout:    Dropout between layers (default 0.2, disabled for num_layers=1).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, input_dim)
        out, _ = self.lstm(x)
        # Use final time-step output from both directions
        return self.head(out[:, -1, :])  # (batch, 1)


# ── TCN ───────────────────────────────────────────────────────────────────────

class _TCNBlock(nn.Module):
    """Single dilated causal convolution residual block."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=padding)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=padding)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, seq) — slice to original seq length to remove causal padding
        out = self.relu(self.conv1(x)[..., :x.size(-1)])
        out = self.drop(out)
        out = self.relu(self.conv2(out)[..., :x.size(-1)])
        out = self.drop(out)
        return self.relu(out + x)


class _TCNRUL(nn.Module):
    """Temporal Convolutional Network for RUL regression.

    Args:
        input_dim:   Number of input features per time-step.
        hidden_dim:  Internal channel width (default 64).
        num_layers:  Number of dilated residual blocks (default 4).
        kernel_size: Convolution kernel size (default 3).
        dropout:     Dropout within each block (default 0.2).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList([
            _TCNBlock(hidden_dim, kernel_size, dilation=2 ** i, dropout=dropout)
            for i in range(num_layers)
        ])
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, input_dim) → transpose to (batch, input_dim, seq)
        h = self.input_proj(x.transpose(1, 2))
        for block in self.blocks:
            h = block(h)
        # Global average pool over time
        return self.head(h.mean(dim=-1))  # (batch, 1)


# ── Transformer ───────────────────────────────────────────────────────────────

class _TransformerRUL(nn.Module):
    """Encoder-only Transformer for RUL regression.

    Args:
        input_dim:  Number of input features per time-step.
        model_dim:  Internal embedding dimension (default 64).
        num_heads:  Attention heads (default 4).
        num_layers: Transformer encoder layers (default 2).
        ff_dim:     Feed-forward hidden dimension (default 256).
        dropout:    Dropout within encoder (default 0.1).
        max_len:    Maximum sequence length for positional encoding (default 512).
    """

    def __init__(
        self,
        input_dim: int,
        model_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.register_buffer(
            "pos_enc",
            _make_sinusoidal_pe(max_len, model_dim),
            persistent=False,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(model_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, input_dim)
        seq_len = x.size(1)
        h = self.input_proj(x) + self.pos_enc[:seq_len]
        h = self.encoder(h)
        return self.head(h[:, -1, :])  # (batch, 1)


def _make_sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
    return pe  # (max_len, d_model)


# ── Handler objects ────────────────────────────────────────────────────────────

class _ModelHandlers:
    """Minimal handler that satisfies the registry contract in torch_forecasters.py."""

    def __init__(self, model_cls: type, default_params: dict[str, Any]) -> None:
        self._cls = model_cls
        self._defaults = default_params
        self.extras: dict[str, Any] | None = None

    def _merged_params(self, model_params: dict[str, Any] | None) -> dict[str, Any]:
        params = dict(self._defaults)
        if isinstance(model_params, dict):
            params.update({k: v for k, v in model_params.items() if v is not None})
        return params

    def build(
        self,
        cfg: SimpleNamespace,
        device: torch.device,
        *,
        data_shapes: dict[str, Any],
        artifacts: Any,
        model_params: dict[str, Any] | None = None,
    ) -> nn.Module:
        input_dim = int(data_shapes.get("input_dim") or 1)
        params = self._merged_params(model_params)
        return self._cls(input_dim=input_dim, **params).to(device)

    def load_from_snapshot(
        self,
        cfg: SimpleNamespace,
        device: torch.device,
        snapshot: dict[str, Any],
        model_params: dict[str, Any] | None = None,
    ) -> nn.Module:
        input_dim = int(snapshot.get("input_dim") or 1)
        params = self._merged_params(model_params)
        return self._cls(input_dim=input_dim, **params).to(device)


_HANDLERS: dict[str, _ModelHandlers] = {
    "bilstm_rul": _ModelHandlers(
        _BiLSTMRUL,
        {"hidden_dim": 64, "num_layers": 2, "dropout": 0.2},
    ),
    "tcn_rul": _ModelHandlers(
        _TCNRUL,
        {"hidden_dim": 64, "num_layers": 4, "kernel_size": 3, "dropout": 0.2},
    ),
    "transformer_rul": _ModelHandlers(
        _TransformerRUL,
        {"model_dim": 64, "num_heads": 4, "num_layers": 2, "ff_dim": 256, "dropout": 0.1},
    ),
}


def get_model_handlers(algo_key: str) -> _ModelHandlers:
    """Return the handler for *algo_key*.

    Raises:
        ModuleNotFoundError: if *algo_key* belongs to the proprietary registry
            (afnocg2 / cifnocg2 / afnocg3 / afnocg3_v1) which is not included
            in this public release.
        ValueError: if *algo_key* is not recognised at all.
    """
    if algo_key in _HANDLERS:
        return _HANDLERS[algo_key]
    if algo_key in {"afnocg2", "cifnocg2", "afnocg3", "afnocg3_v1"}:
        raise ModuleNotFoundError(
            f"Proprietary model '{algo_key}' is not available in the public registry.",
            name="src.models.registry",
        )
    raise ValueError(f"Unknown algo_key: '{algo_key}'")
