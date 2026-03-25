from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

import torch

import models.registry as registry_module
from models.registry import get_model_handlers


def _cfg(algo: str) -> SimpleNamespace:
    return SimpleNamespace(MODEL_NAME=algo, TARGET_KIND="regression", MODEL_SEED=0, MODEL_PARAMS={})


def _data_shapes(input_dim: int = 8) -> dict:
    return {"input_dim": input_dim, "n_groups": 1}


@pytest.mark.parametrize("algo", ["bilstm_rul", "tcn_rul", "transformer_rul"])
def test_get_model_handlers_returns_handler(algo: str) -> None:
    h = get_model_handlers(algo)
    assert callable(getattr(h, "build", None))
    assert callable(getattr(h, "load_from_snapshot", None))


@pytest.mark.parametrize("algo", ["bilstm_rul", "tcn_rul", "transformer_rul"])
def test_build_output_shape(algo: str) -> None:
    h = get_model_handlers(algo)
    model = h.build(_cfg(algo), torch.device("cpu"), data_shapes=_data_shapes(8), artifacts=None)
    x = torch.randn(4, 30, 8)
    out = model(x)
    assert out.shape == (4, 1), f"{algo}: expected (4,1) got {out.shape}"


@pytest.mark.parametrize("algo", ["bilstm_rul", "tcn_rul", "transformer_rul"])
def test_build_with_model_params(algo: str) -> None:
    params = {
        "bilstm_rul": {"hidden_dim": 32, "num_layers": 1, "dropout": 0.0},
        "tcn_rul": {"hidden_dim": 32, "num_layers": 2, "kernel_size": 3, "dropout": 0.0},
        "transformer_rul": {
            "model_dim": 32, "num_heads": 2, "num_layers": 1, "ff_dim": 64, "dropout": 0.0,
        },
    }[algo]
    h = get_model_handlers(algo)
    model = h.build(
        _cfg(algo), torch.device("cpu"),
        data_shapes=_data_shapes(4), artifacts=None, model_params=params,
    )
    x = torch.randn(2, 10, 4)
    assert model(x).shape == (2, 1)


@pytest.mark.parametrize("algo", ["bilstm_rul", "tcn_rul", "transformer_rul"])
def test_load_from_snapshot_output_shape(algo: str) -> None:
    h = get_model_handlers(algo)
    snapshot = {"input_dim": 8, "context_len": 30}
    model = h.load_from_snapshot(_cfg(algo), torch.device("cpu"), snapshot)
    x = torch.randn(2, 30, 8)
    assert model(x).shape == (2, 1)


@pytest.mark.parametrize("algo", ["bilstm_rul", "tcn_rul", "transformer_rul"])
def test_state_dict_round_trip(algo: str) -> None:
    h = get_model_handlers(algo)
    model_a = h.build(_cfg(algo), torch.device("cpu"), data_shapes=_data_shapes(6), artifacts=None)
    sd = model_a.state_dict()
    model_b = h.load_from_snapshot(_cfg(algo), torch.device("cpu"), {"input_dim": 6})
    model_b.load_state_dict(sd, strict=False)
    x = torch.randn(1, 20, 6)
    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        assert torch.allclose(model_a(x), model_b(x))


def test_afno_raises_module_not_found_with_correct_name() -> None:
    with pytest.raises(ModuleNotFoundError) as exc_info:
        get_model_handlers("afnocg3_v1")
    assert exc_info.value.name == "src.models.registry"


def test_afno_variants_all_raise() -> None:
    for algo in ["afnocg2", "cifnocg2", "afnocg3", "afnocg3_v1"]:
        with pytest.raises(ModuleNotFoundError):
            get_model_handlers(algo)


def test_unknown_algo_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown algo_key"):
        get_model_handlers("nonexistent_model")


def test_model_handler_merged_params_ignore_none_and_keep_defaults() -> None:
    handler = get_model_handlers("transformer_rul")

    merged = handler._merged_params({"model_dim": 32, "dropout": None})

    assert merged["model_dim"] == 32
    assert merged["dropout"] == 0.1
    assert merged["num_heads"] == 4
    assert handler.extras is None


@pytest.mark.parametrize("algo", ["bilstm_rul", "tcn_rul", "transformer_rul"])
def test_build_and_load_use_default_input_dim_when_missing(algo: str) -> None:
    handler = get_model_handlers(algo)

    built = handler.build(_cfg(algo), torch.device("cpu"), data_shapes={}, artifacts=None)
    loaded = handler.load_from_snapshot(_cfg(algo), torch.device("cpu"), snapshot={})

    sample = torch.randn(2, 5, 1)
    assert built(sample).shape == (2, 1)
    assert loaded(sample).shape == (2, 1)


def test_bilstm_dropout_is_disabled_for_single_layer() -> None:
    handler = get_model_handlers("bilstm_rul")
    model = handler.build(
        _cfg("bilstm_rul"),
        torch.device("cpu"),
        data_shapes=_data_shapes(3),
        artifacts=None,
        model_params={"hidden_dim": 16, "num_layers": 1, "dropout": 0.9},
    )

    assert model.lstm.dropout == 0.0


def test_tcn_block_and_transformer_helpers_cover_internal_shapes() -> None:
    block = registry_module._TCNBlock(channels=4, kernel_size=3, dilation=2, dropout=0.0)
    tcn_input = torch.randn(2, 4, 9)
    tcn_output = block(tcn_input)

    pe = registry_module._make_sinusoidal_pe(max_len=7, d_model=6)
    transformer = registry_module._TransformerRUL(
        input_dim=3,
        model_dim=8,
        num_heads=2,
        num_layers=1,
        ff_dim=16,
        dropout=0.0,
        max_len=12,
    )
    transformer_output = transformer(torch.randn(2, 7, 3))

    assert tcn_output.shape == tcn_input.shape
    assert pe.shape == (7, 6)
    assert torch.allclose(pe[0, 0::2], torch.zeros(3))
    assert transformer_output.shape == (2, 1)
