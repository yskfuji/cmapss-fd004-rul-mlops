from __future__ import annotations

import math

import pytest

pytest.importorskip("torch")
pytestmark = pytest.mark.experimental

import torch
from forecasting_api import torch_forecasters


def _loss(  # noqa: E501
    over_penalty: float = 2.0,
    under_penalty: float = 1.0,
    max_rul: float = 125.0,
) -> torch_forecasters._AsymmetricRULLoss:
    return torch_forecasters._AsymmetricRULLoss(
        over_penalty=over_penalty, under_penalty=under_penalty, max_rul=max_rul
    )


def test_normalize_protocol_accepts_asymmetric_rul() -> None:
    protocol = torch_forecasters._normalize_training_protocol({"loss": "asymmetric_rul"})
    assert protocol["loss"] == "asymmetric_rul"
    assert protocol["asym_over_penalty"] == 2.0
    assert protocol["asym_under_penalty"] == 1.0
    assert protocol["asym_max_rul"] == 125.0


def test_normalize_protocol_rejects_unknown_loss() -> None:
    protocol = torch_forecasters._normalize_training_protocol({"loss": "banana"})
    assert protocol["loss"] == "mse"


def test_asymmetric_loss_over_penalized_more_than_under() -> None:
    # At low RUL (y_true=10), weight = (1 - 10/125)^2 ≈ 0.85.
    # over: 2.0 * 100 * 0.85 = 170  >  under: 1.0 * 100 = 100
    fn = _loss()
    y_true = torch.tensor([10.0])
    loss_over = fn(torch.tensor([20.0]), y_true).item()
    loss_under = fn(torch.tensor([0.0]), y_true).item()
    assert loss_over > loss_under


def test_asymmetric_loss_weight_zero_at_max_rul() -> None:
    fn = _loss()
    # y_true = 125 → weight = (1 - 125/125)^2 = 0 → overprediction penalty = 0
    loss_val = fn(torch.tensor([135.0]), torch.tensor([125.0])).item()
    assert math.isclose(loss_val, 0.0, abs_tol=1e-6)


def test_asymmetric_loss_weight_one_at_zero_rul() -> None:
    fn = _loss(over_penalty=2.0)
    # y_true = 0, error = +10 → weight = 1.0 → loss = 2.0 * 100 * 1.0 = 200.0
    loss_val = fn(torch.tensor([10.0]), torch.tensor([0.0])).item()
    assert math.isclose(loss_val, 200.0, rel_tol=1e-5)


def test_asymmetric_loss_clamp_handles_rul_above_max() -> None:
    fn = _loss()
    # y_true = 200 > max_rul=125 → weight = clamp(1 - 200/125, min=0)^2 = 0
    loss_val = fn(torch.tensor([210.0]), torch.tensor([200.0]))
    assert torch.isfinite(loss_val)
    assert loss_val.item() >= 0.0


def test_asymmetric_loss_underprediction_uses_beta_only() -> None:
    fn = _loss(under_penalty=1.0)
    # y_true = 0, error = -10 → underprediction branch → loss = 1.0 * 100 = 100.0
    loss_val = fn(torch.tensor([-10.0]), torch.tensor([0.0])).item()
    assert math.isclose(loss_val, 100.0, rel_tol=1e-5)


def test_normalize_protocol_passes_custom_asym_params() -> None:
    protocol = torch_forecasters._normalize_training_protocol({
        "loss": "asymmetric_rul",
        "asym_over_penalty": 5.0,
        "asym_under_penalty": 0.25,
        "asym_max_rul": 100.0,
    })
    assert protocol["asym_over_penalty"] == 5.0
    assert protocol["asym_under_penalty"] == 0.25
    assert protocol["asym_max_rul"] == 100.0


def test_asymmetric_loss_log1p_space_weight_varies() -> None:
    """Regression test for the log1p space mismatch bug.

    When target_transform='log1p' is active, y_true is in log-space [0, log1p(125)≈4.83].
    Using max_rul=125 (raw space) collapses the weight to near-constant ≈ 0.926.
    Using max_rul=log1p(125) makes the weight vary from 1.0 (y_true=0) to 0.0 (y_true=4.83).
    """
    log1p_max = math.log1p(125.0)  # ≈ 4.828

    # With raw-space max_rul=125 and log-space y_true ≈ 4.83, weight ≈ 0.926 (near-constant)
    fn_wrong = _loss(max_rul=125.0)
    y_true_log = torch.tensor([log1p_max])  # log1p(125) — "high RUL" in log space
    loss_wrong = fn_wrong(y_true_log + 1.0, y_true_log).item()  # overprediction

    # With log-space max_rul, y_true=log1p(125) → weight=0 → overprediction penalty=0
    fn_correct = _loss(max_rul=log1p_max)
    loss_correct = fn_correct(y_true_log + 1.0, y_true_log).item()

    # The correctly-scaled loss should be near zero (weight≈0 at max RUL)
    assert loss_correct < loss_wrong, (
        f"Log-space max_rul should reduce high-RUL overprediction penalty: "
        f"correct={loss_correct:.4f} wrong={loss_wrong:.4f}"
    )
    assert math.isclose(loss_correct, 0.0, abs_tol=1e-5), (
        f"At y_true=log1p(125), weight should be 0: got {loss_correct:.6f}"
    )

    # At y_true=0 (imminent failure), weight should be 1.0 regardless of max_rul
    y_true_zero = torch.tensor([0.0])
    loss_zero = fn_correct(y_true_zero + 1.0, y_true_zero).item()
    # over_penalty=2.0, error=1.0, weight=1.0 → 2.0 * 1.0 * 1.0 = 2.0
    assert math.isclose(loss_zero, 2.0, rel_tol=1e-5)
