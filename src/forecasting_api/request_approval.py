from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import Request

from enterprise.iam import TwoPersonApprovalRequest, add_approval, enforce_two_person_approved

from .config import env_bool, env_first
from .errors import ApiError


@dataclass(frozen=True)
class TrainApprovalConfig:
    require_train_approval: bool
    oidc_required_subjects: tuple[str, ...]
    oidc_required_groups: tuple[str, ...]
    oidc_group_claim_names: tuple[str, ...]


def _csv_tokens(raw: str | None) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(raw or "").split(",") if item.strip())


def _claim_values(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, list | tuple | set):
        out: list[str] = []
        for item in value:
            token = str(item or "").strip()
            if token:
                out.append(token)
        return tuple(out)
    return ()


def _oidc_approval_enabled(config: TrainApprovalConfig) -> bool:
    return bool(config.oidc_required_subjects or config.oidc_required_groups)


def _extract_oidc_groups(
    claims: dict[str, Any],
    *,
    claim_names: tuple[str, ...],
) -> tuple[str, ...]:
    groups: list[str] = []
    seen: set[str] = set()
    for claim_name in claim_names:
        for value in _claim_values(claims.get(claim_name)):
            if value not in seen:
                seen.add(value)
                groups.append(value)
    return tuple(groups)


def load_train_approval_config() -> TrainApprovalConfig:
    return TrainApprovalConfig(
        require_train_approval=env_bool(
            "RULFM_FORECASTING_API_REQUIRE_TRAIN_APPROVAL",
            default=False,
        ),
        oidc_required_subjects=_csv_tokens(
            env_first(
                "RULFM_FORECASTING_API_TRAIN_APPROVER_SUBJECTS",
            )
        ),
        oidc_required_groups=_csv_tokens(
            env_first(
                "RULFM_FORECASTING_API_TRAIN_APPROVER_GROUPS",
            )
        ),
        oidc_group_claim_names=_csv_tokens(
            env_first(
                "RULFM_FORECASTING_API_TRAIN_APPROVER_GROUP_CLAIMS",
            )
            or "groups,roles"
        ),
    )


def enforce_train_request_approval(
    request: Request,
    *,
    approved_by: str | None,
    approval_reason: str | None,
) -> None:
    config = load_train_approval_config()
    if not config.require_train_approval:
        return

    if _oidc_approval_enabled(config):
        if str(getattr(request.state, "auth_method", "") or "") != "oidc-bearer":
            raise ApiError(
                status_code=403,
                error_code="A16",
                message="train には OIDC 承認済み principal が必要です",
                details={
                    "error": "oidc bearer token with approval claims is required",
                    "next_action": "OIDC Bearer token で再認証してください",
                },
            )

        subject = str(getattr(request.state, "auth_subject", "") or "").strip()
        claims_obj = getattr(request.state, "auth_claims", {})
        claims = dict(claims_obj) if isinstance(claims_obj, dict) else {}
        claim_groups = _extract_oidc_groups(claims, claim_names=config.oidc_group_claim_names)

        if config.oidc_required_subjects and subject not in set(config.oidc_required_subjects):
            raise ApiError(
                status_code=403,
                error_code="A16",
                message="train には承認済み subject が必要です",
                details={
                    "error": "authenticated subject is not allowed for train approval",
                    "next_action": "承認済み OIDC principal で再実行してください",
                },
            )

        if config.oidc_required_groups and not (
            set(claim_groups) & set(config.oidc_required_groups)
        ):
            raise ApiError(
                status_code=403,
                error_code="A16",
                message="train には承認済み group claim が必要です",
                details={
                    "error": "authenticated principal does not belong to an approved train group",
                    "next_action": (
                        "承認済み group claim を持つ OIDC principal で"
                        "再実行してください"
                    ),
                },
            )

        request.state.approvers = (subject,) if subject else ()
        request.state.approval_mode = "oidc-claim"
        return

    approvers = [item.strip() for item in str(approved_by or "").split(",") if item.strip()]
    if len(approvers) < 2:
        raise ApiError(
            status_code=403,
            error_code="A16",
            message="train には two-person approval が必要です",
            details={
                "error": "at least two distinct approvers are required",
                "next_action": "X-Approved-By に 2 名以上をカンマ区切りで指定してください",
            },
        )

    req = TwoPersonApprovalRequest(
        tenant_id=str(getattr(request.state, "tenant_id", "public") or "public"),
        request_id=str(getattr(request.state, "request_id", "") or "train-request"),
        action="forecasting.train.start",
        requested_by=str(getattr(request.state, "auth_subject", None) or "operator"),
        requested_at="1970-01-01T00:00:00+00:00",
        reason=str(approval_reason or "train approval"),
    )
    for approver in approvers:
        req = add_approval(req=req, approver=approver, at="1970-01-01T00:00:00+00:00")
    try:
        enforce_two_person_approved(req=req)
    except ValueError as exc:
        raise ApiError(
            status_code=403,
            error_code="A16",
            message="train には two-person approval が必要です",
            details={
                "error": str(exc),
                "next_action": "X-Approved-By の承認者を見直してください",
            },
        ) from exc

    request.state.approvers = tuple(sorted({item for item in approvers}))
    request.state.approval_mode = "header"