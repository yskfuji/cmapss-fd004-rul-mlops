from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .audit import AuditEvent
from .tenancy import validate_tenant_id


@dataclass(frozen=True)
class IamSnapshot:
    tenant_id: str
    require_sso: bool


@dataclass(frozen=True)
class CustomRole:
    tenant_id: str
    role_id: str
    name: str
    permissions: tuple[str, ...]
    description: str = ""


def validate_custom_role_id(role_id: str) -> None:
    if not isinstance(role_id, str) or not role_id.strip():
        raise ValueError("role_id must be a non-empty string")
    if len(role_id) > 64:
        raise ValueError("role_id too long")


def validate_custom_role_name(name: str) -> None:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")
    if len(name) > 128:
        raise ValueError("name too long")


def validate_custom_role_permissions(permissions: tuple[str, ...]) -> None:
    if not isinstance(permissions, tuple):
        raise ValueError("permissions must be a tuple")
    if not permissions:
        raise ValueError("permissions must be non-empty")
    for p in permissions:
        if not isinstance(p, str) or not p.strip():
            raise ValueError("permission must be a non-empty string")
        if len(p) > 128:
            raise ValueError("permission too long")


def build_custom_role_changed_event(
    *,
    role: CustomRole,
    actor: str,
    occurred_at: str,
    request_id: str,
    reason: str = "",
) -> AuditEvent:
    validate_tenant_id(role.tenant_id)
    validate_custom_role_id(role.role_id)
    validate_custom_role_name(role.name)
    validate_custom_role_permissions(role.permissions)

    return AuditEvent(
        tenant_id=role.tenant_id,
        event_type="CUSTOM_ROLE_CHANGED",
        occurred_at=occurred_at,
        actor=actor,
        action="iam.custom_role.upsert",
        reason=reason,
        request_id=request_id,
        details={
            "role_id": role.role_id,
            "name": role.name,
            "permissions": list(role.permissions),
            "description": role.description,
        },
    )


def require_two_person_approval(*, is_high_impact_change: bool) -> None:
    if is_high_impact_change:
        raise ValueError("two-person approval required")


@dataclass(frozen=True)
class Approval:
    approver: str
    at: str


@dataclass(frozen=True)
class TwoPersonApprovalRequest:
    tenant_id: str
    request_id: str
    action: str
    requested_by: str
    requested_at: str
    reason: str
    approvals: tuple[Approval, ...] = ()


def add_approval(
    *,
    req: TwoPersonApprovalRequest,
    approver: str,
    at: str,
) -> TwoPersonApprovalRequest:
    validate_tenant_id(req.tenant_id)
    if not isinstance(approver, str) or not approver.strip():
        raise ValueError("approver must be non-empty string")
    if not isinstance(at, str) or not at.strip():
        raise ValueError("at must be non-empty string")

    existing = {a.approver for a in req.approvals}
    if approver in existing:
        raise ValueError("duplicate approver")

    return TwoPersonApprovalRequest(
        tenant_id=req.tenant_id,
        request_id=req.request_id,
        action=req.action,
        requested_by=req.requested_by,
        requested_at=req.requested_at,
        reason=req.reason,
        approvals=req.approvals + (Approval(approver=approver, at=at),),
    )


def is_two_person_approved(*, req: TwoPersonApprovalRequest) -> bool:
    return len({a.approver for a in req.approvals}) >= 2


def enforce_two_person_approved(*, req: TwoPersonApprovalRequest) -> None:
    validate_tenant_id(req.tenant_id)
    if not is_two_person_approved(req=req):
        raise ValueError("two-person approval required")


def build_approval_audit_details(*, req: TwoPersonApprovalRequest) -> dict[str, Any]:
    approvers = sorted({a.approver for a in req.approvals})
    return {
        "approvers": approvers,
        "requested_by": req.requested_by,
        "requested_at": req.requested_at,
    }


@dataclass(frozen=True)
class BreakGlassRequest:
    tenant_id: str
    request_id: str
    requested_by: str
    requested_at: str
    reason: str
    expires_at: str
    approvals: tuple[Approval, ...] = ()


def add_break_glass_approval(
    *,
    req: BreakGlassRequest,
    approver: str,
    at: str,
) -> BreakGlassRequest:
    validate_tenant_id(req.tenant_id)
    if not isinstance(approver, str) or not approver.strip():
        raise ValueError("approver must be non-empty string")
    if not isinstance(at, str) or not at.strip():
        raise ValueError("at must be non-empty string")

    existing = {a.approver for a in req.approvals}
    if approver in existing:
        raise ValueError("duplicate approver")

    return BreakGlassRequest(
        tenant_id=req.tenant_id,
        request_id=req.request_id,
        requested_by=req.requested_by,
        requested_at=req.requested_at,
        reason=req.reason,
        expires_at=req.expires_at,
        approvals=req.approvals + (Approval(approver=approver, at=at),),
    )


def enforce_break_glass_approved(*, req: BreakGlassRequest) -> None:
    validate_tenant_id(req.tenant_id)
    if len({a.approver for a in req.approvals}) < 2:
        raise ValueError("break-glass requires two distinct approvals")


def build_break_glass_started_event(
    *,
    req: BreakGlassRequest,
    occurred_at: str,
) -> AuditEvent:
    validate_tenant_id(req.tenant_id)
    enforce_break_glass_approved(req=req)
    approvers = sorted({a.approver for a in req.approvals})
    return AuditEvent(
        tenant_id=req.tenant_id,
        event_type="BREAK_GLASS_STARTED",
        occurred_at=occurred_at,
        actor=req.requested_by,
        action="break_glass.start",
        reason=req.reason,
        request_id=req.request_id,
        details={
            "expires_at": req.expires_at,
            "approvers": approvers,
        },
    )


def build_break_glass_ended_event(
    *,
    tenant_id: str,
    actor: str,
    occurred_at: str,
    request_id: str,
    reason: str = "",
) -> AuditEvent:
    validate_tenant_id(tenant_id)
    return AuditEvent(
        tenant_id=tenant_id,
        event_type="BREAK_GLASS_ENDED",
        occurred_at=occurred_at,
        actor=actor,
        action="break_glass.end",
        reason=reason,
        request_id=request_id,
        details={},
    )
