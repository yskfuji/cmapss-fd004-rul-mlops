from __future__ import annotations

from dataclasses import dataclass

from .audit import AuditEvent
from .tenancy import validate_tenant_id


@dataclass(frozen=True)
class PortabilityExportRequest:
    tenant_id: str
    request_id: str
    requested_by: str
    requested_at: str
    format: str = "jsonl"


def build_portability_export_requested_event(
    *,
    req: PortabilityExportRequest,
    occurred_at: str,
) -> AuditEvent:
    validate_tenant_id(req.tenant_id)
    if not isinstance(req.request_id, str) or not req.request_id.strip():
        raise ValueError("request_id must be a non-empty string")
    if not isinstance(req.requested_by, str) or not req.requested_by.strip():
        raise ValueError("requested_by must be a non-empty string")

    return AuditEvent(
        tenant_id=req.tenant_id,
        event_type="PORTABILITY_EXPORT_REQUESTED",
        occurred_at=occurred_at,
        actor=req.requested_by,
        action="data.portability.export.request",
        reason="",
        request_id=req.request_id,
        details={
            "requested_at": req.requested_at,
            "format": req.format,
        },
    )
