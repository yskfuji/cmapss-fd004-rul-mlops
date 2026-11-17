from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from .tenancy import validate_tenant_id

_EVENT_TYPE_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_ACTION_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_IDENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@+\-]{0,127}$")


@dataclass(frozen=True)
class AuditQuery:
    tenant_id: str
    actor: str | None = None
    event_type: str | None = None
    action: str | None = None
    request_id: str | None = None
    limit: int = 100


@dataclass(frozen=True)
class AuditEvent:
    tenant_id: str
    event_type: str
    occurred_at: str
    actor: str
    action: str
    reason: str
    request_id: str
    details: dict


def parse_iso8601_with_tz(iso: str) -> datetime:
    if not isinstance(iso, str) or not iso.strip():
        raise ValueError("iso must be a non-empty string")
    if iso != iso.strip():
        raise ValueError("iso must not contain leading/trailing whitespace")

    s = iso
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(s)
    except Exception as exc:
        raise ValueError(f"invalid iso8601: {exc}") from exc

    if dt.tzinfo is None:
        raise ValueError("iso must include timezone")
    return dt


def _validate_identifier(value: str, *, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{field} must not contain leading/trailing whitespace")
    if any(ch in value for ch in ("/", "\\")):
        raise ValueError(f"{field} must not contain path separators")
    if any(ch in value for ch in ("\n", "\r", "\t", "\x00")):
        raise ValueError(f"{field} must not contain control characters")
    if _IDENT_RE.fullmatch(value) is None:
        raise ValueError(f"{field} contains forbidden characters")


def validate_audit_event(event: AuditEvent) -> None:
    validate_tenant_id(event.tenant_id)
    if not isinstance(event.event_type, str) or not event.event_type.strip():
        raise ValueError("event_type must be a non-empty string")
    if _EVENT_TYPE_RE.fullmatch(event.event_type) is None:
        raise ValueError("event_type invalid")

    parse_iso8601_with_tz(event.occurred_at)

    _validate_identifier(event.actor, field="actor")
    if not isinstance(event.action, str) or not event.action.strip():
        raise ValueError("action must be a non-empty string")
    if _ACTION_RE.fullmatch(event.action) is None:
        raise ValueError("action invalid")

    if not isinstance(event.reason, str):
        raise ValueError("reason must be a string")
    if len(event.reason) > 2048:
        raise ValueError("reason too long")

    _validate_identifier(event.request_id, field="request_id")
    if not isinstance(event.details, dict):
        raise ValueError("details must be an object")


def now_iso_utc() -> str:
    return datetime.now(UTC).isoformat()


def to_jsonl(event: AuditEvent) -> str:
    validate_audit_event(event)
    return json.dumps(asdict(event), ensure_ascii=False, sort_keys=True)


def append_audit_event(event: AuditEvent, *, path: str | Path) -> None:
    validate_audit_event(event)
    audit_path = Path(path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(to_jsonl(event))
        handle.write("\n")


def filter_audit_events(*, events: list[AuditEvent], query: AuditQuery) -> list[AuditEvent]:
    validate_tenant_id(query.tenant_id)
    if not isinstance(query.limit, int) or query.limit < 1 or query.limit > 1000:
        raise ValueError("limit out of range")

    if query.actor is not None:
        _validate_identifier(query.actor, field="actor")
    if query.request_id is not None:
        _validate_identifier(query.request_id, field="request_id")
    if query.event_type is not None:
        if _EVENT_TYPE_RE.fullmatch(query.event_type) is None:
            raise ValueError("event_type invalid")
    if query.action is not None:
        if _ACTION_RE.fullmatch(query.action) is None:
            raise ValueError("action invalid")

    out: list[AuditEvent] = []
    for ev in events:
        if ev.tenant_id != query.tenant_id:
            continue

        validate_audit_event(ev)
        if query.actor is not None and ev.actor != query.actor:
            continue
        if query.event_type is not None and ev.event_type != query.event_type:
            continue
        if query.action is not None and ev.action != query.action:
            continue
        if query.request_id is not None and ev.request_id != query.request_id:
            continue
        out.append(ev)
        if len(out) >= query.limit:
            break

    return out
