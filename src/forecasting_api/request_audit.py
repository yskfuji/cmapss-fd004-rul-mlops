from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from enterprise.audit import AuditEvent, to_jsonl

from .file_store import exclusive_lock


def _canonical_hash_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _last_entry_hash(path: Path) -> str | None:
    try:
        if not path.exists() or path.stat().st_size <= 0:
            return None
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            return None
        last = json.loads(lines[-1])
        if not isinstance(last, dict):
            return None
        raw_details = last.get("details")
        details = raw_details if isinstance(raw_details, dict) else {}
        value = details.get("entry_hash")
        return str(value) if value else None
    except Exception:
        return None


def append_request_audit_log(
    entry: dict[str, Any],
    *,
    path: Path,
    enabled: bool = True,
) -> None:
    if not enabled:
        return
    payload = dict(entry)
    append_request_audit_event(
        path=path,
        request_id=(str(payload.get("request_id")) if payload.get("request_id") else None),
        method=str(payload.get("method") or "UNKNOWN"),
        path_name=str(payload.get("path") or "unknown"),
        query=str(payload.get("query") or ""),
        status_code=int(payload.get("status_code") or 0),
        client_host=(str(payload.get("client_host")) if payload.get("client_host") else None),
        auth_method=str(payload.get("auth_method") or "none"),
        request_body_bytes=int(payload.get("request_body_bytes") or 0),
        actor=(str(payload.get("actor")) if payload.get("actor") else None),
        occurred_at=(str(payload.get("timestamp_utc")) if payload.get("timestamp_utc") else None),
        tenant_id=str(payload.get("tenant_id") or "public"),
    )


def append_request_audit_event(
    *,
    path: Path,
    request_id: str | None,
    method: str,
    path_name: str,
    query: str,
    status_code: int,
    client_host: str | None,
    auth_method: str,
    request_body_bytes: int,
    actor: str | None = None,
    occurred_at: str | None = None,
    tenant_id: str = "public",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(path):
        prev_hash = _last_entry_hash(path)
        event = AuditEvent(
            tenant_id=tenant_id,
            event_type="API_REQUEST_COMPLETED",
            occurred_at=occurred_at or datetime.now(UTC).isoformat(),
            actor=(actor or auth_method or "anonymous"),
            action="api.request.complete",
            reason="",
            request_id=request_id or uuid4().hex,
            details={
                "path": path_name,
                "method": method,
                "query": query,
                "status_code": int(status_code),
                "client_host": client_host,
                "auth_method": auth_method,
                "request_body_bytes": int(request_body_bytes),
                "prev_hash": prev_hash,
            },
        )
        payload = json.loads(to_jsonl(event))
        details = payload.get("details") if isinstance(payload.get("details"), dict) else {}
        entry_hash = sha256(
            f"{prev_hash or ''}\n{_canonical_hash_payload(payload)}".encode("utf-8")
        ).hexdigest()
        details["entry_hash"] = entry_hash
        payload["details"] = details
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            handle.write("\n")