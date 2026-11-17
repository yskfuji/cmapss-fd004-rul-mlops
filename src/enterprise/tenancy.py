from __future__ import annotations

import re
from dataclasses import dataclass

TENANT_ID_MAX_LEN = 64

# Keep tenant_id safe for logs, filenames, and directory names.
# - allows typical IDs like: t_demo_001, example-tenant
# - forbids whitespace and path-ish characters
_TENANT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


@dataclass(frozen=True)
class Tenant:
    tenant_id: str


def validate_tenant_id(tenant_id: str) -> None:
    if not isinstance(tenant_id, str) or not tenant_id:
        raise ValueError("tenant_id must be a non-empty string")

    if tenant_id != tenant_id.strip():
        raise ValueError("tenant_id must not contain leading/trailing whitespace")

    if len(tenant_id) > TENANT_ID_MAX_LEN:
        raise ValueError("tenant_id too long")

    if any(ch in tenant_id for ch in ("/", "\\")):
        raise ValueError("tenant_id must not contain path separators")
    if ".." in tenant_id:
        raise ValueError("tenant_id must not contain '..'")
    if any(ch in tenant_id for ch in ("\n", "\r", "\t", "\x00")):
        raise ValueError("tenant_id must not contain control characters")

    if _TENANT_ID_RE.fullmatch(tenant_id) is None:
        raise ValueError("tenant_id contains forbidden characters")


def create_tenant(*, tenant_id: str) -> Tenant:
    validate_tenant_id(tenant_id)
    return Tenant(tenant_id=tenant_id)
