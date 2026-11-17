from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import Request

from enterprise.network import ConnectionType, NetworkAccessPolicy, is_network_access_allowed
from enterprise.tenancy import validate_tenant_id
from enterprise.tenant_settings import IpAllowlist, PrivateConnectivity

from .config import env_bool, env_first
from .errors import ApiError


@dataclass(frozen=True)
class RequestPolicyContext:
    tenant_id: str
    client_ip: str
    connection_type: ConnectionType


@dataclass(frozen=True)
class RequestPolicyConfig:
    require_tenant_header: bool
    network_policy: NetworkAccessPolicy


def load_request_policy_config() -> RequestPolicyConfig:
    require_tenant_header = env_bool(
        "RULFM_FORECASTING_API_REQUIRE_TENANT",
        default=False,
    )
    allowlist_enabled = env_bool(
        "RULFM_FORECASTING_API_IP_ALLOWLIST_ENABLED",
        default=False,
    )
    private_connectivity_enabled = env_bool(
        "RULFM_FORECASTING_API_PRIVATE_CONNECTIVITY_REQUIRED",
        default=False,
    )
    raw_allowlist = (
        env_first(
            "RULFM_FORECASTING_API_IP_ALLOWLIST",
        )
        or ""
    )
    entries = tuple(item.strip() for item in raw_allowlist.split(",") if item.strip())
    return RequestPolicyConfig(
        require_tenant_header=require_tenant_header,
        network_policy=NetworkAccessPolicy(
            ip_allowlist=IpAllowlist(enabled=allowlist_enabled, entries=entries),
            private_connectivity=PrivateConnectivity(enabled=private_connectivity_enabled),
        ),
    )


def resolve_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    client = getattr(request, "client", None)
    host = getattr(client, "host", None)
    return str(host or "unknown")


def resolve_connection_type(raw: str | None) -> ConnectionType:
    value = str(raw or "public").strip().lower()
    return "private" if value == "private" else "public"


def enforce_request_policy(
    request: Request,
    *,
    tenant_id: str | None,
    connection_type: str | None,
) -> RequestPolicyContext:
    config = load_request_policy_config()
    resolved_tenant_id = str(tenant_id or "public").strip() or "public"
    if config.require_tenant_header and not tenant_id:
        raise ApiError(
            status_code=400,
            error_code="A14",
            message="tenant header が必要です",
            details={
                "error": "X-Tenant-Id header is required",
                "next_action": "X-Tenant-Id を指定してください",
            },
        )
    try:
        validate_tenant_id(resolved_tenant_id)
    except ValueError as exc:
        raise ApiError(
            status_code=400,
            error_code="A14",
            message="tenant_id が不正です",
            details={
                "error": str(exc),
                "next_action": "X-Tenant-Id を見直してください",
            },
        ) from exc

    resolved_connection_type = resolve_connection_type(connection_type)
    client_ip = resolve_client_ip(request)
    policy = config.network_policy
    policy_active = policy.ip_allowlist.enabled or policy.private_connectivity.enabled
    if policy_active:
        try:
            allowed = is_network_access_allowed(
                policy=policy,
                ip=client_ip,
                connection=resolved_connection_type,
            )
        except ValueError:
            allowed = False
        if not allowed:
            raise ApiError(
                status_code=403,
                error_code="A15",
                message="network policy により拒否されました",
                details={
                    "error": (
                        "network access denied for "
                        f"connection_type={resolved_connection_type}"
                    ),
                    "next_action": "allowlist と private connectivity 設定を確認してください",
                },
            )

    request.state.tenant_id = resolved_tenant_id
    request.state.connection_type = resolved_connection_type
    request.state.client_ip = client_ip
    return RequestPolicyContext(
        tenant_id=resolved_tenant_id,
        client_ip=client_ip,
        connection_type=resolved_connection_type,
    )


def append_tenant_context(details: dict[str, Any], *, tenant_id: str) -> dict[str, Any]:
    payload = dict(details)
    payload["tenant_id"] = tenant_id
    return payload
