import json

import pytest

from enterprise.audit import (
    AuditEvent,
    AuditQuery,
    append_audit_event,
    filter_audit_events,
    now_iso_utc,
    parse_iso8601_with_tz,
    to_jsonl,
    validate_audit_event,
)
from enterprise.iam import (
    Approval,
    BreakGlassRequest,
    CustomRole,
    TwoPersonApprovalRequest,
    add_approval,
    add_break_glass_approval,
    build_approval_audit_details,
    build_break_glass_ended_event,
    build_break_glass_started_event,
    build_custom_role_changed_event,
    enforce_break_glass_approved,
    enforce_two_person_approved,
    is_two_person_approved,
    require_two_person_approval,
    validate_custom_role_id,
    validate_custom_role_name,
    validate_custom_role_permissions,
)
from enterprise.network import NetworkAccessPolicy, is_network_access_allowed
from enterprise.tenancy import create_tenant, validate_tenant_id
from enterprise.tenant_settings import IpAllowlist, PrivateConnectivity


def test_audit_query_defaults():
    query = AuditQuery(tenant_id="tenant-a")
    assert query.actor is None
    assert query.event_type is None


def test_append_audit_event_roundtrip(tmp_path):
    path = tmp_path / "audit.jsonl"
    event = AuditEvent(
        tenant_id="tenant-a",
        actor="tester",
        event_type="ACCESS",
        occurred_at="2026-03-23T00:00:00+00:00",
        action="read",
        reason="unit test",
        request_id="req-1",
        details={},
    )
    append_audit_event(event, path=path)
    assert path.exists()
    assert "ACCESS" in path.read_text(encoding="utf-8")


def test_parse_iso8601_with_timezone_accepts_z_suffix():
    parsed = parse_iso8601_with_tz("2026-03-23T00:00:00Z")
    assert parsed.tzinfo is not None


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("", "iso must be a non-empty string"),
        (" 2026-03-23T00:00:00+00:00", "iso must not contain leading/trailing whitespace"),
        ("2026-03-23T00:00:00", "iso must include timezone"),
        ("not-a-date", "invalid iso8601"),
    ],
)
def test_parse_iso8601_with_timezone_rejects_invalid_values(raw, expected):
    with pytest.raises(ValueError, match=expected):
        parse_iso8601_with_tz(raw)


def test_validate_audit_event_and_to_jsonl_roundtrip():
    event = AuditEvent(
        tenant_id="tenant-a",
        actor="tester",
        event_type="ACCESS",
        occurred_at="2026-03-23T00:00:00+00:00",
        action="read",
        reason="unit test",
        request_id="req-1",
        details={"path": "/metrics"},
    )
    validate_audit_event(event)
    payload = json.loads(to_jsonl(event))
    assert payload["event_type"] == "ACCESS"
    assert payload["details"]["path"] == "/metrics"


def test_validate_audit_event_rejects_invalid_fields():
    event = AuditEvent(
        tenant_id="tenant/a",
        actor="bad actor",
        event_type="bad",
        occurred_at="2026-03-23T00:00:00",
        action="Bad",
        reason="x" * 3000,
        request_id="req\n1",
        details=[],
    )
    with pytest.raises(ValueError):
        validate_audit_event(event)


def test_filter_audit_events_applies_query_fields():
    events = [
        AuditEvent(
            tenant_id="tenant-a",
            actor="alice",
            event_type="ACCESS",
            occurred_at="2026-03-23T00:00:00+00:00",
            action="read",
            reason="ok",
            request_id="req-1",
            details={},
        ),
        AuditEvent(
            tenant_id="tenant-a",
            actor="bob",
            event_type="ACCESS",
            occurred_at="2026-03-23T00:01:00+00:00",
            action="write",
            reason="ok",
            request_id="req-2",
            details={},
        ),
        AuditEvent(
            tenant_id="tenant-b",
            actor="alice",
            event_type="ACCESS",
            occurred_at="2026-03-23T00:02:00+00:00",
            action="read",
            reason="ok",
            request_id="req-3",
            details={},
        ),
    ]
    query = AuditQuery(tenant_id="tenant-a", actor="alice", action="read", limit=5)
    filtered = filter_audit_events(events=events, query=query)
    assert [item.request_id for item in filtered] == ["req-1"]


def test_filter_audit_events_rejects_invalid_query():
    with pytest.raises(ValueError, match="limit out of range"):
        filter_audit_events(events=[], query=AuditQuery(tenant_id="tenant-a", limit=0))


def test_tenancy_helpers_validate_and_create():
    validate_tenant_id("tenant-a")
    tenant = create_tenant(tenant_id="tenant-a")
    assert tenant.tenant_id == "tenant-a"
    with pytest.raises(ValueError):
        validate_tenant_id("tenant/../bad")


def test_now_iso_utc_contains_timezone():
    assert now_iso_utc().endswith("+00:00")


def test_require_two_person_approval_rejects_high_impact_change() -> None:
    with pytest.raises(ValueError, match="two-person approval required"):
        require_two_person_approval(is_high_impact_change=True)


def test_require_two_person_approval_allows_non_high_impact_change() -> None:
    require_two_person_approval(is_high_impact_change=False)


@pytest.mark.parametrize(
    "role_id,expected",
    [
        ("", "role_id must be a non-empty string"),
        ("x" * 65, "role_id too long"),
    ],
)
def test_validate_custom_role_id_rejects_invalid_values(role_id: str, expected: str) -> None:
    with pytest.raises(ValueError, match=expected):
        validate_custom_role_id(role_id)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("", "name must be a non-empty string"),
        ("x" * 129, "name too long"),
    ],
)
def test_validate_custom_role_name_rejects_invalid_values(name: str, expected: str) -> None:
    with pytest.raises(ValueError, match=expected):
        validate_custom_role_name(name)


@pytest.mark.parametrize(
    "permissions,expected",
    [
        ([], "permissions must be a tuple"),
        ((), "permissions must be non-empty"),
        (("",), "permission must be a non-empty string"),
        ((("x" * 129),), "permission too long"),
    ],
)
def test_validate_custom_role_permissions_rejects_invalid_values(
    permissions,
    expected: str,
) -> None:
    with pytest.raises(ValueError, match=expected):
        validate_custom_role_permissions(permissions)


def test_build_custom_role_changed_event_includes_role_details() -> None:
    event = build_custom_role_changed_event(
        role=CustomRole(
            tenant_id="tenant-a",
            role_id="role-observer",
            name="Observer",
            permissions=("jobs.read", "models.read"),
            description="read-only role",
        ),
        actor="alice",
        occurred_at="2026-03-23T00:00:00+00:00",
        request_id="req-iam-1",
        reason="sync from policy repo",
    )

    assert event.event_type == "CUSTOM_ROLE_CHANGED"
    assert event.action == "iam.custom_role.upsert"
    assert event.details["role_id"] == "role-observer"
    assert event.details["permissions"] == ["jobs.read", "models.read"]


def test_add_approval_rejects_duplicate_approver() -> None:
    req = TwoPersonApprovalRequest(
        tenant_id="tenant-a",
        request_id="req-1",
        action="model.promote",
        requested_by="alice",
        requested_at="2026-03-23T00:00:00+00:00",
        reason="promote candidate",
        approvals=(Approval(approver="bob", at="2026-03-23T00:01:00+00:00"),),
    )

    with pytest.raises(ValueError, match="duplicate approver"):
        add_approval(req=req, approver="bob", at="2026-03-23T00:02:00+00:00")


def test_two_person_approval_helpers_require_two_distinct_approvers() -> None:
    req = TwoPersonApprovalRequest(
        tenant_id="tenant-a",
        request_id="req-1",
        action="model.promote",
        requested_by="alice",
        requested_at="2026-03-23T00:00:00+00:00",
        reason="promote candidate",
    )
    req = add_approval(req=req, approver="bob", at="2026-03-23T00:01:00+00:00")

    assert is_two_person_approved(req=req) is False
    with pytest.raises(ValueError, match="two-person approval required"):
        enforce_two_person_approved(req=req)

    req = add_approval(req=req, approver="carol", at="2026-03-23T00:02:00+00:00")

    assert is_two_person_approved(req=req) is True
    enforce_two_person_approved(req=req)
    assert build_approval_audit_details(req=req) == {
        "approvers": ["bob", "carol"],
        "requested_by": "alice",
        "requested_at": "2026-03-23T00:00:00+00:00",
    }


def test_add_break_glass_approval_and_events_require_two_distinct_approvers() -> None:
    req = BreakGlassRequest(
        tenant_id="tenant-a",
        request_id="req-bg-1",
        requested_by="alice",
        requested_at="2026-03-23T00:00:00+00:00",
        reason="emergency fix",
        expires_at="2026-03-23T04:00:00+00:00",
    )
    req = add_break_glass_approval(req=req, approver="bob", at="2026-03-23T00:01:00+00:00")

    with pytest.raises(ValueError, match="break-glass requires two distinct approvals"):
        enforce_break_glass_approved(req=req)

    with pytest.raises(ValueError, match="duplicate approver"):
        add_break_glass_approval(req=req, approver="bob", at="2026-03-23T00:02:00+00:00")

    req = add_break_glass_approval(req=req, approver="carol", at="2026-03-23T00:03:00+00:00")
    enforce_break_glass_approved(req=req)

    started = build_break_glass_started_event(
        req=req,
        occurred_at="2026-03-23T00:04:00+00:00",
    )
    ended = build_break_glass_ended_event(
        tenant_id="tenant-a",
        actor="alice",
        occurred_at="2026-03-23T05:00:00+00:00",
        request_id="req-bg-1",
        reason="maintenance completed",
    )

    assert started.event_type == "BREAK_GLASS_STARTED"
    assert started.details["approvers"] == ["bob", "carol"]
    assert ended.event_type == "BREAK_GLASS_ENDED"
    assert ended.details == {}


def test_network_access_policy_handles_allowlist_and_private_connectivity() -> None:
    policy = NetworkAccessPolicy(
        ip_allowlist=IpAllowlist(enabled=True, entries=("10.0.0.0/8",)),
        private_connectivity=PrivateConnectivity(enabled=True),
    )

    assert (
        is_network_access_allowed(policy=policy, ip="10.1.2.3", connection="private") is True
    )
    assert (
        is_network_access_allowed(policy=policy, ip="10.1.2.3", connection="public") is False
    )
    assert (
        is_network_access_allowed(policy=policy, ip="192.168.1.10", connection="private") is False
    )


