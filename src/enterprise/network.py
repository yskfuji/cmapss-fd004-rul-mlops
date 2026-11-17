from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from typing import Literal

from .tenant_settings import IpAllowlist, PrivateConnectivity


@dataclass(frozen=True)
class NetworkAccessPolicy:
    ip_allowlist: IpAllowlist
    private_connectivity: PrivateConnectivity


ConnectionType = Literal["public", "private"]


def is_network_access_allowed(
    *,
    policy: NetworkAccessPolicy,
    ip: str,
    connection: ConnectionType,
) -> bool:
    """Return True if access should be allowed under the policy.

    Rules (minimal):
    - If private_connectivity.enabled, only "private" connection is allowed.
    - If ip_allowlist.enabled, IP must be contained in at least one CIDR entry.
    - Invalid IP/CIDR -> deny by raising ValueError (caller should treat as deny).
    """

    if policy.private_connectivity.enabled and connection != "private":
        return False

    if not policy.ip_allowlist.enabled:
        return True

    addr = ipaddress.ip_address(ip)
    for cidr in policy.ip_allowlist.entries:
        net = ipaddress.ip_network(cidr, strict=False)
        if addr in net:
            return True
    return False
