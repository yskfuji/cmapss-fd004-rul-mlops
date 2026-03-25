from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IpAllowlist:
    enabled: bool
    entries: tuple[str, ...] = ()


@dataclass(frozen=True)
class PrivateConnectivity:
    enabled: bool
