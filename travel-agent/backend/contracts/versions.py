"""Current public wire versions and explicit legacy-client boundaries."""

from __future__ import annotations

from enum import StrEnum

CURRENT_PROTOCOL_VERSION = "2.0.0"
CURRENT_SCHEMA_VERSION = "2.0.0"
V4_PROTOCOL_VERSION = "v4"
V4_SCHEMA_VERSION = "4.0.0"
LEGACY_PROTOCOL_VERSIONS = frozenset({"1.0.0"})
LEGACY_SCHEMA_VERSIONS = frozenset({"1.0.0"})


class ContractVersionDisposition(StrEnum):
    CURRENT = "current"
    UPGRADE_REQUIRED = "upgrade_required"
    UNSUPPORTED = "unsupported"


class V4ContractVersionDisposition(StrEnum):
    CURRENT = "current"
    READ_ONLY_COMPATIBLE = "read_only_compatible"
    UNSUPPORTED = "unsupported"


def classify_contract_versions(
    protocol_version: object,
    schema_version: object,
) -> ContractVersionDisposition:
    """Classify a wire pair without attempting a lossy payload conversion."""

    if protocol_version == CURRENT_PROTOCOL_VERSION and schema_version == CURRENT_SCHEMA_VERSION:
        return ContractVersionDisposition.CURRENT
    if protocol_version in LEGACY_PROTOCOL_VERSIONS and schema_version in LEGACY_SCHEMA_VERSIONS:
        return ContractVersionDisposition.UPGRADE_REQUIRED
    return ContractVersionDisposition.UNSUPPORTED


def classify_v4_contract_versions(
    protocol_version: object,
    schema_version: object,
) -> V4ContractVersionDisposition:
    """Classify V4 negotiation without changing the active V2 command path."""

    if protocol_version == V4_PROTOCOL_VERSION and schema_version == V4_SCHEMA_VERSION:
        return V4ContractVersionDisposition.CURRENT
    if (
        protocol_version == CURRENT_PROTOCOL_VERSION and schema_version == CURRENT_SCHEMA_VERSION
    ) or (
        protocol_version in LEGACY_PROTOCOL_VERSIONS and schema_version in LEGACY_SCHEMA_VERSIONS
    ):
        return V4ContractVersionDisposition.READ_ONLY_COMPATIBLE
    return V4ContractVersionDisposition.UNSUPPORTED
