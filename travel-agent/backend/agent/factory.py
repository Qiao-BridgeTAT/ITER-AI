"""Runtime composition for disabled and Qwen-backed model gateways."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from backend.agent.model_audit import (
    EncryptedFileModelAuditRecorder,
    ModelAuditRecorder,
    NoopModelAuditRecorder,
)
from backend.agent.model_gateway import DisabledModelGateway, ModelGateway, ModelRuntimeConfig
from backend.agent.qwen_gateway import QwenModelGateway
from backend.config import Settings


@dataclass
class ModelGatewayRuntime:
    gateway: ModelGateway
    closeable: object | None = None
    audit_recorder: ModelAuditRecorder = field(default_factory=NoopModelAuditRecorder)

    async def start(self) -> None:
        if isinstance(self.audit_recorder, EncryptedFileModelAuditRecorder):
            self.audit_recorder.start_retention_maintenance()

    async def close(self) -> None:
        try:
            close = getattr(self.closeable, "aclose", None)
            if close is not None:
                await close()
        finally:
            if isinstance(self.audit_recorder, EncryptedFileModelAuditRecorder):
                await self.audit_recorder.close()


def build_model_gateway(settings: Settings) -> ModelGatewayRuntime:
    config = ModelRuntimeConfig(
        model_name=settings.model.model_name,
        temperature=settings.model.temperature,
        timeout_seconds=settings.model.timeout_seconds,
        prompt_version=settings.model.prompt_version,
    )
    audit_recorder: ModelAuditRecorder = (
        EncryptedFileModelAuditRecorder(
            Path(settings.model_audit.local_path),
            settings.model_audit.encryption_key,
            ttl_days=settings.model_audit.ttl_days,
        )
        if settings.model.enabled
        else NoopModelAuditRecorder()
    )
    if not settings.model.enabled:
        return ModelGatewayRuntime(
            DisabledModelGateway(config),
            audit_recorder=audit_recorder,
        )
    gateway = QwenModelGateway(
        settings.model.api_key,
        settings.model.base_url,
        config,
        audit_recorder=audit_recorder,
    )
    return ModelGatewayRuntime(gateway, gateway, audit_recorder)
