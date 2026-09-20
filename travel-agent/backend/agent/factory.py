"""Runtime composition for disabled and Qwen-backed model gateways."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from backend.agent.model_audit import (
    EncryptedFileModelAuditRecorder,
    ModelAuditRecorder,
    NoopModelAuditRecorder,
)
from backend.agent.model_gateway import DisabledModelGateway, ModelGateway, ModelRuntimeConfig
from backend.agent.qwen_gateway import QwenModelGateway
from backend.config import Settings

ATTRACTION_MODEL_NAME = "qwen3.8-flash"

DINING_REVIEW_MODEL_NAME = "qwen3.8-max"
PLANNER_MODEL_NAME = "qwen3.8-flash"


@dataclass
class ModelGatewayRuntime:
    gateway: ModelGateway
    closeable: object | None = None
    audit_recorder: ModelAuditRecorder = field(default_factory=NoopModelAuditRecorder)
    dining_review_gateway: ModelGateway | None = None

    attraction_gateway: ModelGateway | None = None
    planner_gateway: ModelGateway | None = None

    async def start(self) -> None:
        if isinstance(self.audit_recorder, EncryptedFileModelAuditRecorder):
            self.audit_recorder.start_retention_maintenance()

    async def close(self) -> None:
        try:
            try:
                close = getattr(self.closeable, "aclose", None)
                if close is not None:
                    await close()
            finally:
                if self.dining_review_gateway is not self.closeable:
                    review_close = getattr(self.dining_review_gateway, "aclose", None)
                    if review_close is not None:
                        await review_close()
        finally:
            try:
                attraction_close = getattr(self.attraction_gateway, "aclose", None)
                if attraction_close is not None:
                    await attraction_close()
            finally:
                try:
                    planner_close = getattr(self.planner_gateway, "aclose", None)
                    if planner_close is not None:
                        await planner_close()
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
    attraction_config = replace(config, model_name=ATTRACTION_MODEL_NAME)
    review_config = replace(config, model_name=DINING_REVIEW_MODEL_NAME)
    planner_config = replace(config, model_name=PLANNER_MODEL_NAME)
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
            dining_review_gateway=DisabledModelGateway(review_config),
            attraction_gateway=DisabledModelGateway(attraction_config),
            planner_gateway=DisabledModelGateway(planner_config),
        )
    gateway = QwenModelGateway(
        settings.model.api_key,
        settings.model.base_url,
        config,
        audit_recorder=audit_recorder,
    )
    review_gateway = QwenModelGateway(
        settings.model.api_key,
        settings.model.base_url,
        review_config,
        audit_recorder=audit_recorder,
    )
    return ModelGatewayRuntime(
        gateway,
        gateway,
        audit_recorder,
        dining_review_gateway=review_gateway,
        attraction_gateway=QwenModelGateway(
            settings.model.api_key,
            settings.model.base_url,
            attraction_config,
            audit_recorder=audit_recorder,
        ),
        planner_gateway=QwenModelGateway(
            settings.model.api_key,
            settings.model.base_url,
            planner_config,
            audit_recorder=audit_recorder,
        ),
    )
