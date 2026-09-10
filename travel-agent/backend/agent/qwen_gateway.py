"""Qwen OpenAI-compatible adapter behind the vendor-neutral ModelGateway."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections.abc import AsyncIterator, Awaitable, Mapping
from typing import Any, TypeVar, cast
from uuid import uuid4

import httpx
from pydantic import ValidationError

from backend.agent.model_audit import (
    ModelAuditError,
    ModelAuditRecorder,
    NoopModelAuditRecorder,
    canonical_audit_hash,
    private_reasoning_was_redacted,
    register_model_call,
)
from backend.agent.model_gateway import (
    ModelCancellation,
    ModelFailureCode,
    ModelGatewayError,
    ModelRequest,
    ModelRuntimeConfig,
    ModelStreamChunk,
    ModelStructuredResult,
    ModelUsage,
    StructuredValue,
)


class QwenModelGateway:
    """Translate Qwen's compatible HTTP shapes into internal model contracts."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        config: ModelRuntimeConfig,
        *,
        client: httpx.AsyncClient | None = None,
        audit_recorder: ModelAuditRecorder | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("QWEN_API_KEY must not be empty")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._config = config
        self._client = client or httpx.AsyncClient(timeout=config.timeout_seconds)
        self._owns_client = client is None
        self._audit_recorder = audit_recorder or NoopModelAuditRecorder()

    @property
    def config(self) -> ModelRuntimeConfig:
        return self._config

    @property
    def audit_recorder(self) -> ModelAuditRecorder:
        return self._audit_recorder

    async def stream_text(
        self,
        request: ModelRequest,
        *,
        cancellation: ModelCancellation | None = None,
    ) -> AsyncIterator[ModelStreamChunk]:
        operation = "stream_text"
        payload = self._request_payload(request, stream=True)
        call_id, audit_base = await self._start_audit_call(
            request,
            operation=operation,
            request_payload=payload,
            output_schema=None,
        )
        started = time.monotonic()
        response: httpx.Response | None = None
        raw_events: list[Any] = []
        normalized_chunks: list[dict[str, Any]] = []
        reconstructed: list[str] = []
        finish_reason: str | None = None
        usage: ModelUsage | None = None
        response_recorded = False
        stream_context = self._client.stream(
            "POST",
            self._endpoint,
            headers=self._headers,
            json=payload,
            timeout=self._config.timeout_seconds,
        )
        try:
            _check_cancellation(cancellation, operation)
            response = await _await_or_cancel(stream_context.__aenter__(), cancellation, operation)
            try:
                if response.status_code >= 400:
                    await response.aread()
                _raise_for_status(response, operation)
                saw_payload = False
                lines = response.aiter_lines().__aiter__()
                saw_done = False
                while True:
                    try:
                        line = await _await_or_cancel(lines.__anext__(), cancellation, operation)
                    except StopAsyncIteration:
                        break
                    _check_cancellation(cancellation, operation)
                    line = line.strip()
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        raise _model_error(ModelFailureCode.MALFORMED_RESPONSE, operation)
                    data = line[5:].strip()
                    if data == "[DONE]":
                        saw_done = True
                        break
                    saw_payload = True
                    try:
                        raw_event: Any = json.loads(data)
                    except json.JSONDecodeError:
                        raw_event = data
                    raw_events.append(raw_event)
                    chunk = _parse_stream_event(data, operation).model_copy(
                        update={"audit_call_id": call_id}
                    )
                    normalized_chunks.append(chunk.model_dump(mode="json"))
                    if chunk.delta:
                        reconstructed.append(chunk.delta)
                    if chunk.finish_reason is not None:
                        finish_reason = chunk.finish_reason
                    if chunk.usage is not None:
                        usage = chunk.usage
                    yield chunk
                if not saw_payload:
                    raise _model_error(ModelFailureCode.MALFORMED_RESPONSE, operation)
                await self._record_audit(
                    "llm_gateway_completed",
                    call_id=call_id,
                    operation=operation,
                    payload={
                        **audit_base,
                        **_response_audit(
                            response,
                            raw_body=raw_events,
                            chunks=normalized_chunks,
                        ),
                        "response_body_reconstructed": "".join(reconstructed),
                        "schema_validation_result": {"status": "not_applicable"},
                        "parsed_output": normalized_chunks,
                        "accepted_or_rejected": "gateway_accepted",
                        "finish_reason": finish_reason,
                        "token_usage": usage.model_dump(mode="json") if usage else None,
                        "stream_done_received": saw_done,
                        "completed_at": _utc_now(),
                        "latency_ms": _elapsed_ms(started),
                    },
                )
                response_recorded = True
            finally:
                await stream_context.__aexit__(None, None, None)
        except asyncio.CancelledError:
            if not response_recorded:
                response_recorded = True
                await self._record_transport_error(
                    call_id,
                    operation,
                    audit_base,
                    _model_error(ModelFailureCode.CANCELLED, operation),
                    started,
                    response,
                    raw_events,
                    normalized_chunks,
                )
            raise
        except ModelGatewayError as error:
            bound = _with_audit_call_id(error, call_id)
            if bound.code is ModelFailureCode.AUDIT_UNAVAILABLE:
                response_recorded = True
                raise
            await self._record_audit(
                "llm_gateway_failed",
                call_id=call_id,
                operation=operation,
                payload={
                    **audit_base,
                    **_response_audit(
                        response,
                        raw_body=raw_events,
                        chunks=normalized_chunks,
                    ),
                    "response_body_reconstructed": "".join(reconstructed),
                    "schema_validation_result": {"status": "not_completed"},
                    "accepted_or_rejected": "rejected",
                    "failure_stage": _failure_stage(bound),
                    "failure_code": bound.code.value,
                    "validation_issues": list(bound.validation_issues),
                    "completed_at": _utc_now(),
                    "latency_ms": _elapsed_ms(started),
                },
            )
            response_recorded = True
            raise bound from error
        except httpx.TimeoutException:
            transport_error = _model_error(
                ModelFailureCode.TIMEOUT,
                operation,
                retryable=True,
                audit_call_id=call_id,
            )
            await self._record_transport_error(
                call_id,
                operation,
                audit_base,
                transport_error,
                started,
                response,
                raw_events,
                normalized_chunks,
            )
            response_recorded = True
            raise transport_error from None
        except httpx.HTTPError:
            transport_error = _model_error(
                ModelFailureCode.UNAVAILABLE,
                operation,
                retryable=True,
                audit_call_id=call_id,
            )
            await self._record_transport_error(
                call_id,
                operation,
                audit_base,
                transport_error,
                started,
                response,
                raw_events,
                normalized_chunks,
            )
            response_recorded = True
            raise transport_error from None
        finally:
            if not response_recorded:
                await self._record_audit(
                    "llm_stream_consumer_closed",
                    call_id=call_id,
                    operation=operation,
                    payload={
                        **audit_base,
                        **_response_audit(
                            response,
                            raw_body=raw_events,
                            chunks=normalized_chunks,
                        ),
                        "response_body_reconstructed": "".join(reconstructed),
                        "accepted_or_rejected": "pending_business_guard",
                        "completed_at": _utc_now(),
                        "latency_ms": _elapsed_ms(started),
                    },
                )

    async def generate_structured(
        self,
        request: ModelRequest,
        output_type: type[StructuredValue],
        *,
        cancellation: ModelCancellation | None = None,
        validation_context: Mapping[str, Any] | None = None,
    ) -> ModelStructuredResult[StructuredValue]:
        operation = "structured"
        payload = self._request_payload(request, stream=False)
        if request.structured_output_mode == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": _schema_name(output_type),
                    "strict": True,
                    "schema": output_type.model_json_schema(),
                },
            }
        else:
            payload["response_format"] = {"type": "json_object"}
            payload["messages"] = [
                {
                    "role": "system",
                    "content": (
                        "只输出一个符合以下 JSON Schema 的完整 JSON 对象，不要 Markdown。"
                        "每个文本值必须保留完整语义，不得用单字或占位符填槽。"
                        "服务端会严格校验字段、枚举、数量和约束：\n"
                        + json.dumps(
                            output_type.model_json_schema(),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    ),
                },
                *payload["messages"],
            ]
        # Both transport modes still require the exact same Pydantic validation.
        # Prepare retains direct structured mode. Planner may opt into bounded
        # reasoning, but only its final content can cross this gateway boundary.
        payload["enable_thinking"] = request.thinking_budget_tokens is not None
        if request.thinking_budget_tokens is not None:
            payload["thinking_budget"] = request.thinking_budget_tokens
        payload["temperature"] = request.temperature_override or 0.0
        call_id, audit_base = await self._start_audit_call(
            request,
            operation=operation,
            request_payload=payload,
            output_schema=output_type.model_json_schema(),
            default_stage=output_type.__name__,
        )
        started = time.monotonic()
        response: httpx.Response | None = None
        body: dict[str, Any] | None = None
        content: str | None = None
        try:
            _check_cancellation(cancellation, operation)
            response = await _await_or_cancel(
                self._client.post(
                    self._endpoint,
                    headers=self._headers,
                    json=payload,
                    timeout=request.reasoning_timeout_seconds or self._config.timeout_seconds,
                ),
                cancellation,
                operation,
            )
            assert response is not None
            _raise_for_status(response, operation)
            _check_cancellation(cancellation, operation)
            body = _json_object(response, operation)
            if request.thinking_budget_tokens is not None:
                _require_reasoning_usage(body, operation)
            content = _assistant_content(body, operation)
            try:
                value = output_type.model_validate_json(
                    content,
                    context=validation_context,
                )
            except ValidationError as exc:
                raise _model_error(
                    ModelFailureCode.MALFORMED_RESPONSE,
                    operation,
                    validation_issues=_safe_validation_issues(exc),
                ) from None
            except ValueError:
                raise _model_error(
                    ModelFailureCode.MALFORMED_RESPONSE,
                    operation,
                    validation_issues=("json:invalid",),
                ) from None
            usage = _usage(body, operation)
            await self._record_audit(
                "llm_gateway_completed",
                call_id=call_id,
                operation=operation,
                payload={
                    **audit_base,
                    **_response_audit(response, raw_body=body, chunks=[]),
                    "response_body_raw": body,
                    "response_assistant_content_raw": content,
                    "parsed_output": value.model_dump(mode="json"),
                    "materialized_output": None,
                    "schema_validation_result": {
                        "status": "accepted",
                        "output_type": output_type.__name__,
                    },
                    "business_guard_result": {"status": "pending_or_not_applicable"},
                    "accepted_or_rejected": "gateway_accepted",
                    "finish_reason": _finish_reason(body),
                    "token_usage": usage.model_dump(mode="json") if usage else None,
                    "private_reasoning_redacted": private_reasoning_was_redacted(body),
                    "completed_at": _utc_now(),
                    "latency_ms": _elapsed_ms(started),
                },
            )
            return ModelStructuredResult(
                value=value,
                usage=usage,
                audit_call_id=call_id,
            )
        except asyncio.CancelledError:
            await self._record_transport_error(
                call_id,
                operation,
                audit_base,
                _model_error(ModelFailureCode.CANCELLED, operation),
                started,
                response,
                body,
                [],
            )
            raise
        except ModelGatewayError as error:
            bound = _with_audit_call_id(error, call_id)
            if bound.code is ModelFailureCode.AUDIT_UNAVAILABLE:
                raise
            await self._record_audit(
                "llm_gateway_failed",
                call_id=call_id,
                operation=operation,
                payload={
                    **audit_base,
                    **_response_audit(response, raw_body=body, chunks=[]),
                    "response_body_raw": body if body is not None else _response_text(response),
                    "response_assistant_content_raw": content,
                    "parsed_output": None,
                    "materialized_output": None,
                    "schema_validation_result": {
                        "status": "rejected",
                        "output_type": output_type.__name__,
                        "issues": list(bound.validation_issues),
                    },
                    "business_guard_result": {"status": "not_run"},
                    "accepted_or_rejected": "rejected",
                    "failure_stage": _failure_stage(bound),
                    "failure_code": bound.code.value,
                    "validation_issues": list(bound.validation_issues),
                    "finish_reason": _finish_reason(body),
                    "token_usage": _safe_usage_dump(body),
                    "private_reasoning_redacted": private_reasoning_was_redacted(body),
                    "completed_at": _utc_now(),
                    "latency_ms": _elapsed_ms(started),
                },
            )
            raise bound from error
        except httpx.TimeoutException:
            transport_error = _model_error(
                ModelFailureCode.TIMEOUT,
                operation,
                retryable=True,
                audit_call_id=call_id,
            )
            await self._record_transport_error(
                call_id, operation, audit_base, transport_error, started, response, [], []
            )
            raise transport_error from None
        except httpx.HTTPError:
            transport_error = _model_error(
                ModelFailureCode.UNAVAILABLE,
                operation,
                retryable=True,
                audit_call_id=call_id,
            )
            await self._record_transport_error(
                call_id, operation, audit_base, transport_error, started, response, [], []
            )
            raise transport_error from None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    @property
    def _endpoint(self) -> str:
        return f"{self._base_url}/chat/completions"

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _request_payload(self, request: ModelRequest, *, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._config.model_name,
            "messages": [message.model_dump(mode="json") for message in request.messages],
            "temperature": (
                request.temperature_override
                if request.temperature_override is not None
                else self._config.temperature
            ),
            "max_completion_tokens": request.max_output_tokens,
            "stream": stream,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
            # Public wording is composed from an already validated Agent decision
            # and grounded observations. A second reasoning pass only adds latency
            # and cost, so request Qwen's direct-answer mode explicitly.
            payload["enable_thinking"] = False
        return payload

    async def _start_audit_call(
        self,
        request: ModelRequest,
        *,
        operation: str,
        request_payload: Mapping[str, Any],
        output_schema: Mapping[str, Any] | None,
        default_stage: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        call_id = str(uuid4())
        metadata = request.audit
        stage = metadata.stage if metadata is not None else default_stage or operation
        linked = register_model_call(
            self._audit_recorder,
            call_id=call_id,
            stage=stage,
            model=self._config.model_name,
            repair=metadata.repair if metadata is not None else False,
            attempt=metadata.attempt if metadata is not None else None,
            parent_call_id=metadata.parent_llm_call_id if metadata is not None else None,
            repair_of_call_id=metadata.repair_of_call_id if metadata is not None else None,
            model_switch_from=metadata.model_switch_from if metadata is not None else None,
        )
        sections = _request_sections(request_payload)
        base = {
            **linked,
            "stage": stage,
            "node": metadata.node if metadata is not None else None,
            "provider": "qwen",
            "model": self._config.model_name,
            "endpoint_class": "openai_compatible_chat_completions",
            "prompt_version": self._config.prompt_version,
            "contract_version": metadata.contract_version if metadata is not None else None,
            "model_parameters": {
                "temperature": request_payload.get("temperature"),
                "max_completion_tokens": request_payload.get("max_completion_tokens"),
                "stream": request_payload.get("stream"),
                "structured_output_mode": request.structured_output_mode,
                "enable_thinking": request_payload.get("enable_thinking"),
                "thinking_budget": request_payload.get("thinking_budget"),
                "timeout_seconds": (
                    request.reasoning_timeout_seconds or self._config.timeout_seconds
                ),
            },
            "request_messages_full": request_payload.get("messages"),
            "request_state_projection_full": sections["state"],
            "request_observations_full": sections["observations"],
            "request_guard_feedback_full": sections["guard_feedback"],
            "request_tools_full": sections["tools"],
            "request_output_schema_full": output_schema,
            "request_body_canonical": _canonical_json(request_payload),
            "request_body_hash": canonical_audit_hash(request_payload),
            "request_headers_allowlist": {"content-type": "application/json"},
            "model_switch_reason": (metadata.model_switch_reason if metadata is not None else None),
            "private_reasoning_redacted": False,
        }
        try:
            await self._record_audit(
                "llm_call_started",
                call_id=call_id,
                operation=operation,
                payload={**base, "started_at": _utc_now()},
            )
        except asyncio.CancelledError:
            await self._record_transport_error(
                call_id,
                operation,
                base,
                _model_error(ModelFailureCode.CANCELLED, operation),
                time.monotonic(),
                None,
                None,
                [],
            )
            raise
        return call_id, base

    async def _record_transport_error(
        self,
        call_id: str,
        operation: str,
        audit_base: Mapping[str, Any],
        error: ModelGatewayError,
        started: float,
        response: httpx.Response | None,
        raw_body: Any,
        chunks: list[dict[str, Any]],
    ) -> None:
        await self._record_audit(
            "llm_gateway_failed",
            call_id=call_id,
            operation=operation,
            payload={
                **audit_base,
                **_response_audit(response, raw_body=raw_body, chunks=chunks),
                "accepted_or_rejected": "rejected",
                "failure_stage": _failure_stage(error),
                "failure_code": error.code.value,
                "validation_issues": list(error.validation_issues),
                "completed_at": _utc_now(),
                "latency_ms": _elapsed_ms(started),
            },
        )

    async def _record_audit(
        self,
        event_kind: str,
        *,
        call_id: str,
        operation: str,
        payload: Mapping[str, Any],
    ) -> None:
        if not self._audit_recorder.enabled:
            return
        try:
            await self._audit_recorder.record(
                event_kind,
                llm_call_id=call_id,
                payload=payload,
            )
        except ModelAuditError as error:
            raise ModelGatewayError(
                ModelFailureCode.AUDIT_UNAVAILABLE,
                operation,
                retryable=True,
                audit_call_id=call_id,
            ) from error


def _check_cancellation(cancellation: ModelCancellation | None, operation: str) -> None:
    if cancellation is not None:
        cancellation.raise_if_cancelled(operation)


AwaitedValue = TypeVar("AwaitedValue")


async def _await_or_cancel(
    awaitable: Awaitable[AwaitedValue],
    cancellation: ModelCancellation | None,
    operation: str,
) -> AwaitedValue:
    if cancellation is None:
        return await awaitable
    operation_task = asyncio.ensure_future(awaitable)
    cancellation_task = asyncio.create_task(cancellation.wait_cancelled())
    try:
        cancellation.raise_if_cancelled(operation)
        done, _ = await asyncio.wait(
            {operation_task, cancellation_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if cancellation_task in done:
            raise _model_error(ModelFailureCode.CANCELLED, operation)
        return await operation_task
    finally:
        # Parent task cancellation (disconnect/shutdown) must stop the HTTP
        # request too, not just the token-based cancellation waiter.
        for task in (operation_task, cancellation_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(operation_task, cancellation_task, return_exceptions=True)


def _raise_for_status(response: httpx.Response, operation: str) -> None:
    code = response.status_code
    if code < 400:
        return
    if code == 400:
        try:
            body = response.json()
        except ValueError:
            body = None
        error = body.get("error", body) if isinstance(body, dict) else None
        if isinstance(error, dict) and error.get("code") == "Arrearage":
            raise _model_error(ModelFailureCode.BILLING_UNAVAILABLE, operation)
    if code == 401:
        raise _model_error(ModelFailureCode.AUTHENTICATION_FAILED, operation)
    if code == 403:
        raise _model_error(ModelFailureCode.PERMISSION_DENIED, operation)
    if code == 429:
        raise _model_error(ModelFailureCode.RATE_LIMITED, operation, retryable=True)
    if code in {408, 504}:
        raise _model_error(ModelFailureCode.TIMEOUT, operation, retryable=True)
    if code >= 500:
        raise _model_error(ModelFailureCode.UNAVAILABLE, operation, retryable=True)
    raise _model_error(ModelFailureCode.UPSTREAM_ERROR, operation)


def _parse_stream_event(data: str, operation: str) -> ModelStreamChunk:
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        raise _model_error(ModelFailureCode.MALFORMED_RESPONSE, operation) from None
    if not isinstance(payload, dict):
        raise _model_error(ModelFailureCode.MALFORMED_RESPONSE, operation)
    body = cast(dict[str, Any], payload)
    usage = _usage(body, operation)
    choices = body.get("choices")
    if not isinstance(choices, list):
        raise _model_error(ModelFailureCode.MALFORMED_RESPONSE, operation)
    if not choices:
        if usage is None:
            raise _model_error(ModelFailureCode.MALFORMED_RESPONSE, operation)
        return ModelStreamChunk(usage=usage)
    choice = choices[0]
    if not isinstance(choice, dict):
        raise _model_error(ModelFailureCode.MALFORMED_RESPONSE, operation)
    delta = choice.get("delta")
    if not isinstance(delta, dict):
        raise _model_error(ModelFailureCode.MALFORMED_RESPONSE, operation)
    content = delta.get("content")
    if content is not None and not isinstance(content, str):
        raise _model_error(ModelFailureCode.MALFORMED_RESPONSE, operation)
    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise _model_error(ModelFailureCode.MALFORMED_RESPONSE, operation)
    return ModelStreamChunk(
        delta=content or "",
        finish_reason=finish_reason,
        usage=usage,
    )


def _json_object(response: httpx.Response, operation: str) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        raise _model_error(ModelFailureCode.MALFORMED_RESPONSE, operation) from None
    if not isinstance(payload, dict):
        raise _model_error(ModelFailureCode.MALFORMED_RESPONSE, operation)
    return cast(dict[str, Any], payload)


def _assistant_content(payload: Mapping[str, Any], operation: str) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise _model_error(ModelFailureCode.MALFORMED_RESPONSE, operation)
    if choices[0].get("finish_reason") == "length":
        raise _model_error(
            ModelFailureCode.MALFORMED_RESPONSE,
            operation,
            validation_issues=("completion:truncated",),
        )
    message = choices[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise _model_error(ModelFailureCode.MALFORMED_RESPONSE, operation)
    return cast(str, message["content"])


def _require_reasoning_usage(payload: Mapping[str, Any], operation: str) -> None:
    """Detect an upstream ignoring opt-in; do not retain or expose reasoning."""
    choices = payload.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else None
    message = choice.get("message") if isinstance(choice, dict) else None
    reasoning = message.get("reasoning_content") if isinstance(message, dict) else None
    usage = payload.get("usage")
    details = usage.get("completion_tokens_details") if isinstance(usage, dict) else None
    tokens = details.get("reasoning_tokens") if isinstance(details, dict) else None
    if (
        not isinstance(reasoning, str)
        or not reasoning.strip()
        or type(tokens) is not int
        or tokens <= 0
    ):
        raise _model_error(
            ModelFailureCode.MALFORMED_RESPONSE,
            operation,
            validation_issues=("reasoning:not_enabled",),
        )


def _usage(payload: Mapping[str, Any], operation: str) -> ModelUsage | None:
    raw = payload.get("usage")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise _model_error(ModelFailureCode.MALFORMED_RESPONSE, operation)
    try:
        return ModelUsage.model_validate(
            {
                "prompt_tokens": raw["prompt_tokens"],
                "completion_tokens": raw["completion_tokens"],
                "total_tokens": raw["total_tokens"],
            }
        )
    except KeyError:
        raise _model_error(ModelFailureCode.MALFORMED_RESPONSE, operation) from None
    except ValidationError:
        raise _model_error(ModelFailureCode.MALFORMED_RESPONSE, operation) from None


def _model_error(
    code: ModelFailureCode,
    operation: str,
    *,
    retryable: bool = False,
    validation_issues: tuple[str, ...] = (),
    audit_call_id: str | None = None,
) -> ModelGatewayError:
    return ModelGatewayError(
        code,
        operation,
        retryable=retryable,
        validation_issues=validation_issues,
        audit_call_id=audit_call_id,
    )


def _with_audit_call_id(error: ModelGatewayError, call_id: str) -> ModelGatewayError:
    if error.audit_call_id == call_id:
        return error
    return ModelGatewayError(
        error.code,
        error.operation,
        retryable=error.retryable,
        validation_issues=error.validation_issues,
        audit_call_id=call_id,
    )


def _failure_stage(error: ModelGatewayError) -> str:
    if error.code is ModelFailureCode.MALFORMED_RESPONSE:
        return "schema_or_response_parse"
    if error.code is ModelFailureCode.CANCELLED:
        return "cancellation"
    if error.code is ModelFailureCode.AUDIT_UNAVAILABLE:
        return "audit_persistence"
    return "transport"


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def _elapsed_ms(started: float) -> int:
    return max(0, round((time.monotonic() - started) * 1000))


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _response_text(response: httpx.Response | None) -> str | None:
    if response is None:
        return None
    try:
        return response.text
    except (httpx.ResponseNotRead, RuntimeError):
        return None


def _response_audit(
    response: httpx.Response | None,
    *,
    raw_body: Any,
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    if response is None:
        return {
            "provider_request_id": None,
            "response_transport_status": None,
            "response_headers_allowlist": {},
            "response_chunks_ordered": chunks,
            "response_body_raw": raw_body,
            "response_body_hash": None,
        }
    headers = {
        key.casefold(): value
        for key, value in response.headers.items()
        if key.casefold()
        in {
            "content-type",
            "date",
            "request-id",
            "x-dashscope-request-id",
            "x-request-id",
        }
    }
    provider_request_id = next(
        (
            headers[key]
            for key in ("x-dashscope-request-id", "x-request-id", "request-id")
            if key in headers
        ),
        None,
    )
    try:
        raw_bytes = response.content
    except httpx.ResponseNotRead:
        raw_bytes = b""
    if not raw_bytes:
        text = _response_text(response)
        raw_bytes = text.encode("utf-8") if text is not None else b""
    if not raw_bytes and raw_body not in (None, [], {}):
        raw_bytes = json.dumps(
            raw_body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    return {
        "provider_request_id": provider_request_id,
        "response_transport_status": response.status_code,
        "response_headers_allowlist": headers,
        "response_chunks_ordered": chunks,
        "response_body_raw": raw_body,
        "response_body_hash": hashlib.sha256(raw_bytes).hexdigest() if raw_bytes else None,
    }


def _safe_usage_dump(body: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if body is None:
        return None
    raw = body.get("usage")
    return dict(raw) if isinstance(raw, dict) else None


def _finish_reason(body: Mapping[str, Any] | None) -> str | None:
    if body is None:
        return None
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    value = choices[0].get("finish_reason")
    return value if isinstance(value, str) else None


def _request_sections(request_payload: Mapping[str, Any]) -> dict[str, Any]:
    decoded_messages: list[dict[str, Any]] = []
    messages = request_payload.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            try:
                decoded = json.loads(content)
            except json.JSONDecodeError:
                continue
            if isinstance(decoded, dict):
                decoded_messages.append(cast(dict[str, Any], decoded))
    state_keys = {
        "context",
        "existing_trip_basics",
        "grounding_context",
        "semantic_state",
        "semantic_state_excerpt",
        "state",
        "trip_context",
        "workspace",
    }
    observation_keys = {
        "capability_observations",
        "observations",
        "tool_observations",
    }
    guard_keys = {
        "guard_feedback",
        "repair_instruction",
        "repair_validation_code",
        "required_repair_contract",
        "validation_issue",
    }
    tool_keys = {"allowed_capabilities", "tools", "tool_requests"}
    return {
        "state": _matching_sections(decoded_messages, state_keys),
        "observations": _matching_sections(decoded_messages, observation_keys),
        "guard_feedback": _matching_sections(decoded_messages, guard_keys),
        "tools": {
            "provider_tools": request_payload.get("tools"),
            "prompt_declared_tools": _matching_sections(decoded_messages, tool_keys),
        },
    }


def _matching_sections(messages: list[dict[str, Any]], keys: set[str]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for message in messages:
        selected = {key: value for key, value in message.items() if key in keys}
        if selected:
            matches.append(selected)
    return matches


def _schema_name(output_type: type[StructuredValue]) -> str:
    """Return a provider-safe stable name for one structured output schema."""

    normalized = re.sub(r"[^A-Za-z0-9_-]", "_", output_type.__name__).strip("_-")
    return (normalized or "structured_output")[:64]


def _safe_validation_issues(error: ValidationError) -> tuple[str, ...]:
    """Keep schema locations and issue types without retaining model or user content."""

    issues: list[str] = []
    for item in error.errors(include_input=False, include_url=False):
        location = ".".join(str(part) for part in item["loc"]) or "root"
        message_fingerprint = hashlib.sha256(item["msg"].encode("utf-8")).hexdigest()[:12]
        issues.append(f"{location}:{item['type']}:{message_fingerprint}")
    return tuple(issues[:20])
