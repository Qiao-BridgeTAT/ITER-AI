"""Retain safe mode-specific failure evidence without leaking Provider payloads."""

from datetime import datetime, timedelta

from backend.contracts.v4.planner_observations import RouteQueryFailure, SpatialRouteEdge
from backend.providers.contracts import (
    ProviderError,
    ProviderFailureDetail,
    ProviderResponse,
    ProviderRoute,
    RouteMode,
)


def missing_route_reason(
    mode: RouteMode,
    response: ProviderResponse[ProviderRoute] | None,
    error: ProviderError | None,
) -> str:
    label = {
        RouteMode.DRIVING: "驾车",
        RouteMode.TRANSIT: "公共交通",
        RouteMode.WALKING: "步行",
        RouteMode.CYCLING: "骑行",
    }[mode]
    detail = response.failures.get(mode.value) if response is not None else None
    if detail is None and error is not None:
        detail = error.failures.get(mode.value)
        # An aggregate error may include failed modes beside genuinely empty ones.
        if detail is None and not error.failures:
            detail = ProviderFailureDetail.from_error(error)
    if detail is None:
        return f"{label}查询完成但未返回路线（empty）；不代表其他交通方式不可用。"
    description = {
        "timeout": "请求超时",
        "rate_limited": "请求限流",
        "authentication_failed": "凭证校验失败",
        "permission_denied": "接口权限不足",
        "invalid_request": "请求参数不被接受",
        "malformed_response": "响应格式无法解析",
        "unavailable": "服务暂不可用",
        "upstream_error": "上游服务异常",
    }[detail.code.value]
    if detail.upstream_code in {"10003", "10044", "10045"}:
        description = "日调用额度已用完，本轮不重复请求"
    if detail.upstream_code == "10045":
        description = "月调用额度已用完，本轮不重复请求"
    if detail.upstream_code in {"10004", "10014", "10021", "10022", "10023"}:
        description = "请求频率超限"
    code = detail.code.value
    if detail.upstream_code is not None:
        code += f"/{detail.upstream_code}"
    return (
        f"{label}查询失败：{description}（{code}，已尝试{detail.attempts}次）；不能认定没有路线。"
    )


def route_query_failure(
    mode: RouteMode,
    response: ProviderResponse[ProviderRoute] | None,
    error: ProviderError | None,
    previous: SpatialRouteEdge | None,
    now: datetime,
    *,
    skipped: bool = False,
) -> RouteQueryFailure:
    detail = response.failures.get(mode.value) if response is not None else None
    if detail is None and error is not None:
        detail = error.failures.get(mode.value)
        if detail is None and not error.failures:
            detail = ProviderFailureDetail.from_error(error)
    previous_count = (
        previous.query_failure.query_count
        if previous and previous.query_failure
        else 1
        if previous and previous.status == "missing"
        else 0
    )
    count = previous_count + 1
    retryable = bool(detail and detail.retryable and count < 2 and not skipped)
    return RouteQueryFailure(
        code=detail.code.value if detail else "budget_exhausted" if skipped else "empty",
        upstream_code=detail.upstream_code if detail else None,
        attempts=detail.attempts if detail else 1,
        query_count=count,
        retryable=retryable,
        retry_after=now + timedelta(seconds=30) if retryable else None,
    )


def route_needs_retry(edge: SpatialRouteEdge, now: datetime) -> bool:
    if edge.status != "missing":
        return False
    if failure := edge.query_failure:
        return (
            failure.retryable
            and failure.query_count < 2
            and failure.retry_after is not None
            and now >= failure.retry_after
        )
    # Legacy checkpoints get one bounded migration retry for transient failures.
    reason = edge.missing_reason or ""
    if any(code in reason for code in ("10003", "10010", "10044", "10045")):
        return False
    return any(
        code in reason for code in ("rate_limited", "timeout", "unavailable", "upstream_error")
    )
