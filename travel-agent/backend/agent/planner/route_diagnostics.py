"""Retain safe mode-specific failure evidence without leaking Provider payloads."""

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
    code = detail.code.value
    if detail.upstream_code is not None:
        code += f"/{detail.upstream_code}"
    return (
        f"{label}查询失败：{description}（{code}，已尝试{detail.attempts}次）；不能认定没有路线。"
    )
