"""Provider-local cache, concurrency, rate, retry, and circuit-breaker runtime."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from backend.contracts.enums import ProviderCode
from backend.persistence.redis_temporary import RateLimitDecision
from backend.providers.contracts import ProviderError, ProviderFailureCode
from backend.providers.interfaces import ProviderCache

ResultT = TypeVar("ResultT", bound=BaseModel)

# AMap and Baidu prohibit caching their service content unless a separate
# authorization says otherwise. Zero keeps the default runtime compliant; a
# deployment with explicit permission can inject a positive operation TTL.
POI_CACHE_TTL_SECONDS = 0
HOURS_CACHE_TTL_SECONDS = 0
ROUTE_CACHE_TTL_SECONDS = 0
WEATHER_CACHE_TTL_SECONDS = 30 * 60
HOTEL_PRICE_CACHE_TTL_SECONDS = 30 * 60
TICKET_PRICE_CACHE_TTL_SECONDS = 30 * 60

_CIRCUIT_FAILURE_CODES = frozenset(
    {
        ProviderFailureCode.TIMEOUT,
        ProviderFailureCode.UNAVAILABLE,
        ProviderFailureCode.MALFORMED_RESPONSE,
        ProviderFailureCode.UPSTREAM_ERROR,
    }
)
_LOGGER = logging.getLogger(__name__)


class ProviderRateLimiter(Protocol):
    async def consume_rate_limit(
        self,
        scope: str,
        subject: str,
        *,
        limit: int,
        window_seconds: int,
    ) -> RateLimitDecision: ...


@dataclass(frozen=True)
class ProviderRuntimePolicy:
    timeout_seconds: float
    max_concurrency: int
    rate_limit: int
    rate_window_seconds: int = 1
    retry_count: int = 1
    retry_delay_seconds: float = 0.1
    rate_wait_seconds: float = 0.0
    circuit_failure_threshold: int = 3
    circuit_recovery_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("provider timeout must be positive")
        if self.max_concurrency <= 0:
            raise ValueError("provider max_concurrency must be positive")
        if self.rate_limit <= 0 or self.rate_window_seconds <= 0:
            raise ValueError("provider rate limit and window must be positive")
        if self.retry_count not in {0, 1}:
            raise ValueError("provider retry_count must be zero or one")
        if self.retry_delay_seconds < 0:
            raise ValueError("provider retry delay cannot be negative")
        if self.rate_wait_seconds < 0:
            raise ValueError("provider rate wait budget cannot be negative")
        if self.circuit_failure_threshold <= 0 or self.circuit_recovery_seconds <= 0:
            raise ValueError("provider circuit settings must be positive")


DEFAULT_PROVIDER_POLICIES: Mapping[ProviderCode, ProviderRuntimePolicy] = {
    ProviderCode.AMAP: ProviderRuntimePolicy(
        timeout_seconds=5,
        max_concurrency=8,
        rate_limit=40,
    ),
    ProviderCode.BAIDU: ProviderRuntimePolicy(
        timeout_seconds=5,
        max_concurrency=4,
        rate_limit=20,
    ),
    ProviderCode.FLYAI: ProviderRuntimePolicy(
        timeout_seconds=12,
        max_concurrency=2,
        rate_limit=5,
    ),
    ProviderCode.WEATHER: ProviderRuntimePolicy(
        timeout_seconds=5,
        max_concurrency=4,
        rate_limit=10,
    ),
}


class ProviderRuntime:
    """Apply one independent stability budget around a normalized Provider call."""

    def __init__(
        self,
        provider: ProviderCode,
        policy: ProviderRuntimePolicy,
        *,
        cache: ProviderCache | None = None,
        rate_limiter: ProviderRateLimiter | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
        operation_timeout_seconds: Mapping[str, float] | None = None,
    ) -> None:
        self.provider = provider
        self.policy = policy
        self._cache = cache
        self._rate_limiter = rate_limiter
        self._clock = clock
        self._sleeper = sleeper
        self.operation_timeout_seconds = dict(operation_timeout_seconds or {})
        if any(value <= 0 for value in self.operation_timeout_seconds.values()):
            raise ValueError("operation timeouts must be positive")
        self._semaphore = asyncio.Semaphore(policy.max_concurrency)
        self._circuit_lock = asyncio.Lock()
        self._consecutive_failures = 0
        self._open_until: float | None = None
        self._probe_in_flight = False

    async def execute(
        self,
        *,
        operation: str,
        semantic_version: str,
        parameters: Mapping[str, Any],
        cache_ttl_seconds: int,
        decode: Callable[[Any], ResultT],
        call: Callable[[], Awaitable[ResultT]],
    ) -> ResultT:
        if cache_ttl_seconds < 0:
            raise ValueError("provider cache TTL cannot be negative")
        cache_parameters = {"operation": operation, **parameters}
        cache_enabled = self._cache is not None and cache_ttl_seconds > 0
        cached = (
            await self._read_cache(semantic_version, cache_parameters, decode)
            if cache_enabled
            else None
        )
        if cached is not None:
            return cached

        is_probe = await self._begin_call(operation)
        try:
            async with self._semaphore:
                result = await self._attempt(operation, call)
        except ProviderError as error:
            if error.code in _CIRCUIT_FAILURE_CODES:
                await self._record_failure(is_probe)
            else:
                await self._release_probe()
            raise
        await self._record_success()
        if cache_enabled:
            await self._write_cache(
                operation,
                semantic_version,
                cache_parameters,
                result,
                cache_ttl_seconds,
            )
        return result

    async def _attempt(
        self,
        operation: str,
        call: Callable[[], Awaitable[ResultT]],
    ) -> ResultT:
        for attempt in range(self.policy.retry_count + 1):
            await self._consume_rate_limit(operation)
            try:
                async with asyncio.timeout(
                    self.operation_timeout_seconds.get(operation, self.policy.timeout_seconds)
                ):
                    return await call()
            except TimeoutError:
                error = ProviderError(
                    self.provider,
                    ProviderFailureCode.TIMEOUT,
                    operation,
                    retryable=True,
                )
            except ProviderError as caught:
                error = caught
            error.attempts = attempt + 1
            if not error.retryable or attempt >= self.policy.retry_count:
                raise error
            if self.policy.retry_delay_seconds:
                await self._sleeper(self.policy.retry_delay_seconds)
        raise AssertionError("provider retry loop exhausted without a result")

    async def _consume_rate_limit(self, operation: str) -> None:
        if self._rate_limiter is None:
            return
        remaining = self.policy.rate_wait_seconds
        while True:
            decision = await self._rate_limiter.consume_rate_limit(
                "provider-outbound",
                self.provider.value,
                limit=self.policy.rate_limit,
                window_seconds=self.policy.rate_window_seconds,
            )
            if decision.allowed:
                return
            delay = max(0.05, float(decision.retry_after_seconds))
            if delay > remaining:
                raise ProviderError(
                    self.provider,
                    ProviderFailureCode.RATE_LIMITED,
                    operation,
                    retryable=True,
                )
            # A bounded asynchronous queue, not a new HTTP retry or unbounded sleep.
            await self._sleeper(delay)
            remaining -= delay

    async def _read_cache(
        self,
        semantic_version: str,
        parameters: Mapping[str, Any],
        decode: Callable[[Any], ResultT],
    ) -> ResultT | None:
        if self._cache is None:
            return None
        try:
            cached = await self._cache.get_cache(
                self.provider.value,
                semantic_version,
                parameters,
            )
        except Exception:
            _LOGGER.warning(
                "provider cache read failed; continuing without cache",
                extra={"provider": self.provider.value},
            )
            return None
        if cached is None:
            return None
        try:
            return decode(cached)
        except (TypeError, ValueError, ValidationError):
            return None

    async def _write_cache(
        self,
        operation: str,
        semantic_version: str,
        parameters: Mapping[str, Any],
        result: ResultT,
        cache_ttl_seconds: int,
    ) -> None:
        if self._cache is None:
            return
        try:
            await self._cache.put_cache(
                self.provider.value,
                semantic_version,
                parameters,
                result.model_dump(mode="json"),
                cache_ttl_seconds,
            )
        except Exception:
            _LOGGER.warning(
                "provider cache write failed; returning live result",
                extra={"provider": self.provider.value, "operation": operation},
            )

    async def _begin_call(self, operation: str) -> bool:
        async with self._circuit_lock:
            if self._open_until is None:
                return False
            if self._clock() < self._open_until or self._probe_in_flight:
                raise ProviderError(
                    self.provider,
                    ProviderFailureCode.UNAVAILABLE,
                    operation,
                    retryable=True,
                )
            self._probe_in_flight = True
            return True

    async def _record_success(self) -> None:
        async with self._circuit_lock:
            self._consecutive_failures = 0
            self._open_until = None
            self._probe_in_flight = False

    async def _release_probe(self) -> None:
        async with self._circuit_lock:
            self._probe_in_flight = False

    async def _record_failure(self, is_probe: bool) -> None:
        async with self._circuit_lock:
            self._probe_in_flight = False
            self._consecutive_failures += 1
            if is_probe or self._consecutive_failures >= self.policy.circuit_failure_threshold:
                self._open_until = self._clock() + self.policy.circuit_recovery_seconds
