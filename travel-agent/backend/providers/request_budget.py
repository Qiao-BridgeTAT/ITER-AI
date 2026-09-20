"""Optional per-run budget applied at external I/O boundaries, including retries."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from contextvars import ContextVar
from datetime import UTC, datetime
from functools import wraps
from typing import Any, ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")


class RequestBudgetExceeded(RuntimeError):
    pass


class RequestBudget:
    def __init__(
        self,
        deadline: datetime | None,
        *,
        used: int = 0,
        persist: Callable[[int], Awaitable[None]] | None = None,
    ) -> None:
        self.deadline = deadline
        self.used = used
        self.persist = persist
        self.semaphore = asyncio.Semaphore(4)
        self.lock = asyncio.Lock()

    def remaining(self) -> float:
        if self.deadline is None:
            return float("inf")
        return max(0, (self.deadline - datetime.now(UTC)).total_seconds())

    async def reserve(self) -> None:
        async with self.lock:
            if self.used >= 80 or self.remaining() <= 20:
                raise RequestBudgetExceeded("planner_external_budget_exhausted")
            self.used += 1
            if self.persist:
                await self.persist(self.used)


active_request_budget: ContextVar[RequestBudget | None] = ContextVar(
    "planner_request_budget", default=None
)


def budgeted_external_request(fn: Callable[P, Awaitable[T]]) -> Callable[P, Coroutine[Any, Any, T]]:
    @wraps(fn)
    async def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        budget = active_request_budget.get()
        if budget is None:
            return await fn(*args, **kwargs)
        async with budget.semaphore:
            await budget.reserve()
            async with asyncio.timeout(
                max(0.01, budget.remaining() - 20) if budget.deadline is not None else None
            ):
                return await fn(*args, **kwargs)

    return wrapped
