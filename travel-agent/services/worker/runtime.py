"""Lifecycle wrapper for background workers and their dependencies."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from backend.persistence import DependencyRegistry

WorkerLoop = Callable[[asyncio.Event], Awaitable[None]]


class WorkerDependencyUnavailable(RuntimeError):
    """Raised before job consumption when a critical dependency is unavailable."""


class WorkerRuntime:
    def __init__(self, dependencies: DependencyRegistry, worker_loop: WorkerLoop) -> None:
        self._dependencies = dependencies
        self._worker_loop = worker_loop

    async def run(self, stop_event: asyncio.Event) -> None:
        await self._dependencies.start()
        try:
            report = await self._dependencies.check()
            if not report.ready:
                unavailable = sorted(
                    item.name
                    for item in report.dependencies
                    if item.critical and not item.available
                )
                raise WorkerDependencyUnavailable(
                    "Critical worker dependencies unavailable: " + ", ".join(unavailable)
                )
            await self._worker_loop(stop_event)
        finally:
            await self._dependencies.close()
