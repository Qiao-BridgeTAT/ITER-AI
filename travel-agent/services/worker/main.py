"""Independent RQ worker entrypoint."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import cast

from redis import Redis
from rq import Queue, Worker

from backend.config import Settings
from backend.persistence import build_dependency_registry
from services.worker.runtime import WorkerDependencyUnavailable

WorkerFactory = Callable[[Redis, list[Queue]], Worker]


def _worker_factory(connection: Redis, queues: list[Queue]) -> Worker:
    return Worker(queues, connection=connection)


async def verify_worker_dependencies(settings: Settings) -> None:
    dependencies = build_dependency_registry(settings)
    await dependencies.start()
    try:
        report = await dependencies.check()
        if not report.ready:
            unavailable = sorted(
                item.name for item in report.dependencies if item.critical and not item.available
            )
            raise WorkerDependencyUnavailable(
                "Critical worker dependencies unavailable: " + ", ".join(unavailable)
            )
    finally:
        await dependencies.close()


def run_worker(settings: Settings, factory: WorkerFactory = _worker_factory) -> None:
    asyncio.run(verify_worker_dependencies(settings))
    connection = Redis.from_url(settings.values["REDIS_URL"])
    try:
        queue = Queue("travel-agent", connection=connection)
        work = cast(Callable[[], bool], factory(connection, [queue]).work)
        work()
    finally:
        connection.close()


def main() -> None:
    settings = Settings.from_environment()
    run_worker(settings)


if __name__ == "__main__":
    main()
