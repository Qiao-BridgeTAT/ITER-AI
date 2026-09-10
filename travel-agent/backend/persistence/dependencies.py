"""Runtime dependency probes shared by the API and worker processes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from backend.config import Settings


@dataclass(frozen=True)
class DependencyStatus:
    name: str
    available: bool
    critical: bool = True
    reason: str | None = None


@dataclass(frozen=True)
class DependencyReport:
    dependencies: tuple[DependencyStatus, ...]

    @property
    def ready(self) -> bool:
        return all(item.available for item in self.dependencies if item.critical)


class DependencyProbe(Protocol):
    name: str
    critical: bool

    async def start(self) -> None: ...

    async def check(self) -> DependencyStatus: ...

    async def close(self) -> None: ...


class DependencyRegistry:
    """Owns dependency lifecycles without making startup imply readiness."""

    def __init__(self, probes: tuple[DependencyProbe, ...]) -> None:
        self._probes = probes

    async def start(self) -> None:
        for probe in self._probes:
            await probe.start()

    async def check(self) -> DependencyReport:
        statuses = tuple([await probe.check() for probe in self._probes])
        return DependencyReport(dependencies=statuses)

    async def close(self) -> None:
        for probe in reversed(self._probes):
            await probe.close()


class DatabaseProbe:
    name = "postgres"
    critical = True

    def __init__(self, database_url: str) -> None:
        self._engine: AsyncEngine = create_async_engine(database_url, pool_pre_ping=True)

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    async def start(self) -> None:
        return None

    async def check(self) -> DependencyStatus:
        try:
            async with self._engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
        except Exception as exc:  # readiness must classify driver/network failures alike
            return DependencyStatus(self.name, False, self.critical, type(exc).__name__)
        return DependencyStatus(self.name, True, self.critical)

    async def close(self) -> None:
        await self._engine.dispose()


class RedisProbe:
    name = "redis"
    critical = True

    def __init__(self, redis_url: str) -> None:
        self._client: Redis = Redis.from_url(redis_url, decode_responses=True)

    async def start(self) -> None:
        return None

    async def check(self) -> DependencyStatus:
        try:
            await self._client.ping()
        except Exception as exc:
            return DependencyStatus(self.name, False, self.critical, type(exc).__name__)
        return DependencyStatus(self.name, True, self.critical)

    async def close(self) -> None:
        await self._client.aclose()


class ObjectStorageProbe:
    name = "object_storage"
    critical = True

    def __init__(self, endpoint: str | None, local_path: Path) -> None:
        self._endpoint = endpoint.rstrip("/") if endpoint else None
        self._local_path = local_path
        self._client = httpx.AsyncClient(timeout=2.0) if self._endpoint else None

    async def start(self) -> None:
        return None

    async def check(self) -> DependencyStatus:
        try:
            if self._endpoint is not None:
                assert self._client is not None
                response = await self._client.get(f"{self._endpoint}/minio/health/ready")
                response.raise_for_status()
            else:
                self._local_path.mkdir(parents=True, exist_ok=True)
                marker = self._local_path / ".healthcheck"
                marker.touch(exist_ok=True)
                marker.unlink(missing_ok=True)
        except Exception as exc:
            return DependencyStatus(self.name, False, self.critical, type(exc).__name__)
        return DependencyStatus(self.name, True, self.critical)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()


def build_dependency_registry(settings: Settings) -> DependencyRegistry:
    values: Mapping[str, str] = settings.values
    endpoint = values.get("OBJECT_STORAGE_ENDPOINT")
    local_path = Path(values.get("OBJECT_STORAGE_LOCAL_PATH", ".local/artifacts"))
    return DependencyRegistry(
        (
            DatabaseProbe(values["DATABASE_URL"]),
            RedisProbe(values["REDIS_URL"]),
            ObjectStorageProbe(endpoint, local_path),
        )
    )
