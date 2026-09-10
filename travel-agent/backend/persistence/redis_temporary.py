"""Redis key spaces and atomic temporary-state primitives for M0-04."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal
from uuid import UUID

LOCK_RELEASE_SCRIPT = """
-- travel-agent:release-lock:v1
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""

LOCK_RENEW_SCRIPT = """
-- travel-agent:renew-lock:v1
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('EXPIRE', KEYS[1], ARGV[2])
end
return 0
"""

RATE_LIMIT_SCRIPT = """
-- travel-agent:rate-limit:v1
local current = redis.call('INCR', KEYS[1])
if current == 1 then
  redis.call('EXPIRE', KEYS[1], ARGV[1])
end
local ttl = redis.call('TTL', KEYS[1])
return {current, ttl}
"""

ANONYMOUS_INDEX_SCRIPT = """
-- travel-agent:anonymous-index:v1
redis.call('SADD', KEYS[1], ARGV[1])
local current_ttl = redis.call('TTL', KEYS[1])
local requested_ttl = tonumber(ARGV[2])
if current_ttl < requested_ttl then
  redis.call('EXPIRE', KEYS[1], requested_ttl)
end
return 1
"""

ANONYMOUS_SESSION_RENEW_SCRIPT = """
-- travel-agent:anonymous-session-renew:v1
local current = redis.call('GET', KEYS[1])
if not current then return nil end
local decoded = cjson.decode(current)
if decoded.session_id ~= ARGV[1] then return nil end
if KEYS[3] ~= KEYS[1] and redis.call('EXISTS', KEYS[3]) == 0 then return nil end
decoded.expires_at = ARGV[2]
local ttl = tonumber(ARGV[3])
local value = cjson.encode(decoded)
redis.call('SET', KEYS[1], value, 'EX', ttl)
for _, key in ipairs(redis.call('SMEMBERS', KEYS[2])) do
  -- Refresh only session-owned data, never execution leases or generation locks.
  if string.sub(key, 1, string.len(ARGV[4])) == ARGV[4] then
    redis.call('EXPIRE', key, ttl)
  end
end
redis.call('SADD', KEYS[2], KEYS[1])
redis.call('EXPIRE', KEYS[2], ttl)
return value
"""

ANONYMOUS_MESSAGE_APPEND_SCRIPT = """
-- travel-agent:anonymous-message-append:v1
local requested_ttl = tonumber(ARGV[2])
redis.call('RPUSH', KEYS[1], ARGV[1])
redis.call('EXPIRE', KEYS[1], requested_ttl)
redis.call('SADD', KEYS[2], KEYS[1])
local current_index_ttl = redis.call('TTL', KEYS[2])
if current_index_ttl < requested_ttl then
  redis.call('EXPIRE', KEYS[2], requested_ttl)
end
return redis.call('LLEN', KEYS[1])
"""

ANONYMOUS_ARTIFACT_SAVE_SCRIPT = """
-- travel-agent:anonymous-artifact-save:v1
local requested_ttl = tonumber(ARGV[2])
redis.call('SET', KEYS[1], ARGV[1], 'EX', requested_ttl)
redis.call('SADD', KEYS[2], KEYS[1])
redis.call('EXPIRE', KEYS[2], requested_ttl)
redis.call('SADD', KEYS[3], KEYS[1], KEYS[2])
local current_index_ttl = redis.call('TTL', KEYS[3])
if current_index_ttl < requested_ttl then
  redis.call('EXPIRE', KEYS[3], requested_ttl)
end
return 1
"""

ANONYMOUS_STATE_COMPARE_AND_SET_SCRIPT = """
-- travel-agent:anonymous-state-cas:v1
local current = redis.call('GET', KEYS[1])
if not current then
  return -1
end
local decoded = cjson.decode(current)
if tonumber(decoded.state_version) ~= tonumber(ARGV[1]) then
  return 0
end
local requested_ttl = tonumber(ARGV[3])
redis.call('SET', KEYS[1], ARGV[2], 'EX', requested_ttl)
redis.call('SADD', KEYS[2], KEYS[1])
local current_index_ttl = redis.call('TTL', KEYS[2])
if current_index_ttl < requested_ttl then
  redis.call('EXPIRE', KEYS[2], requested_ttl)
end
return 1
"""

ACTIVE_GENERATION_REPLACE_SCRIPT = """
-- travel-agent:active-generation-replace:v1
local previous = redis.call('GET', KEYS[1])
local requested_ttl = tonumber(ARGV[2])
redis.call('SET', KEYS[1], ARGV[1], 'EX', requested_ttl)
redis.call('DEL', KEYS[2])
if ARGV[3] == '1' then
  redis.call('SADD', KEYS[3], KEYS[1], KEYS[2])
  local current_index_ttl = redis.call('TTL', KEYS[3])
  if current_index_ttl < requested_ttl then
    redis.call('EXPIRE', KEYS[3], requested_ttl)
  end
end
return previous or ''
"""

ACTIVE_GENERATION_CLEAR_SCRIPT = """
-- travel-agent:active-generation-clear:v1
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""

PLANNER_GENERATION_FINISH_SCRIPT = """
-- travel-agent:planner-generation-finish:v1
if redis.call('GET', KEYS[1]) ~= ARGV[1] then
  return 0
end
if redis.call('GET', KEYS[2]) == ARGV[2] then
  redis.call('DEL', KEYS[2])
end
if ARGV[3] == '1' then
  redis.call('DEL', KEYS[1])
end
return 1
"""

GENERATION_SEQUENCE_SCRIPT = """
-- travel-agent:generation-sequence:v1
local sequence = redis.call('INCR', KEYS[1])
redis.call('EXPIRE', KEYS[1], ARGV[1])
return sequence
"""

SMS_CHALLENGE_CONSUME_SCRIPT = """
-- travel-agent:sms-challenge-consume:v1
local current = redis.call('GET', KEYS[1])
if not current then
  return -1
end
if current ~= ARGV[1] then
  return 0
end
redis.call('DEL', KEYS[1])
return 1
"""

AUTH_SESSION_SAVE_SCRIPT = """
-- travel-agent:auth-session-save:v1
local requested_ttl = tonumber(ARGV[2])
redis.call('SET', KEYS[1], ARGV[1], 'EX', requested_ttl)
redis.call('SADD', KEYS[2], KEYS[1])
local current_ttl = redis.call('TTL', KEYS[2])
if current_ttl < requested_ttl then
  redis.call('EXPIRE', KEYS[2], requested_ttl)
end
return 1
"""


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True, default=str)


def _json_load(value: str | bytes | None) -> Any | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(value)


def _positive_ttl(ttl_seconds: int) -> int:
    if ttl_seconds <= 0:
        raise ValueError("TTL must be greater than zero")
    return ttl_seconds


@dataclass(frozen=True)
class RedisKeySpace:
    prefix: str = "travel-agent:v1"

    def anonymous_index(self, session_id: str) -> str:
        return f"{self.prefix}:anonymous:{{{_digest(session_id)}}}:keys"

    def anonymous_session(self, session_id: str) -> str:
        return f"{self.prefix}:anonymous:{{{_digest(session_id)}}}:session"

    def anonymous_state(self, session_id: str, trip_id: UUID | str) -> str:
        return (
            f"{self.prefix}:anonymous:{{{_digest(session_id)}}}:trip:{_digest(str(trip_id))}:state"
        )

    def anonymous_messages(self, session_id: str, trip_id: UUID | str) -> str:
        return (
            f"{self.prefix}:anonymous:{{{_digest(session_id)}}}:"
            f"trip:{_digest(str(trip_id))}:messages"
        )

    def anonymous_artifact(self, session_id: str, artifact_id: UUID | str) -> str:
        return (
            f"{self.prefix}:anonymous:{{{_digest(session_id)}}}:"
            f"artifact:{_digest(str(artifact_id))}"
        )

    def anonymous_trip_artifacts(self, session_id: str, trip_id: UUID | str) -> str:
        return (
            f"{self.prefix}:anonymous:{{{_digest(session_id)}}}:"
            f"trip:{_digest(str(trip_id))}:artifacts"
        )

    def checkpoint(
        self,
        owner_type: Literal["anonymous", "user"],
        owner_id: str,
        trip_id: UUID | str,
    ) -> str:
        return (
            f"{self.prefix}:checkpoint:{owner_type}:{_digest(owner_id)}:"
            f"trip:{_digest(str(trip_id))}"
        )

    def cache(self, provider: str, semantic_version: str, parameters: Mapping[str, Any]) -> str:
        fingerprint = _digest(_json_dump(parameters))
        return f"{self.prefix}:cache:{_digest(provider)}:{_digest(semantic_version)}:{fingerprint}"

    def idempotency(self, scope: str, owner_id: str, idempotency_key: str) -> str:
        return (
            f"{self.prefix}:idempotency:{_digest(scope)}:{_digest(owner_id)}:"
            f"{_digest(idempotency_key)}"
        )

    def trip_lock(self, trip_id: UUID | str) -> str:
        return f"{self.prefix}:lock:trip:{_digest(str(trip_id))}"

    def rate_limit(self, scope: str, subject: str, window_number: int) -> str:
        return f"{self.prefix}:rate:{_digest(scope)}:{_digest(subject)}:window:{window_number}"

    def sms_challenge(self, phone_hash: str) -> str:
        return f"{self.prefix}:auth:sms:{_digest(phone_hash)}"

    def auth_session(self, session_token: str) -> str:
        return f"{self.prefix}:auth:session:{_digest(session_token)}"

    def user_auth_sessions(self, user_id: UUID | str) -> str:
        return f"{self.prefix}:auth:user:{_digest(str(user_id))}:sessions"

    def active_generation(self, trip_id: UUID | str) -> str:
        return f"{self.prefix}:generation:trip:{_digest(str(trip_id))}:active"

    def generation_sequence(self, generation_id: UUID | str) -> str:
        return f"{self.prefix}:generation:{_digest(str(generation_id))}:sequence"

    def unused_anonymous_index(self, trip_id: UUID | str) -> str:
        return f"{self.prefix}:generation:trip:{_digest(str(trip_id))}:unused-index"


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    count: int
    limit: int
    remaining: int
    retry_after_seconds: int


class RedisTemporaryStore:
    """Small typed facade over redis-py; durable business truth never lives here."""

    def __init__(
        self,
        client: Any,
        *,
        key_space: RedisKeySpace | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._client = client
        self.keys = key_space or RedisKeySpace()
        self._clock = clock

    async def save_anonymous_session(
        self,
        session_id: str,
        session: Mapping[str, Any],
        ttl_seconds: int,
    ) -> None:
        ttl = _positive_ttl(ttl_seconds)
        key = self.keys.anonymous_session(session_id)
        await self._client.set(key, _json_dump(session), ex=ttl)
        await self._register_anonymous_key(session_id, key, ttl)

    async def load_anonymous_session(self, session_id: str) -> dict[str, Any] | None:
        value = _json_load(await self._client.get(self.keys.anonymous_session(session_id)))
        return value if isinstance(value, dict) else None

    async def renew_anonymous_session(
        self,
        session_id: str,
        *,
        expires_at: str,
        ttl_seconds: int,
        trip_id: UUID | None = None,
    ) -> dict[str, Any] | None:
        """Atomically renew an existing credential; never resurrect expired ownership."""
        session_key = self.keys.anonymous_session(session_id)
        value = _json_load(
            await self._client.eval(
                ANONYMOUS_SESSION_RENEW_SCRIPT,
                3,
                session_key,
                self.keys.anonymous_index(session_id),
                self.keys.anonymous_state(session_id, trip_id) if trip_id else session_key,
                session_id,
                expires_at,
                _positive_ttl(ttl_seconds),
                session_key.removesuffix("session"),
            )
        )
        return value if isinstance(value, dict) else None

    async def save_anonymous_state(
        self,
        session_id: str,
        trip_id: UUID | str,
        state: Mapping[str, Any],
        ttl_seconds: int,
    ) -> None:
        ttl = _positive_ttl(ttl_seconds)
        key = self.keys.anonymous_state(session_id, trip_id)
        await self._client.set(key, _json_dump(state), ex=ttl)
        await self._register_anonymous_key(session_id, key, ttl)

    async def load_anonymous_state(
        self, session_id: str, trip_id: UUID | str
    ) -> dict[str, Any] | None:
        value = _json_load(await self._client.get(self.keys.anonymous_state(session_id, trip_id)))
        return value if isinstance(value, dict) else None

    async def compare_and_set_anonymous_state(
        self,
        session_id: str,
        trip_id: UUID | str,
        *,
        expected_state_version: int,
        state: Mapping[str, Any],
        ttl_seconds: int,
    ) -> Literal["updated", "conflict", "missing"]:
        ttl = _positive_ttl(ttl_seconds)
        result = int(
            await self._client.eval(
                ANONYMOUS_STATE_COMPARE_AND_SET_SCRIPT,
                2,
                self.keys.anonymous_state(session_id, trip_id),
                self.keys.anonymous_index(session_id),
                expected_state_version,
                _json_dump(state),
                ttl,
            )
        )
        return "updated" if result == 1 else "missing" if result == -1 else "conflict"

    async def append_anonymous_message(
        self,
        session_id: str,
        trip_id: UUID | str,
        message: Mapping[str, Any],
        ttl_seconds: int,
    ) -> None:
        ttl = _positive_ttl(ttl_seconds)
        key = self.keys.anonymous_messages(session_id, trip_id)
        await self._client.eval(
            ANONYMOUS_MESSAGE_APPEND_SCRIPT,
            2,
            key,
            self.keys.anonymous_index(session_id),
            _json_dump(message),
            ttl,
        )

    async def load_anonymous_messages(
        self, session_id: str, trip_id: UUID | str
    ) -> list[dict[str, Any]]:
        values = await self._client.lrange(self.keys.anonymous_messages(session_id, trip_id), 0, -1)
        return [decoded for item in values if isinstance((decoded := _json_load(item)), dict)]

    async def delete_anonymous_trip(self, session_id: str, trip_id: UUID | str) -> int:
        artifact_index_key = self.keys.anonymous_trip_artifacts(session_id, trip_id)
        artifact_members = await self._client.smembers(artifact_index_key)
        artifact_keys = [
            item.decode("utf-8") if isinstance(item, bytes) else item for item in artifact_members
        ]
        keys = [
            self.keys.anonymous_state(session_id, trip_id),
            self.keys.anonymous_messages(session_id, trip_id),
            self.keys.checkpoint("anonymous", session_id, trip_id),
            *artifact_keys,
            artifact_index_key,
        ]
        removed = await self._client.delete(*keys)
        index_key = self.keys.anonymous_index(session_id)
        await self._client.srem(index_key, *keys)
        return int(removed)

    async def save_anonymous_artifact(
        self,
        session_id: str,
        trip_id: UUID | str,
        artifact_id: UUID | str,
        artifact: Mapping[str, Any],
        ttl_seconds: int,
    ) -> None:
        ttl = _positive_ttl(ttl_seconds)
        key = self.keys.anonymous_artifact(session_id, artifact_id)
        await self._client.eval(
            ANONYMOUS_ARTIFACT_SAVE_SCRIPT,
            3,
            key,
            self.keys.anonymous_trip_artifacts(session_id, trip_id),
            self.keys.anonymous_index(session_id),
            _json_dump(artifact),
            ttl,
        )

    async def load_anonymous_artifact(
        self, session_id: str, artifact_id: UUID | str
    ) -> dict[str, Any] | None:
        value = _json_load(
            await self._client.get(self.keys.anonymous_artifact(session_id, artifact_id))
        )
        return value if isinstance(value, dict) else None

    async def save_checkpoint(
        self,
        owner_type: Literal["anonymous", "user"],
        owner_id: str,
        trip_id: UUID | str,
        checkpoint: Mapping[str, Any],
        ttl_seconds: int,
    ) -> None:
        ttl = _positive_ttl(ttl_seconds)
        key = self.keys.checkpoint(owner_type, owner_id, trip_id)
        await self._client.set(key, _json_dump(checkpoint), ex=ttl)
        if owner_type == "anonymous":
            await self._register_anonymous_key(owner_id, key, ttl)

    async def load_checkpoint(
        self,
        owner_type: Literal["anonymous", "user"],
        owner_id: str,
        trip_id: UUID | str,
    ) -> dict[str, Any] | None:
        value = _json_load(
            await self._client.get(self.keys.checkpoint(owner_type, owner_id, trip_id))
        )
        return value if isinstance(value, dict) else None

    async def put_cache(
        self,
        provider: str,
        semantic_version: str,
        parameters: Mapping[str, Any],
        value: Any,
        ttl_seconds: int,
    ) -> str:
        ttl = _positive_ttl(ttl_seconds)
        key = self.keys.cache(provider, semantic_version, parameters)
        await self._client.set(key, _json_dump(value), ex=ttl)
        return key

    async def get_cache(
        self, provider: str, semantic_version: str, parameters: Mapping[str, Any]
    ) -> Any | None:
        return _json_load(
            await self._client.get(self.keys.cache(provider, semantic_version, parameters))
        )

    async def delete_cache(
        self, provider: str, semantic_version: str, parameters: Mapping[str, Any]
    ) -> bool:
        return bool(
            await self._client.delete(self.keys.cache(provider, semantic_version, parameters))
        )

    async def claim_idempotency(
        self,
        scope: str,
        owner_id: str,
        idempotency_key: str,
        result: Mapping[str, Any],
        ttl_seconds: int,
    ) -> bool:
        ttl = _positive_ttl(ttl_seconds)
        key = self.keys.idempotency(scope, owner_id, idempotency_key)
        return bool(await self._client.set(key, _json_dump(result), ex=ttl, nx=True))

    async def get_idempotency(
        self, scope: str, owner_id: str, idempotency_key: str
    ) -> dict[str, Any] | None:
        value = _json_load(
            await self._client.get(self.keys.idempotency(scope, owner_id, idempotency_key))
        )
        return value if isinstance(value, dict) else None

    async def save_idempotency_result(
        self,
        scope: str,
        owner_id: str,
        idempotency_key: str,
        result: Mapping[str, Any],
        ttl_seconds: int,
    ) -> bool:
        ttl = _positive_ttl(ttl_seconds)
        key = self.keys.idempotency(scope, owner_id, idempotency_key)
        return bool(await self._client.set(key, _json_dump(result), ex=ttl, xx=True))

    async def clear_idempotency(self, scope: str, owner_id: str, idempotency_key: str) -> None:
        await self._client.delete(self.keys.idempotency(scope, owner_id, idempotency_key))

    async def acquire_trip_lock(self, trip_id: UUID | str, token: str, ttl_seconds: int) -> bool:
        ttl = _positive_ttl(ttl_seconds)
        return bool(await self._client.set(self.keys.trip_lock(trip_id), token, ex=ttl, nx=True))

    async def has_trip_lock(self, trip_id: UUID | str) -> bool:
        """An active-generation marker alone does not prove a live worker after a crash."""
        return await self._client.get(self.keys.trip_lock(trip_id)) is not None

    async def release_trip_lock(self, trip_id: UUID | str, token: str) -> bool:
        released = await self._client.eval(
            LOCK_RELEASE_SCRIPT, 1, self.keys.trip_lock(trip_id), token
        )
        return bool(released)

    async def renew_trip_lock(self, trip_id: UUID | str, token: str, ttl_seconds: int) -> bool:
        return bool(
            await self._client.eval(
                LOCK_RENEW_SCRIPT,
                1,
                self.keys.trip_lock(trip_id),
                token,
                _positive_ttl(ttl_seconds),
            )
        )

    async def consume_rate_limit(
        self,
        scope: str,
        subject: str,
        *,
        limit: int,
        window_seconds: int,
    ) -> RateLimitDecision:
        if limit <= 0:
            raise ValueError("rate limit must be greater than zero")
        window = _positive_ttl(window_seconds)
        window_number = int(self._clock()) // window
        result = await self._client.eval(
            RATE_LIMIT_SCRIPT,
            1,
            self.keys.rate_limit(scope, subject, window_number),
            window,
        )
        count, ttl = (int(result[0]), max(0, int(result[1])))
        return RateLimitDecision(
            allowed=count <= limit,
            count=count,
            limit=limit,
            remaining=max(0, limit - count),
            retry_after_seconds=ttl if count > limit else 0,
        )

    async def save_sms_challenge(self, phone_hash: str, code_digest: str, ttl_seconds: int) -> None:
        ttl = _positive_ttl(ttl_seconds)
        await self._client.set(self.keys.sms_challenge(phone_hash), code_digest, ex=ttl)

    async def consume_sms_challenge(
        self, phone_hash: str, code_digest: str
    ) -> Literal["matched", "invalid", "missing"]:
        result = int(
            await self._client.eval(
                SMS_CHALLENGE_CONSUME_SCRIPT,
                1,
                self.keys.sms_challenge(phone_hash),
                code_digest,
            )
        )
        if result == 1:
            return "matched"
        return "missing" if result == -1 else "invalid"

    async def delete_sms_challenge(self, phone_hash: str) -> None:
        await self._client.delete(self.keys.sms_challenge(phone_hash))

    async def save_auth_session(
        self,
        session_token: str,
        session: Mapping[str, Any],
        ttl_seconds: int,
    ) -> None:
        ttl = _positive_ttl(ttl_seconds)
        user_id = session.get("user_id")
        if not isinstance(user_id, str) or not user_id:
            raise ValueError("auth session requires a user_id")
        await self._client.eval(
            AUTH_SESSION_SAVE_SCRIPT,
            2,
            self.keys.auth_session(session_token),
            self.keys.user_auth_sessions(user_id),
            _json_dump(session),
            ttl,
        )

    async def load_auth_session(self, session_token: str) -> dict[str, Any] | None:
        value = _json_load(await self._client.get(self.keys.auth_session(session_token)))
        return value if isinstance(value, dict) else None

    async def delete_auth_session(self, session_token: str) -> None:
        key = self.keys.auth_session(session_token)
        current = await self.load_auth_session(session_token)
        await self._client.delete(key)
        if current is not None and isinstance(current.get("user_id"), str):
            await self._client.srem(self.keys.user_auth_sessions(current["user_id"]), key)

    async def clear_user_auth_sessions(self, user_id: UUID | str) -> int:
        index_key = self.keys.user_auth_sessions(user_id)
        members = await self._client.smembers(index_key)
        keys = [item.decode("utf-8") if isinstance(item, bytes) else item for item in members]
        removed = await self._client.delete(*keys) if keys else 0
        await self._client.delete(index_key)
        return int(removed)

    async def set_active_generation(
        self,
        trip_id: UUID | str,
        generation_id: UUID | str,
        ttl_seconds: int,
        *,
        anonymous_session_id: str | None = None,
    ) -> None:
        ttl = _positive_ttl(ttl_seconds)
        key = self.keys.active_generation(trip_id)
        await self._client.set(key, str(generation_id), ex=ttl)
        if anonymous_session_id is not None:
            await self._register_anonymous_key(anonymous_session_id, key, ttl)

    async def replace_active_generation(
        self,
        trip_id: UUID | str,
        generation_id: UUID | str,
        ttl_seconds: int,
        *,
        anonymous_session_id: str | None = None,
    ) -> str | None:
        """Atomically replace the active generation and reset the new sequence epoch."""

        ttl = _positive_ttl(ttl_seconds)
        index_key = (
            self.keys.anonymous_index(anonymous_session_id)
            if anonymous_session_id is not None
            else self.keys.unused_anonymous_index(trip_id)
        )
        previous = await self._client.eval(
            ACTIVE_GENERATION_REPLACE_SCRIPT,
            3,
            self.keys.active_generation(trip_id),
            self.keys.generation_sequence(generation_id),
            index_key,
            str(generation_id),
            ttl,
            "1" if anonymous_session_id is not None else "0",
        )
        if isinstance(previous, bytes):
            previous = previous.decode("utf-8")
        return previous if isinstance(previous, str) and previous else None

    async def get_active_generation(self, trip_id: UUID | str) -> str | None:
        value = await self._client.get(self.keys.active_generation(trip_id))
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return value if isinstance(value, str) else None

    async def clear_active_generation_if_matches(
        self, trip_id: UUID | str, generation_id: UUID | str
    ) -> bool:
        result = await self._client.eval(
            ACTIVE_GENERATION_CLEAR_SCRIPT,
            1,
            self.keys.active_generation(trip_id),
            str(generation_id),
        )
        return bool(result)

    async def finish_planner_execution(
        self,
        trip_id: UUID | str,
        generation_id: UUID | str,
        token: str,
        *,
        release_lock: bool,
    ) -> bool:
        """A resumed job keeps its generation ID but owns a different execution lease."""
        return bool(
            await self._client.eval(
                PLANNER_GENERATION_FINISH_SCRIPT,
                2,
                self.keys.trip_lock(trip_id),
                self.keys.active_generation(trip_id),
                token,
                str(generation_id),
                "1" if release_lock else "0",
            )
        )

    async def next_generation_sequence(
        self,
        generation_id: UUID | str,
        ttl_seconds: int,
    ) -> int:
        ttl = _positive_ttl(ttl_seconds)
        return int(
            await self._client.eval(
                GENERATION_SEQUENCE_SCRIPT,
                1,
                self.keys.generation_sequence(generation_id),
                ttl,
            )
        )

    async def clear_anonymous_session(self, session_id: str) -> int:
        index_key = self.keys.anonymous_index(session_id)
        members = await self._client.smembers(index_key)
        keys = [item.decode("utf-8") if isinstance(item, bytes) else item for item in members]
        if keys:
            await self._client.delete(*keys)
        await self._client.delete(index_key)
        return len(keys)

    async def _register_anonymous_key(self, session_id: str, key: str, ttl_seconds: int) -> None:
        index_key = self.keys.anonymous_index(session_id)
        await self._client.eval(ANONYMOUS_INDEX_SCRIPT, 1, index_key, key, ttl_seconds)
